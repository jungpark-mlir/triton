#include "TritonAMDGPUToLLVM/Passes.h"

#include "AsyncUtility.h"
#include "ScheduleMembar.h"
#include "TDMUtility.h"
#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "third_party/amd/include/Analysis/AMDGPUAllocation.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/MathExtras.h"

#include <memory>
#include <optional>

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONAMDGPUMEMBAR
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace mlir::triton {
namespace {

// Synchronous LDS accesses eligible for slice-hazard relaxation.
bool isOrdinarySharedAccess(Operation *op) {
  if (isa<gpu::LocalLoadOp, gpu::LocalStoreOp>(op))
    return true;
  auto alloc = dyn_cast<gpu::LocalAllocOp>(op);
  return alloc && alloc.getSrc();
}

// Shared-memory counterpart of TMemWarpOwnership. Both slice hazards and
// completion rendezvous use this proof and these caches.
class SharedWarpOwnership {
public:
  explicit SharedWarpOwnership(BufferRegionAnalysis &regions)
      : regions(regions) {}

  // Check execution regions, then compare every pair of accessed destinations.
  bool isSameWarp(Operation *lhsOp, Operation *rhsOp, Allocation *allocation) {
    auto function = cast<FunctionOpInterface>(allocation->getOperation());
    if (lhsOp->getParentRegion() != rhsOp->getParentRegion() ||
        lhsOp->getParentOfType<FunctionOpInterface>() != function)
      return false;
    auto [it, inserted] = sameWarps.try_emplace({lhsOp, rhsOp}, false);
    if (!inserted)
      return it->second;
    const AccessList *lhs = getAccesses(lhsOp);
    const AccessList *rhs = getAccesses(rhsOp);
    if (!lhs || !rhs)
      return false;
    // A fused op is several accesses, each with its own destination and hint.
    // Unknown ownership only blocks the proof when that member may overlap.
    for (const Access &a : *lhs)
      for (const Access &b : *rhs)
        if (!a.hasSameWarp(b))
          return false;
    return it->second = true;
  }

  // Require a proven ownership map for every access made by the operation.
  bool hasKnownOwnership(Operation *op) {
    const auto *accesses = getAccesses(op);
    return accesses && llvm::all_of(*accesses, [](const Access &access) {
             return access.warps.has_value();
           });
  }

private:
  struct Access {
    // LDS element offset -> warp. Footprints themselves are in bytes.
    std::optional<LinearLayout> warps;
    unsigned elementBytes;
    const BufferRegionFootprint *footprint;

    // Compare owners on overlapping bytes, with a fast path for identical maps.
    bool hasSameWarp(const Access &other) const {
      if (!mayOverlap(footprint, other.footprint))
        return true;
      if (!warps || !other.warps || !footprint || !other.footprint)
        return false;
      for (const auto &a : footprint->regionInfo.views)
        for (const auto &b : other.footprint->regionInfo.views) {
          if (!a.allocationFrame || a.allocationFrame != b.allocationFrame)
            return false;
          if (a.region.baseOffset == b.region.baseOffset &&
              elementBytes == other.elementBytes && warps == other.warps)
            continue;
          for (const auto &[cta, addresses] : a.region.ctaAddresses)
            for (const auto &[otherCTA, otherAddresses] :
                 b.region.ctaAddresses) {
              if (cta != otherCTA)
                continue;
              // Storage bases use integer addition, not layout XOR.
              // Compare bytes across different ranks and element widths.
              for (uint32_t address : addresses.intersection(otherAddresses)) {
                auto warp = getWarp(address - a.region.baseOffset);
                auto otherWarp = other.getWarp(address - b.region.baseOffset);
                if (!warp || !otherWarp || warp != otherWarp)
                  return false;
              }
            }
        }
      return true;
    }

    // Map a relative byte offset to its owner; out-of-range offsets fail proof.
    std::optional<uint32_t> getWarp(uint32_t byteOffset) const {
      auto offset = *warps->getInDimNames().begin();
      uint32_t element = byteOffset / elementBytes;
      if (element >= warps->getInDimSize(offset))
        return std::nullopt;
      return warps->apply({{offset, int32_t(element)}}).front().second;
    }
  };
  using AccessList = SmallVector<Access>;

  // Accept only descriptor chains whose physical offsets this proof supports.
  bool hasSupportedOrigin(Value mem) {
    auto found = origins.find(mem);
    if (found != origins.end())
      return found->second;
    bool supported = false;
    if (mem.getDefiningOp<gpu::LocalAllocOp>())
      supported = true;
    else if (auto index = mem.getDefiningOp<gpu::MemDescIndexOp>())
      supported = hasSupportedOrigin(index.getSrc());
    else if (auto reshape = mem.getDefiningOp<gpu::MemDescReshapeOp>())
      supported = hasSupportedOrigin(reshape.getSrc());
    else if (auto trans = mem.getDefiningOp<gpu::MemDescTransOp>())
      supported = hasSupportedOrigin(trans.getSrc());
    else if (auto select = mem.getDefiningOp<arith::SelectOp>())
      supported = hasSupportedOrigin(select.getTrueValue()) &&
                  hasSupportedOrigin(select.getFalseValue());
    // Subslice offsets compose by XOR; keep them (and unknown/block argument
    // origins) conservative, even when hidden behind other views.
    origins[mem] = supported;
    return supported;
  }

  // Derive LDS-offset -> warp from the access layout. Retain the footprint even
  // when ownership cannot be proven, so disjoint accesses can still be excluded.
  Access computeAccess(Operation *op, Value mem,
                       std::optional<LinearLayout> layout) {
    auto memTy = cast<gpu::MemDescType>(mem.getType());
    auto function = op->getParentOfType<FunctionOpInterface>();
    auto *footprint = regions.getFootprint(mem, function);
    Access result{std::nullopt, memTy.getElementTypeBitWidth() / 8, footprint};
    if (!layout || memTy.getElementTypeBitWidth() < 8 ||
        gpu::isPaddedEncoding(memTy.getEncoding()) ||
        isa<gpu::PartitionedSharedEncodingAttr>(memTy.getEncoding()) ||
        !hasSupportedOrigin(mem) || !footprint ||
        footprint->regionInfo.views.empty())
      return result;
    for (const auto &view : footprint->regionInfo.views)
      if (view.affineOffset || view.affinePartitionOffset ||
          view.affineCTAOffset || !view.partitionBases.empty())
        return result;

    auto *ctx = op->getContext();
    auto warp = StringAttr::get(ctx, "warp");
    auto offset = StringAttr::get(ctx, "offset");
    auto block = StringAttr::get(ctx, "block");
    auto padding = StringAttr::get(ctx, "padding");
    // A zero warp basis permits another warp to access the same elements.
    // Predication of inactive TDM warps is not modeled here.
    if (!layout->hasInDim(warp) || layout->getFreeVariableMasks().lookup(warp))
      return result;
    auto shared = gpu::toLinearLayoutIgnoringPadding(memTy);
    auto accesses = invertAndComposeLocal(shared, *layout, {block});
    // Complete the image after the real inputs so the inverse prefers them.
    auto completion =
        LinearLayout::identity1D(accesses.getOutDimSize(offset), padding,
                                 offset) *
        LinearLayout::identity1D(accesses.getOutDimSize(block), padding, block);
    auto warps = accesses.concatIns(completion)
                     .pseudoinvert()
                     .sublayout({offset}, {warp});
    // As in TMEM, a non-surjective inverse hides other owning warps.
    if (warps.isSurjective())
      result.warps = std::move(warps);
    return result;
  }

  // Reuse lowering's message/warp layout for all-warp, single-instruction copies.
  std::optional<LinearLayout> getTDMLayout(Operation *op, Value mem,
                                           std::optional<uint32_t> hint) {
    auto ty = cast<gpu::MemDescType>(mem.getType());
    int numWarps = gpu::lookupNumWarps(op);
    // Full hints have the same mapping as an unhinted copy. Proper subsets
    // require a proof of consumer participation, which is not modeled here.
    if ((hint && llvm::popcount(*hint) != numWarps) ||
        gpu::isPaddedEncoding(ty.getEncoding()) ||
        isa<gpu::PartitionedSharedEncodingAttr>(ty.getEncoding()))
      return std::nullopt;
    auto [warpsPerCTA, numInstructions] =
        LLVM::AMD::distributeTDMWarpsAlignToPartition(ty.getShape(), numWarps,
                                                      ty.getEncoding());
    unsigned distributedWarps = 1;
    for (unsigned count : warpsPerCTA)
      distributedWarps *= count;
    if (numInstructions != 1 || distributedWarps != numWarps)
      return std::nullopt;
    auto cga = gpu::getCGALayout(ty.getEncoding()).getLinearLayout();
    return gpu::getTDMLinearLayout(ty.getShape(), warpsPerCTA, cga, numWarps,
                                   hint);
  }

  // Dispatch by op kind and cache each destination's footprint and ownership.
  const AccessList *getAccesses(Operation *op) {
    auto [it, inserted] = accessesByOp.try_emplace(op);
    if (!inserted)
      return it->second.get();
    if (gpu::lookupNumCTAs(op) != 1)
      return nullptr;
    auto accesses = std::make_unique<AccessList>();
    auto addOrdinary = [&](Value mem, RankedTensorType regTy) {
      std::optional<LinearLayout> layout;
      if (regTy.getEncoding())
        layout = gpu::toLinearLayout(regTy);
      accesses->push_back(computeAccess(op, mem, std::move(layout)));
    };
    if (auto load = dyn_cast<gpu::LocalLoadOp>(op))
      addOrdinary(load.getSrc(), load.getType());
    else if (auto store = dyn_cast<gpu::LocalStoreOp>(op))
      addOrdinary(store.getDst(), store.getSrc().getType());
    else if (auto alloc = dyn_cast<gpu::LocalAllocOp>(op)) {
      if (!alloc.getSrc())
        return nullptr;
      addOrdinary(alloc.getResult(), alloc.getSrc().getType());
    } else if (auto copy = dyn_cast<gpu::AsyncCopyGlobalToLocalOp>(op))
      addOrdinary(copy.getResult(), copy.getSrc().getType());
    else if (auto copy = dyn_cast<amdgpu::BufferLoadToLocalOp>(op))
      addOrdinary(copy.getDest(),
                  cast<RankedTensorType>(copy.getOffsets().getType()));
    else if (auto copy = dyn_cast<amdgpu::AsyncTDMCopyGlobalToLocalOp>(op)) {
      if (copy.getBarrier())
        return nullptr;
      accesses->push_back(computeAccess(
          op, copy.getResult(),
          getTDMLayout(op, copy.getResult(), copy.getWarpUsedHint())));
    } else if (auto fused =
                   dyn_cast<amdgpu::AsyncTDMFusedCopyGlobalToLocalOp>(op)) {
      for (auto [mem, hint] :
           llvm::zip_equal(fused.getDests(), fused.getWarpUsedHints()))
        accesses->push_back(
            computeAccess(op, mem, getTDMLayout(op, mem, uint32_t(hint))));
    } else
      return nullptr;
    it->second = std::move(accesses);
    return it->second.get();
  }

  BufferRegionAnalysis &regions;
  DenseMap<Value, bool> origins;
  DenseMap<Operation *, std::unique_ptr<AccessList>> accessesByOp;
  DenseMap<std::pair<Operation *, Operation *>, bool> sameWarps;
};

class AMDMembarAnalysis : public MembarAnalysis {
public:
  // Build function-wide writer candidates for later completion checks.
  AMDMembarAnalysis(Allocation &allocation, MembarFilterFn filter,
                    BufferRegionAnalysis &regions,
                    SharedWarpOwnership &ownership, bool enableWarpLocal)
      : MembarAnalysis(allocation, std::move(filter), regions),
        ownership(ownership), enableWarpLocal(enableWarpLocal) {
    if (!enableWarpLocal)
      return;
    auto function = cast<FunctionOpInterface>(allocation.getOperation());
    function.walk([&](Operation *op) {
      hasCalls |= isa<CallOpInterface>(op);
      if (!op->hasTrait<OpTrait::GlobalToLocalCopyTrait>())
        return;
      bool foundWrite = false;
      for (const auto &access : getMemoryAccesses(op)) {
        if (!access.isShared() || !access.isWrite)
          continue;
        writers.push_back({op, regions.getFootprint(access.value, function)});
        foundWrite = true;
      }
      if (!foundWrite)
        writers.push_back({op, nullptr});
    });
  }

protected:
  // Relax shared completion only when other thread demands permit it and all
  // overlapping async writers have the same ownership as the upcoming accesses.
  bool requiresThreadSync(const BlockInfo &pending,
                          const BlockInfo &effects) override {
    if (!MembarAnalysis::requiresThreadSync(pending, effects))
      return false;
    if (!enableWarpLocal ||
        pending.threadSync.completion != CompletionSync::Shared || hasCalls)
      return true;
    // Other synchronization reasons must hold even if ownership matches.
    BlockInfo withoutCompletion;
    withoutCompletion.threadSync = pending.threadSync;
    withoutCompletion.threadSync.completion = CompletionSync::None;
    if (MembarAnalysis::requiresThreadSync(withoutCompletion, effects) ||
        (effects.syncReadSlices.empty() && effects.syncWriteSlices.empty()))
      return true;
    for (auto *slices : {&effects.syncReadSlices, &effects.syncWriteSlices})
      for (const auto &[slice, ops] : *slices)
        if (!isCompletionWarpLocal(slice, ops))
          return true;
    // Retain the completion state for any later cross-warp consumer.
    return false;
  }

private:
  // Prove every consumer agrees with every overlapping async writer.
  bool isCompletionWarpLocal(const AllocationSlice &slice,
                             const std::set<Operation *> &ops) {
    if (!slice.physicalFootprint || slice.argumentIndex || ops.empty())
      return false;
    for (Operation *op : ops)
      if (!ownership.hasKnownOwnership(op))
        return false;
    // An earlier barrier may have cleared pending writes while a copy was
    // outstanding. Include writers from all iterations and execution regions.
    for (const auto &[writer, footprint] : writers) {
      if (!mayOverlap(slice.physicalFootprint, footprint))
        continue;
      for (Operation *op : ops)
        if (!ownership.isSameWarp(writer, op, &allocation))
          return false;
    }
    return true;
  }

  SharedWarpOwnership &ownership;
  bool enableWarpLocal;
  SmallVector<std::pair<Operation *, const BufferRegionFootprint *>> writers;
  bool hasCalls = false;
};

struct TritonAMDGPUMembar
    : public impl::TritonAMDGPUMembarBase<TritonAMDGPUMembar> {
  using TritonAMDGPUMembarBase::TritonAMDGPUMembarBase;

  // Build physical footprints and share one ownership proof between both checks.
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    AMD::TargetInfo targetInfo(gfxArch);
    if (targetInfo.getISAFamily() == amdgpu::ISAFamily::Unknown) {
      mod.emitError("unsupported target: '") << gfxArch << "'";
      signalPassFailure();
      return;
    }

    auto allocationFn = [&targetInfo](Operation *op) {
      return AMD::AMDAllocationAnalysisScratchSizeFn(op, targetInfo);
    };
    ModuleAllocation allocation(mod, allocationFn,
                                targetInfo.getSharedMemoryPartitionSize());

    if (targetInfo.requiresAliasInfoForAsyncOps())
      AMD::annotateLocalLoadsSyncedViaAsyncWait(mod);

    // LLVM AMDGPUUsage, "Memory Model GFX6-GFX9" and "Memory Model GFX942":
    // LDS operations are wavefront-wide, uncached, and ordered within a wave.
    // Async-mark waits complete that wave's copies; they do not join waves.
    // LLVM AMDGPUUsage, "Memory Model GFX125x": non-async LDS operations
    // are wavefront-wide and ordered; async/tensor completion is explicit.
    // CDNA5 uses per-wave asynccnt/tensorcnt waits for ordinary/TDM copies.
    // Keep those waits; only the cross-wave rendezvous is relaxed.
    bool enableWarpLocal =
        targetInfo.useAsyncMarks() || targetInfo.supportsTDM();
    auto solver = createDataFlowSolver();
    auto *regions = solver->load<BufferRegionAnalysis>(
        BufferRegionAnalysis::Mode::AllMemory, &allocation);
    if (failed(solver->initializeAndRun(mod))) {
      signalPassFailure();
      return;
    }
    SharedWarpOwnership ownership(*regions);
    // Slice hazards are checked independently of the completion override above.
    auto filter = [&](Operation *lhs, Operation *rhs, bool lhsIsRead,
                      bool rhsIsRead, Allocation *alloc,
                      const AllocationSlice &lhsSlice,
                      const AllocationSlice &rhsSlice) {
      if (AMD::membarFilter(lhs, rhs, lhsIsRead, rhsIsRead, alloc, lhsSlice,
                            rhsSlice))
        return true;
      if (!enableWarpLocal || !isOrdinarySharedAccess(rhs))
        return false;
      // CDNA5 async copies use explicit hardware counter waits, including
      // tokenless TDM waits. Ownership removes only the rendezvous; programs
      // must still wait for their async copies. Keep ordinary -> async-refill
      // hazards conservative, as on CDNA3/CDNA4.
      bool supportedProducer =
          isOrdinarySharedAccess(lhs) ||
          (targetInfo.supportsTDM() &&
           lhs->hasTrait<OpTrait::GlobalToLocalCopyTrait>());
      return supportedProducer && ownership.isSameWarp(lhs, rhs, alloc);
    };
    AMDMembarAnalysis::runModule(mod, [&](FunctionOpInterface function) {
      return AMDMembarAnalysis(*allocation.getFuncData(function), filter,
                               *regions, ownership, enableWarpLocal);
    });
    // Reschedule local rendezvous while waits and memory effects are explicit,
    // before ConSan and LLVM lowering.
    AMD::scheduleMembarBarriers(mod);
  }
};

} // namespace
} // namespace mlir::triton
