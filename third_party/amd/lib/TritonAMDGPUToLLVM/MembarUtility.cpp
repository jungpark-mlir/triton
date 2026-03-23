#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "../../backend/include/TDMCommon.h"

namespace mlir::triton::AMD {
namespace {
// Returns true if one of the operands is a LocalLoad synced via AsyncWait.
bool filterAsyncLocalLoadsDependencies(Operation *op1, Operation *op2,
                                       Allocation *allocation) {
  auto isAsyncLoad = [](Operation *op) {
    return llvm::isa<triton::gpu::AsyncCopyGlobalToLocalOp,
                     triton::amdgpu::BufferLoadToLocalOp,
                     triton::amdgpu::AsyncTDMCopyLocalToGlobalOp>(op);
  };
  auto isLocalLoadWithAsyncWaitToken = [](Operation *op) {
    auto localLoad = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op);
    return localLoad && isSyncedViaAsyncWait(localLoad);
  };
  auto getMemdescValue = [](Operation *op) -> Value {
    return llvm::TypeSwitch<Operation *, Value>(op)
        .Case<triton::amdgpu::BufferLoadToLocalOp>(
            [](auto op) { return op.getDest(); })
        .Case<triton::gpu::AsyncCopyGlobalToLocalOp>(
            [](auto op) { return op.getResult(); })
        .Case<triton::gpu::LocalLoadOp>([](auto op) { return op.getSrc(); })
        .Default([](Operation *) { return Value(); });
  };

  // Early return if neither or both operands are an AsyncLoad
  if (isAsyncLoad(op1) == isAsyncLoad(op2)) {
    return false;
  }

  Value op1Memdesc = getMemdescValue(op1);
  Value op2Memdesc = getMemdescValue(op2);
  if (!op1Memdesc || !op2Memdesc)
    return false;
  auto op1BufferIds = allocation->getAllBufferIdsWithAliases(op1Memdesc);
  auto op2BufferIds = allocation->getAllBufferIdsWithAliases(op2Memdesc);

  // Check if operations access the same buffer
  bool sameBuffer = llvm::any_of(
      op1BufferIds, [&](auto id) { return op2BufferIds.count(id); });

  if (!sameBuffer)
    return false;

  return isLocalLoadWithAsyncWaitToken(op1) ||
         isLocalLoadWithAsyncWaitToken(op2);
}

bool filterLDSMemoryBarriersDependencies(Operation *op1, Operation *op2) {
  auto isLDSMemoryBarrierOp = [](Operation *op) {
    return llvm::isa<triton::amdgpu::InitBarrierOp,
                     triton::amdgpu::ArriveBarrierOp,
                     triton::amdgpu::AsyncCopyMbarrierArriveOp,
                     triton::amdgpu::WaitBarrierOp>(op);
  };

  return (isLDSMemoryBarrierOp(op1) && isLDSMemoryBarrierOp(op2));
}

// ---------------------------------------------------------------------------
// Warp-local shared memory access — barrier suppression
// ---------------------------------------------------------------------------
//
// When a shared memory layout guarantees that each warp accesses a disjoint
// set of byte addresses, writes by one warp are invisible to every other warp.
// In that case, a CTA-wide barrier (s_barrier) between a write and a
// subsequent read is unnecessary — the access is warp-local.
//
// This filter checks whether a (writer, reader) operation pair has warp-local
// access patterns, and if so, tells the membar analysis to suppress the
// barrier it would otherwise insert.
//
// The check compares the warpsPerCTA distribution on both sides: if both the
// write and read distribute warps identically across tensor dimensions, each
// warp owns the same partition of tensor elements on both sides. Since all
// Triton shared memory encodings (padded, swizzled, linear, rotating) are
// bijections from tensor elements to byte addresses, identical tensor-space
// partitioning implies disjoint byte-address partitioning. Swizzling permutes
// which byte offset a (row, col) pair maps to but never collapses two
// distinct elements onto the same address.
//
// Trailing 1s in warpsPerCTA are stripped before comparison to handle rank
// changes from memdesc_reshape/trans (e.g. [4,1] == [4,1,1]).
//
// A tile element count check guards against false positives when different-
// shaped allocations reuse the same physical memory (dealloc/realloc):
// matching warpsPerCTA on different-shaped tiles would partition different
// byte ranges per warp.
//
// Currently scoped to operation pairs involving AsyncTDMCopyGlobalToLocalOp
// to avoid changing barrier behavior for existing non-TDM code paths.
// ---------------------------------------------------------------------------

// Get the distributed tensor type (register side) from an op.
static RankedTensorType getDistributedType(Operation *op) {
  return llvm::TypeSwitch<Operation *, RankedTensorType>(op)
      .Case<triton::gpu::LocalLoadOp>(
          [](auto op) {
            return cast<RankedTensorType>(op.getResult().getType());
          })
      .Case<triton::gpu::LocalStoreOp>(
          [](auto op) {
            return cast<RankedTensorType>(op.getSrc().getType());
          })
      .Case<triton::gpu::LocalAllocOp>(
          [](auto op) -> RankedTensorType {
            if (op.getSrc())
              return cast<RankedTensorType>(op.getSrc().getType());
            return RankedTensorType();
          })
      .Default([](Operation *) { return RankedTensorType(); });
}

// Get TDM warpsPerCTA from the tensor descriptor's block shape.
static SmallVector<unsigned>
getTDMWarpsPerCTA(triton::amdgpu::AsyncTDMCopyGlobalToLocalOp tdmOp) {
  auto descTy = tdmOp.getDesc().getType();
  auto blockShape =
      SmallVector<int64_t>(descTy.getBlockType().getShape());
  int numWarps = triton::gpu::lookupNumWarps(tdmOp);
  int numDims = blockShape.size();
  SmallVector<int> warpsRaw(numDims);
  tdmGetWarpDistribution(blockShape.data(), numDims, numWarps, warpsRaw.data());
  return SmallVector<unsigned>(warpsRaw.begin(), warpsRaw.end());
}

// Get warpsPerCTA for a register-side op from its distributed encoding.
static SmallVector<unsigned>
getRegWarpsPerCTA(Operation *op) {
  auto regTy = getDistributedType(op);
  if (!regTy)
    return {};
  return triton::gpu::getWarpsPerCTA(regTy);
}

// Strip trailing 1s so warpsPerCTA vectors of different ranks compare equal
// when the extra dimensions carry no warps. memdesc_reshape/trans can change
// the tensor rank without altering the physical warp distribution.
//
// Example: TDM writes a 2D tile with warpsPerCTA = [4, 1]. After
// reshape+trans, local_load sees a 3D view with warpsPerCTA = [4, 1, 1].
// Both distribute all 4 warps along the row/batch dimension.
// After normalization: [4] == [4] -> match.
static SmallVector<unsigned>
normalizeWarpsPerCTA(SmallVector<unsigned> warps) {
  while (warps.size() > 1 && warps.back() == 1)
    warps.pop_back();
  return warps;
}

// Get the total number of tile elements for an op.
static int64_t getTileElements(Operation *op) {
  if (auto tdm = dyn_cast<triton::amdgpu::AsyncTDMCopyGlobalToLocalOp>(op))
    return product(tdm.getDesc().getType().getBlockType().getShape());
  auto regTy = getDistributedType(op);
  return regTy ? product(regTy.getShape()) : 0;
}

// Returns true if both ops distribute warps identically over same-sized tiles,
// proving each warp accesses a disjoint set of shared memory addresses on both
// sides. The tile element count check guards against false positives when
// different-shaped allocations reuse the same physical memory (dealloc/realloc).
static bool hasMatchingWarpDistribution(Operation *op1, Operation *op2) {
  SmallVector<unsigned> warps1, warps2;

  if (auto tdm = dyn_cast<triton::amdgpu::AsyncTDMCopyGlobalToLocalOp>(op1))
    warps1 = getTDMWarpsPerCTA(tdm);
  else
    warps1 = getRegWarpsPerCTA(op1);

  if (auto tdm = dyn_cast<triton::amdgpu::AsyncTDMCopyGlobalToLocalOp>(op2))
    warps2 = getTDMWarpsPerCTA(tdm);
  else
    warps2 = getRegWarpsPerCTA(op2);

  if (warps1.empty() || warps2.empty())
    return false;

  int64_t elems1 = getTileElements(op1);
  int64_t elems2 = getTileElements(op2);
  if (elems1 == 0 || elems2 == 0 || elems1 != elems2)
    return false;

  return normalizeWarpsPerCTA(warps1) == normalizeWarpsPerCTA(warps2);
}

// Returns true if the barrier between op1 and op2 can be suppressed because
// their shared memory accesses are warp-local.
bool filterWarpLocalAccesses(Operation *op1, Operation *op2) {
  if (!isa<triton::amdgpu::AsyncTDMCopyGlobalToLocalOp>(op1) &&
      !isa<triton::amdgpu::AsyncTDMCopyGlobalToLocalOp>(op2))
    return false;

  return hasMatchingWarpDistribution(op1, op2);
}

} // namespace

bool membarFilter(Operation *op1, Operation *op2, bool /*op1IsRead*/,
                  bool /*op2IsRead*/, Allocation *allocation) {
  return (filterAsyncLocalLoadsDependencies(op1, op2, allocation) ||
          filterLDSMemoryBarriersDependencies(op1, op2) ||
          filterWarpLocalAccesses(op1, op2));
}
} // namespace mlir::triton::AMD
