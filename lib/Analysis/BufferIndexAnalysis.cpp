#include "triton/Analysis/BufferIndexAnalysis.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include <optional>

namespace mlir {

static int64_t normalizeModuloOffset(int64_t offset, int64_t modulus) {
  return ((offset % modulus) + modulus) % modulus;
}

/// A buffer index decomposed as `baseValue + constantOffset`, optionally
/// under a known modulus (recorded when the index is wrapped by remsi or
/// the pipeliner's select/cmpi idiom). Two expressions rooted at the
/// same base value with different offsets (modulo the same recorded modulus,
/// when both carry one) provably target different slots.
struct BufferIndexExpr {
  Value baseValue;
  int64_t constantOffset = 0;
  std::optional<int64_t> modulus;

  bool operator==(const BufferIndexExpr &other) const {
    return baseValue == other.baseValue &&
           constantOffset == other.constantOffset && modulus == other.modulus;
  }

  bool isProvablyDifferentFrom(const BufferIndexExpr &other) const {
    if (baseValue != other.baseValue)
      return false;
    if (modulus || other.modulus) {
      if (modulus != other.modulus)
        return false;
      int64_t m = *modulus;
      // Euclidean normalization: make 0 <= offset < m so that
      // negative constants compare correctly against positive ones.
      int64_t a = normalizeModuloOffset(constantOffset, m);
      int64_t b = normalizeModuloOffset(other.constantOffset, m);
      return a != b;
    }
    return constantOffset != other.constantOffset;
  }
};

namespace {

using ValueSubstitutionFn = BufferIndexAnalysis::ValueSubstitutionFn;
using ValueStabilityFn = BufferIndexAnalysis::ValueStabilityFn;

std::optional<BufferIndexValueSubstitution> noValueSubstitution(Value) {
  return std::nullopt;
}

bool allValuesStable(Value) { return true; }

struct SubstitutedValue {
  Value value;
  int64_t constantOffset = 0;
};

std::optional<SubstitutedValue>
applyValueSubstitution(Value value, ValueSubstitutionFn substitute,
                       bool &allowSubstitution) {
  llvm::SmallPtrSet<Value, 8> seen;
  int64_t constantOffset = 0;
  while (allowSubstitution) {
    if (!seen.insert(value).second)
      return std::nullopt;

    auto replacement = substitute(value);
    if (!replacement)
      break;
    if (!replacement->replacement)
      return std::nullopt;

    if (__builtin_add_overflow(constantOffset, replacement->constantOffset,
                               &constantOffset))
      return std::nullopt;
    Value replacementValue = replacement->replacement;
    allowSubstitution = !replacement->stopAfterReplacement;
    if (replacementValue == value)
      break;
    value = replacementValue;
  }

  return SubstitutedValue{value, constantOffset};
}

std::optional<int64_t> getConstantIntValue(Value v) {
  APInt val;
  if (matchPattern(v, m_ConstantInt(&val)))
    return val.getSExtValue();
  return std::nullopt;
}

std::optional<int64_t> getConstantIntValue(Value v,
                                           ValueSubstitutionFn substitute,
                                           bool allowSubstitution) {
  auto resolved = applyValueSubstitution(v, substitute, allowSubstitution);
  if (!resolved)
    return std::nullopt;
  auto constant = getConstantIntValue(resolved->value);
  if (!constant)
    return std::nullopt;
  int64_t result;
  if (__builtin_add_overflow(*constant, resolved->constantOffset, &result))
    return std::nullopt;
  return result;
}

std::optional<BufferIndexExpr>
analyzeBufferIndex(Value indexValue, ValueSubstitutionFn substitute,
                   ValueStabilityFn isStable, bool allowSubstitution);

bool isCFBlockArgProvablyBounded(BlockArgument blockArg, int64_t modulus,
                                 arith::SelectOp selectOp) {
  // For cf-form loops, the loop-carried value is a block argument. Each
  // incoming value must be either an in-range initial value or the matched
  // select that advances the counter.
  Block *header = blockArg.getOwner();
  unsigned argIdx = blockArg.getArgNumber();
  // Entry block arguments have no incoming operands to prove the bound.
  if (header->pred_begin() == header->pred_end())
    return false;

  for (auto predIt = header->pred_begin(), e = header->pred_end(); predIt != e;
       ++predIt) {
    Block *pred = *predIt;
    auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
    if (!branch)
      return false;

    auto operands = branch.getSuccessorOperands(predIt.getSuccessorIndex());
    Value incoming = operands[argIdx];

    if (incoming == selectOp.getResult())
      continue;

    auto c = getConstantIntValue(incoming);
    if (!c || *c < -1 || *c >= modulus)
      return false;
  }

  return true;
}

/// Verify that `base` is provably bounded by -1 <= base < N so that
///   select(cmpi sge/slt (addi(base, 1), N), ...)
/// truly equals (base + 1) % N. The select form returns 0 on the wrap arm
/// rather than (base + 1) - N; outside that range the two expressions
/// diverge and the match would be unsound.
///
/// Constants are checked directly. For cf-form loop-carried counters we prove
/// the bound inductively: incoming values to the loop-header block argument
/// must each be either an in-range constant or the matched select itself.
/// Given -1 <= base < N we have 0 <= base + 1 <= N, and the select maps N to 0
/// and otherwise returns base + 1, so the next value satisfies 0 <= next < N.
bool isBaseProvablyBounded(Value base, int64_t modulus,
                           arith::SelectOp selectOp) {
  assert(modulus > 0);

  if (auto c = getConstantIntValue(base))
    return *c >= -1 && *c < modulus;

  auto blockArg = dyn_cast<BlockArgument>(base);
  if (!blockArg)
    return false;

  return isCFBlockArgProvablyBounded(blockArg, modulus, selectOp);
}

/// Match the one-step modular wrap the pipeliner emits on its iter_arg:
///   select(cmpi sge (addi(base, 1), N), zero, addi(base, 1))
///   select(cmpi slt (addi(base, 1), N), addi(base, 1), zero)
/// Both equal (base + 1) % N only when -1 <= base < N; outside that range
/// the wrap arm would need to be (base + 1) - N, not 0, so accepting the
/// match unconditionally would be unsound. We require C == 1 and verify
/// the range assumption via isBaseProvablyBounded.
std::optional<BufferIndexExpr>
matchModuloPattern(arith::SelectOp selectOp, ValueSubstitutionFn substitute,
                   ValueStabilityFn isStable, bool allowSubstitution) {
  auto cmp = selectOp.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return std::nullopt;

  Value wrapVal, noWrapVal;
  if (cmp.getPredicate() == arith::CmpIPredicate::sge) {
    wrapVal = selectOp.getTrueValue();
    noWrapVal = selectOp.getFalseValue();
  } else if (cmp.getPredicate() == arith::CmpIPredicate::slt) {
    noWrapVal = selectOp.getTrueValue();
    wrapVal = selectOp.getFalseValue();
  } else {
    return std::nullopt;
  }

  auto wrapConst =
      getConstantIntValue(wrapVal, substitute, allowSubstitution);
  if (!wrapConst || *wrapConst != 0)
    return std::nullopt;

  auto addOp = noWrapVal.getDefiningOp<arith::AddIOp>();
  if (!addOp || cmp.getLhs() != addOp.getResult())
    return std::nullopt;

  // Try constant on RHS then LHS (addi is commutative).
  std::optional<int64_t> c =
      getConstantIntValue(addOp.getRhs(), substitute, allowSubstitution);
  Value base = addOp.getLhs();
  if (!c) {
    c = getConstantIntValue(addOp.getLhs(), substitute, allowSubstitution);
    base = addOp.getRhs();
  }
  if (!c || *c != 1)
    return std::nullopt;

  // Modulus must be a positive compile-time constant.
  auto mod = getConstantIntValue(cmp.getRhs(), substitute, allowSubstitution);
  if (!mod || *mod <= 0)
    return std::nullopt;

  // The (base + 1) % N rewrite is only valid for -1 <= base < N.
  if (!isBaseProvablyBounded(base, *mod, selectOp))
    return std::nullopt;

  auto baseExpr =
      analyzeBufferIndex(base, substitute, isStable, allowSubstitution);
  if (!baseExpr)
    return std::nullopt;
  // Nested moduli ((x mod M) + 1) mod N don't reduce to (x + 1) mod N in
  // general; keep the full select as the expression root.
  if (baseExpr->modulus)
    return std::nullopt;
  int64_t offset;
  if (__builtin_add_overflow(baseExpr->constantOffset, *c, &offset))
    return std::nullopt;
  BufferIndexExpr result{baseExpr->baseValue, offset};
  result.modulus = *mod;
  return result;
}

// Slot indices are assumed not to overflow signed integer arithmetic; use a
// wider index type if the pipeline counter can reach the integer range.
std::optional<BufferIndexExpr>
analyzeBufferIndex(Value indexValue, ValueSubstitutionFn substitute,
                   ValueStabilityFn isStable, bool allowSubstitution) {
  auto resolved =
      applyValueSubstitution(indexValue, substitute, allowSubstitution);
  if (!resolved)
    return std::nullopt;
  indexValue = resolved->value;
  auto withSubstitutionOffset = [&](BufferIndexExpr expr)
      -> std::optional<BufferIndexExpr> {
    if (__builtin_add_overflow(expr.constantOffset,
                               resolved->constantOffset,
                               &expr.constantOffset))
      return std::nullopt;
    return expr;
  };

  if (auto c = getConstantIntValue(indexValue))
    return withSubstitutionOffset(BufferIndexExpr{nullptr, *c});

  if (auto addOp = indexValue.getDefiningOp<arith::AddIOp>()) {
    auto composeWithConstant = [&](Value nonConst,
                                   int64_t constant)
        -> std::optional<BufferIndexExpr> {
      auto baseExpr = analyzeBufferIndex(nonConst, substitute, isStable,
                                         allowSubstitution);
      if (!baseExpr)
        return std::nullopt;
      // (x mod N) + C is not represented as (base, offset, mod); keep the
      // full addi as the expression root.
      if (baseExpr->modulus) {
        if (allowSubstitution && !isStable(indexValue))
          return std::nullopt;
        return withSubstitutionOffset(BufferIndexExpr{indexValue, 0});
      }
      int64_t offset;
      if (__builtin_add_overflow(baseExpr->constantOffset, constant, &offset))
        return std::nullopt;
      return withSubstitutionOffset(
          BufferIndexExpr{baseExpr->baseValue, offset});
    };
    if (auto offset =
            getConstantIntValue(addOp.getRhs(), substitute, allowSubstitution))
      return composeWithConstant(addOp.getLhs(), *offset);
    if (auto offset =
            getConstantIntValue(addOp.getLhs(), substitute, allowSubstitution))
      return composeWithConstant(addOp.getRhs(), *offset);
  }

  if (auto selectOp = indexValue.getDefiningOp<arith::SelectOp>())
    if (auto result = matchModuloPattern(selectOp, substitute, isStable,
                                         allowSubstitution))
      return withSubstitutionOffset(*result);

  // arith.remsi(x, N): strip the remainder and record N as the modulus.
  // N must be a positive compile-time constant.
  if (auto remOp = indexValue.getDefiningOp<arith::RemSIOp>()) {
    if (auto mod = getConstantIntValue(remOp.getRhs(), substitute,
                                       allowSubstitution);
        mod && *mod > 0) {
      auto result = analyzeBufferIndex(remOp.getLhs(), substitute, isStable,
                                       allowSubstitution);
      if (!result)
        return std::nullopt;
      if (result->modulus == mod)
        return withSubstitutionOffset(*result);
      // Nested modulus: keep the full remsi as the expression root.
      if (result->modulus) {
        if (allowSubstitution && !isStable(indexValue))
          return std::nullopt;
        return withSubstitutionOffset(BufferIndexExpr{indexValue, 0});
      }
      result->modulus = *mod;
      return withSubstitutionOffset(*result);
    }
  }

  if (allowSubstitution && !isStable(indexValue))
    return std::nullopt;
  return withSubstitutionOffset(BufferIndexExpr{indexValue, 0});
}

std::pair<Value, bool> extractBufferIndex(Value value,
                                          ValueSubstitutionFn substitute,
                                          bool allowSubstitution) {
  // MemDescIndexOp selects a whole slot of a multi-buffered allocation; its
  // index operand identifies the slot. MemDescViewTrait producers (trans,
  // reshape, reinterpret, subslice) are slot-preserving, so we can walk
  // through them to find the underlying MemDescIndexOp.
  Value v = value;
  bool allowNestedSubstitution = allowSubstitution;
  while (true) {
    auto resolved =
        applyValueSubstitution(v, substitute, allowNestedSubstitution);
    if (!resolved)
      return {Value(), allowNestedSubstitution};
    // Memdesc aliases do not carry scalar offsets. Offset substitutions are
    // only meaningful after the selected MemDescIndexOp exposes its index.
    if (resolved->constantOffset != 0)
      return {Value(), allowNestedSubstitution};
    v = resolved->value;
    auto *def = v.getDefiningOp();
    if (!def)
      break;
    if (auto indexOp = dyn_cast<triton::gpu::MemDescIndexOp>(def))
      return {indexOp.getIndex(), allowNestedSubstitution};
    if (!def->hasTrait<OpTrait::MemDescViewTrait>())
      break;
    v = def->getOperand(0);
  }
  return {Value(), allowNestedSubstitution};
}

} // namespace

BufferIndexAnalysis::BufferIndexAnalysis(FunctionOpInterface funcOp)
    : dominanceInfo(funcOp) {}

BufferIndexAnalysis::~BufferIndexAnalysis() = default;

bool areBufferIndicesProvablyDifferent(const AllocationSlice &a,
                                       const AllocationSlice &b) {
  auto *aExpr = a.bufferIndexExpr;
  auto *bExpr = b.bufferIndexExpr;
  if (!aExpr || !bExpr)
    return false;
  return aExpr->isProvablyDifferentFrom(*bExpr);
}

const BufferIndexExpr *BufferIndexAnalysis::intern(BufferIndexExpr expr) {
  // Canonicalize the modular offset so 0 <= constantOffset < m. Equivalent
  // expressions (e.g. offset 0 and offset m) share a single interned entry.
  if (expr.modulus) {
    int64_t m = *expr.modulus;
    expr.constantOffset = normalizeModuloOffset(expr.constantOffset, m);
  }

  for (const auto &existing : expressions)
    if (*existing == expr)
      return existing.get();

  auto owned = std::make_unique<BufferIndexExpr>(expr);
  const BufferIndexExpr *result = owned.get();
  expressions.push_back(std::move(owned));
  return result;
}

AllocationSlice
BufferIndexAnalysis::makeSlice(Value value, Interval<size_t> allocationInterval,
                               Allocation::BufferId bufferId) {
  AllocationSlice slice(value, allocationInterval, bufferId);
  attachBufferIndex(slice, value, noValueSubstitution, allValuesStable);
  return slice;
}

AllocationSlice BufferIndexAnalysis::makeSliceWithValueSubstitution(
    Value value, Interval<size_t> allocationInterval,
    Allocation::BufferId bufferId, ValueSubstitutionFn substitute,
    ValueStabilityFn isStable) {
  AllocationSlice slice(value, allocationInterval, bufferId);
  attachBufferIndex(slice, value, substitute, isStable);
  return slice;
}

void BufferIndexAnalysis::attachBufferIndex(AllocationSlice &slice,
                                            Value value) {
  attachBufferIndex(slice, value, noValueSubstitution, allValuesStable);
}

void BufferIndexAnalysis::attachBufferIndex(AllocationSlice &slice, Value value,
                                            ValueSubstitutionFn substitute,
                                            ValueStabilityFn isStable) {
  auto [index, allowSubstitution] =
      extractBufferIndex(value, substitute, /*allowSubstitution=*/true);
  if (!index)
    return;
  auto expr =
      analyzeBufferIndex(index, substitute, isStable, allowSubstitution);
  if (expr)
    slice.bufferIndexExpr = intern(*expr);
}

bool BufferIndexAnalysis::isBackedgeSuccessor(Operation *terminator,
                                              Block *successor) const {
  if (isa<BranchOpInterface>(terminator))
    return dominanceInfo.dominates(successor, terminator->getBlock());
  return false;
}

void BufferIndexAnalysis::invalidateBufferIndices(BlockInfo &info) const {
  auto rebuild = [](BlockInfo::SliceMapT &m) {
    BlockInfo::SliceMapT rebuilt;
    for (const auto &[slice, ops] : m) {
      AllocationSlice key = slice;
      key.bufferIndexExpr = nullptr;
      rebuilt[key].insert(ops.begin(), ops.end());
    }
    m = std::move(rebuilt);
  };
  rebuild(info.syncReadSlices);
  rebuild(info.syncWriteSlices);
}

} // namespace mlir
