/// Presburger-arithmetic-based index disjointness analysis for membar
/// elimination.  Given two buffer index Values (typically from
/// MemDescIndexOp), this file builds an IntegerPolyhedron that encodes
/// the arith SSA chains producing each index, adds the query "idx1 == idx2",
/// and checks whether the resulting system has any integer solution.
/// If not, the indices are provably different and no CTA-wide barrier
/// is needed between accesses to those buffer slots.
///
/// See also: docs/membar-presburger.md

#include "triton/Analysis/PresburgerIndexAnalysis.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <chrono>

#define DEBUG_TYPE "presburger-index-analysis"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::presburger;

namespace {

/// Builds an IntegerPolyhedron constraint system by walking arith SSA
/// def-use chains from buffer index values.  Each arith op is translated
/// into linear / modular constraints; unrecognized ops produce
/// unconstrained variables (safe: the solver cannot prove disjointness
/// through an opaque value, so it falls back to conservative).
struct IndexConstraintBuilder {
  IntegerPolyhedron poly;
  DenseMap<Value, unsigned> valueToSetDimIdx;

  IndexConstraintBuilder()
      : poly(PresburgerSpace::getSetSpace(/*numDims=*/0, /*numSymbols=*/0,
                                          /*numLocals=*/0)) {}

  // ---------------------------------------------------------------------------
  // Variable management — all positions are computed dynamically via
  // getVarKindOffset so they remain valid after insertions.
  // ---------------------------------------------------------------------------

  unsigned getSetDimCol(unsigned setDimIdx) const {
    return poly.getVarKindOffset(VarKind::SetDim) + setDimIdx;
  }

  unsigned getLocalCol(unsigned localIdx) const {
    return poly.getVarKindOffset(VarKind::Local) + localIdx;
  }

  unsigned getConstantCol() const { return poly.getNumCols() - 1; }

  SmallVector<int64_t> zeroRow() const {
    return SmallVector<int64_t>(poly.getNumCols(), 0);
  }

  unsigned allocateSetDimVar() {
    unsigned idx = poly.getNumVarKind(VarKind::SetDim);
    poly.appendVar(VarKind::SetDim);
    return idx;
  }

  unsigned getOrCreateVar(Value v) {
    auto [it, inserted] = valueToSetDimIdx.try_emplace(v, 0);
    if (inserted) {
      it->second = allocateSetDimVar();
      LDBG("  var[" << it->second << "] ← unconstrained (opaque value)");
    }
    return it->second;
  }

  // ---------------------------------------------------------------------------
  // Top-level encoder — returns a SetDim-relative index for the value.
  // ---------------------------------------------------------------------------

  unsigned encodeIndex(Value index) {
    auto it = valueToSetDimIdx.find(index);
    if (it != valueToSetDimIdx.end())
      return it->second;

    Operation *defOp = index.getDefiningOp();
    if (!defOp)
      return getOrCreateVar(index);

    return llvm::TypeSwitch<Operation *, unsigned>(defOp)
        .Case<arith::ConstantIntOp>(
            [&](auto op) { return encodeConstant(index, op.value()); })
        .Case<arith::AddIOp>(
            [&](auto op) { return encodeAdd(index, op); })
        .Case<arith::SubIOp>(
            [&](auto op) { return encodeSub(index, op); })
        .Case<arith::MulIOp>(
            [&](auto op) { return encodeMul(index, op); })
        .Case<arith::RemSIOp>(
            [&](auto op) { return encodeRemS(index, op); })
        .Case<arith::RemUIOp>(
            [&](auto op) { return encodeRemU(index, op); })
        .Case<arith::AndIOp>(
            [&](auto op) { return encodeAnd(index, op); })
        .Case<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp,
              arith::ExtUIOp, arith::TruncIOp>(
            [&](auto op) { return encodeCast(index, op.getIn()); })
        .Default([&](Operation *) { return getOrCreateVar(index); });
  }

  // ---------------------------------------------------------------------------
  // Per-op encoders
  // ---------------------------------------------------------------------------

  unsigned encodeConstant(Value v, int64_t val) {
    unsigned idx = allocateSetDimVar();
    valueToSetDimIdx[v] = idx;

    auto eq = zeroRow();
    eq[getSetDimCol(idx)] = 1;
    eq[getConstantCol()] = -val;
    poly.addEquality(eq);

    LDBG("  var[" << idx << "] = " << val << "  (constant)");
    return idx;
  }

  unsigned encodeAdd(Value v, arith::AddIOp addOp) {
    unsigned lhsIdx = encodeIndex(addOp.getLhs());
    unsigned rhsIdx = encodeIndex(addOp.getRhs());
    unsigned resultIdx = allocateSetDimVar();
    valueToSetDimIdx[v] = resultIdx;

    // result = lhs + rhs  →  result - lhs - rhs = 0
    auto eq = zeroRow();
    eq[getSetDimCol(resultIdx)] = 1;
    eq[getSetDimCol(lhsIdx)] = -1;
    eq[getSetDimCol(rhsIdx)] = -1;
    poly.addEquality(eq);

    LDBG("  var[" << resultIdx << "] = var[" << lhsIdx << "] + var["
                  << rhsIdx << "]  (addi)");
    return resultIdx;
  }

  unsigned encodeSub(Value v, arith::SubIOp subOp) {
    unsigned lhsIdx = encodeIndex(subOp.getLhs());
    unsigned rhsIdx = encodeIndex(subOp.getRhs());
    unsigned resultIdx = allocateSetDimVar();
    valueToSetDimIdx[v] = resultIdx;

    // result = lhs - rhs  →  result - lhs + rhs = 0
    auto eq = zeroRow();
    eq[getSetDimCol(resultIdx)] = 1;
    eq[getSetDimCol(lhsIdx)] = -1;
    eq[getSetDimCol(rhsIdx)] = 1;
    poly.addEquality(eq);

    LDBG("  var[" << resultIdx << "] = var[" << lhsIdx << "] - var["
                  << rhsIdx << "]  (subi)");
    return resultIdx;
  }

  unsigned encodeMul(Value v, arith::MulIOp mulOp) {
    auto lhsConst = mulOp.getLhs().getDefiningOp<arith::ConstantIntOp>();
    auto rhsConst = mulOp.getRhs().getDefiningOp<arith::ConstantIntOp>();

    if (!lhsConst && !rhsConst) {
      LDBG("  muli with two dynamic operands — unconstrained");
      return getOrCreateVar(v);
    }

    int64_t factor;
    Value dynamicOperand;
    if (lhsConst) {
      factor = lhsConst.value();
      dynamicOperand = mulOp.getRhs();
    } else {
      factor = rhsConst.value();
      dynamicOperand = mulOp.getLhs();
    }

    unsigned operandIdx = encodeIndex(dynamicOperand);
    unsigned resultIdx = allocateSetDimVar();
    valueToSetDimIdx[v] = resultIdx;

    // result = factor * operand
    auto eq = zeroRow();
    eq[getSetDimCol(resultIdx)] = 1;
    eq[getSetDimCol(operandIdx)] = -factor;
    poly.addEquality(eq);

    LDBG("  var[" << resultIdx << "] = " << factor << " * var["
                  << operandIdx << "]  (muli)");
    return resultIdx;
  }

  /// Encode r = dividend mod N using floor division.
  /// Creates local variable q = floor(dividend / N) with constraints
  /// N*q <= dividend < N*(q+1), then defines r = dividend - N*q.
  ///
  /// addLocalFloorDiv expects a coefficient vector in the column layout
  /// [SetDims... | Locals... | constant], representing the numerator
  /// expression.  We set only the dividend's SetDim column to 1 and
  /// the divisor to N, encoding q = floor(dividend / N).  The library
  /// appends a new Local variable for q and adds two inequalities:
  ///   N*q <= dividend   and   dividend < N*(q+1)
  unsigned encodeRemainder(Value v, Value dividendVal, int64_t N) {
    unsigned dividendIdx = encodeIndex(dividendVal);

    auto dividendCoeffs = zeroRow();
    dividendCoeffs[getSetDimCol(dividendIdx)] = 1;

    poly.addLocalFloorDiv(dividendCoeffs, N);
    unsigned qLocalIdx = poly.getNumVarKind(VarKind::Local) - 1;

    // Create SetDim for the remainder.
    unsigned rIdx = allocateSetDimVar();
    valueToSetDimIdx[v] = rIdx;

    // r = dividend - N*q
    auto eq = zeroRow();
    eq[getSetDimCol(rIdx)] = 1;
    eq[getSetDimCol(dividendIdx)] = -1;
    eq[getLocalCol(qLocalIdx)] = N;
    poly.addEquality(eq);

    LDBG("  var[" << rIdx << "] = var[" << dividendIdx << "] mod " << N
                  << "  (quotient local[" << qLocalIdx << "])");
    return rIdx;
  }

  unsigned encodeRemS(Value v, arith::RemSIOp remOp) {
    auto modConst = remOp.getRhs().getDefiningOp<arith::ConstantIntOp>();
    if (!modConst) {
      LDBG("  remsi with non-constant modulus — unconstrained");
      return getOrCreateVar(v);
    }
    int64_t N = modConst.value();
    if (N <= 0) {
      LDBG("  remsi with non-positive modulus " << N << " — unconstrained");
      return getOrCreateVar(v);
    }
    // remsi has C semantics (result sign matches dividend).
    // The floor-division encoding produces 0 <= r < N, which matches
    // remsi only when the dividend is non-negative.  Buffer phase
    // counters are always non-negative so this is safe for our use case.
    return encodeRemainder(v, remOp.getLhs(), N);
  }

  unsigned encodeRemU(Value v, arith::RemUIOp remOp) {
    auto modConst = remOp.getRhs().getDefiningOp<arith::ConstantIntOp>();
    if (!modConst) {
      LDBG("  remui with non-constant modulus — unconstrained");
      return getOrCreateVar(v);
    }
    int64_t N = modConst.value();
    if (N <= 0) {
      LDBG("  remui with non-positive modulus " << N << " — unconstrained");
      return getOrCreateVar(v);
    }
    return encodeRemainder(v, remOp.getLhs(), N);
  }

  unsigned encodeAnd(Value v, arith::AndIOp andOp) {
    auto rhsConst = andOp.getRhs().getDefiningOp<arith::ConstantIntOp>();
    auto lhsConst = andOp.getLhs().getDefiningOp<arith::ConstantIntOp>();

    int64_t mask;
    Value dynamicOperand;
    if (rhsConst) {
      mask = rhsConst.value();
      dynamicOperand = andOp.getLhs();
    } else if (lhsConst) {
      mask = lhsConst.value();
      dynamicOperand = andOp.getRhs();
    } else {
      LDBG("  andi with no constant mask — unconstrained");
      return getOrCreateVar(v);
    }

    // x & (N-1) = x mod N when N is a power of 2 and x >= 0.
    int64_t N = mask + 1;
    if (mask <= 0 || !llvm::isPowerOf2_64(N)) {
      LDBG("  andi mask 0x" << llvm::Twine::utohexstr(mask)
                            << " is not (power-of-2 - 1) — unconstrained");
      return getOrCreateVar(v);
    }

    LDBG("  andi mask 0x" << llvm::Twine::utohexstr(mask) << " → mod " << N);
    return encodeRemainder(v, dynamicOperand, N);
  }

  unsigned encodeCast(Value v, Value source) {
    unsigned srcIdx = encodeIndex(source);
    valueToSetDimIdx[v] = srcIdx;
    LDBG("  var[" << srcIdx << "] ← cast pass-through");
    return srcIdx;
  }

  // ---------------------------------------------------------------------------
  // Debug dump
  // ---------------------------------------------------------------------------

  void dumpConstraintSystem() const {
    LLVM_DEBUG({
      DBGS() << "Constraint system: " << poly.getNumVarKind(VarKind::SetDim)
             << " dims, " << poly.getNumVarKind(VarKind::Local) << " locals, "
             << poly.getNumEqualities() << " equalities, "
             << poly.getNumInequalities() << " inequalities\n";

      auto printRow = [&](unsigned row, bool isEq) {
        DBGS() << (isEq ? "  EQ: " : "  GE: ");
        unsigned numCols = poly.getNumCols();
        unsigned numSetDims = poly.getNumVarKind(VarKind::SetDim);
        unsigned numLocals = poly.getNumVarKind(VarKind::Local);

        bool first = true;
        for (unsigned j = 0; j < numCols; ++j) {
          int64_t c =
              isEq ? poly.atEq64(row, j) : poly.atIneq64(row, j);
          if (c == 0)
            continue;
          if (!first && c > 0)
            llvm::dbgs() << " + ";
          else if (c < 0)
            llvm::dbgs() << " - ";

          int64_t absC = std::abs(c);
          if (j == numCols - 1) {
            llvm::dbgs() << absC;
          } else if (j < numSetDims) {
            if (absC != 1)
              llvm::dbgs() << absC << "*";
            llvm::dbgs() << "d" << j;
          } else if (j < numSetDims + numLocals) {
            if (absC != 1)
              llvm::dbgs() << absC << "*";
            llvm::dbgs() << "q" << (j - numSetDims);
          }
          first = false;
        }
        llvm::dbgs() << (isEq ? " = 0" : " >= 0") << "\n";
      };

      for (unsigned i = 0; i < poly.getNumEqualities(); ++i)
        printRow(i, /*isEq=*/true);
      for (unsigned i = 0; i < poly.getNumInequalities(); ++i)
        printRow(i, /*isEq=*/false);
    });
  }
};

} // anonymous namespace

namespace mlir::triton {

bool areIndicesProvablyDifferent(Value idx1, Value idx2) {
  if (idx1 == idx2)
    return false;

  LDBG("Checking disjointness of two index values");

#ifndef NDEBUG
  auto totalStart = std::chrono::high_resolution_clock::now();
#endif

  IndexConstraintBuilder builder;

  LDBG("Encoding index 1:");
  unsigned v1 = builder.encodeIndex(idx1);
  LDBG("Encoding index 2:");
  unsigned v2 = builder.encodeIndex(idx2);

  if (v1 == v2) {
    LDBG("Both indices resolved to the same variable — cannot be disjoint");
    return false;
  }

  // Add equality constraint: v1 = v2
  auto eq = builder.zeroRow();
  eq[builder.getSetDimCol(v1)] = 1;
  eq[builder.getSetDimCol(v2)] = -1;
  builder.poly.addEquality(eq);

  LDBG("Query: var[" << v1 << "] = var[" << v2 << "]");
  builder.dumpConstraintSystem();

#ifndef NDEBUG
  auto solveStart = std::chrono::high_resolution_clock::now();
#endif
  bool empty = builder.poly.isIntegerEmpty();
#ifndef NDEBUG
  auto solveEnd = std::chrono::high_resolution_clock::now();
#endif

  LLVM_DEBUG({
    auto encodeElapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        solveStart - totalStart);
    auto solveElapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        solveEnd - solveStart);
    auto totalElapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        solveEnd - totalStart);
    DBGS() << "isIntegerEmpty() = " << (empty ? "true" : "false")
           << "  encode=" << encodeElapsed.count()
           << "µs  solve=" << solveElapsed.count()
           << "µs  total=" << totalElapsed.count() << "µs ("
           << builder.poly.getNumVarKind(VarKind::SetDim) << " dims, "
           << builder.poly.getNumVarKind(VarKind::Local) << " locals, "
           << builder.poly.getNumEqualities() << " eqs, "
           << builder.poly.getNumInequalities() << " ineqs)\n";
    if (empty)
      DBGS() << "→ Indices are PROVABLY DISJOINT — no barrier needed\n";
    else
      DBGS() << "→ Indices MAY overlap — conservative barrier\n";
  });

  return empty;
}

} // namespace mlir::triton
