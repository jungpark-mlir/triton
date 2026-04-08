#ifndef TRITON_ANALYSIS_PRESBURGER_INDEX_ANALYSIS_H
#define TRITON_ANALYSIS_PRESBURGER_INDEX_ANALYSIS_H

#include "mlir/IR/Value.h"

namespace mlir::triton {

/// Check if two index values are provably different using Presburger
/// arithmetic. Walks the arith op chains from each value, builds an
/// IntegerPolyhedron constraint system, adds the equality constraint
/// idx1 = idx2, and checks emptiness via GCD test / Simplex.
///
/// Returns true if the constraint system is empty (no integer solution),
/// meaning the indices can never be equal.
///
/// Supported arith ops: ConstantIntOp, AddIOp, SubIOp, MulIOp (constant
/// factor), RemSIOp, RemUIOp (constant modulus), AndIOp (power-of-2 mask),
/// IndexCastOp, ExtSIOp, ExtUIOp, TruncIOp.
/// Unrecognized ops produce unconstrained variables (conservative).
///
/// Enable debug output with:
///   triton-opt --debug-only=presburger-index-analysis
///   TRITON_LLVM_DEBUG_ONLY=presburger-index-analysis
bool areIndicesProvablyDifferent(Value idx1, Value idx2);

} // namespace mlir::triton

#endif // TRITON_ANALYSIS_PRESBURGER_INDEX_ANALYSIS_H
