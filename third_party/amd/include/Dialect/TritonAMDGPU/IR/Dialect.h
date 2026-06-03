/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_IR_DIALECT_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Traits.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::amd {
struct L2Cache : public SideEffects::Resource::Base<L2Cache> {
  StringRef getName() const final { return "<AMDGPU::L2Cache>"; }
};
} // namespace mlir::triton::amd

namespace mlir::triton::amdgpu {
/// Returns the number of dwords for a TDM tensor descriptor based on rank.
/// 2D tensors: group0 (4) + group1 (8) = 12 dwords
/// 3D-5D tensors: group0 (4) + group1 (8) + group2 (4) + group3 (4) = 20 dwords
inline int getTensorDescNumDwords(triton::TensorDescType type) {
  auto shape = type.getShape();
  return (shape.size() > 2) ? (4 + 8 + 4 + 4) : (4 + 8);
}

/// Returns true iff `hint` is a legal `warp_used_hint`: an i32 bitmask whose
/// active warps form a regular axis-aligned bit pattern (an axis-aligned coset
/// of warp IDs).  Concretely: num_warps is a power of two below 32, the hint is
/// non-zero with no bits beyond num_warps, K = popcount(hint) is a power of
/// two, and the active warps span exactly log2(K) warpId bit positions.  See
/// `AsyncTDMCopyGlobalToLocalOp` in TritonAMDGPUOps.td and
/// triton-lang/triton#10056.  This is the op verifier's per-hint legality
/// check; the TDM merge lowering relies on the same coset structure when it
/// builds per-member warp predicates.
bool isAxisAlignedWarpHint(uint32_t hint, int64_t numWarps);
} // namespace mlir::triton::amdgpu

// clang-format off
#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h.inc"
#include "amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUEnums.h.inc"
// clang-format on

#define GET_ATTRDEF_CLASSES
#include "amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUAttrDefs.h.inc"

#include "amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOpInterfaces.h.inc"
#define GET_OP_CLASSES
#include "amd/include/Dialect/TritonAMDGPU/IR/Ops.h.inc"

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_IR_DIALECT_H_
