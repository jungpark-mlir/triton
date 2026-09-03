"""Canonical, performance-retained GFX1250 Gluon GEMM implementations.

This module owns four optimized GEMM families:

* ``bf16``: BF16 operands with FP32 accumulation and BF16 output.
* ``mxfp8``: E4M3 operands with block-32 E8M0 scales.
* ``fp8_mxfp4``: unscaled E4M3 A and block-scaled packed E2M1 B.
* ``mxfp4``: packed E2M1 operands with block-32 E8M0 scales.

Every default kernel uses eight warps and a 4x4 CTA cluster. A cluster computes
one 1024x1024 output tile, so each CTA owns a 256x256 partition. Operand tiles
and scales are distributed across the cluster through CGA-aware shared-memory
layouts; TDM moves global tiles to and from that distributed storage.

The device functions intentionally remain explicit and format-specific.
Statement order around ``warp_pipeline_stage``, cluster barriers, TDM issue,
and ``tdm.async_wait`` is part of the performance contract, not stylistic
boilerplate. Before changing those regions, compare correctness, executable
AMDGCN, register/scratch use, and locked benchmark medians.

The ``snapshot_*`` helper prefix is temporarily retained because this package
was promoted losslessly from the validated experiment. It identifies source
lineage only; this package is now the canonical maintained owner.

Historical target measurements at M=N=4096, K=65536 were approximately
3.2 PFLOPS for BF16, 8.3 PFLOPS for MXFP8, 9.9 PFLOPS for FP8 x MXFP4, and
15.6 PFLOPS for MXFP4. Performance depends on input data and GPU DVFS state;
see README.md for the reproducible benchmark protocol and current results.
"""

import argparse
import gc
from pathlib import Path
import sys
import time

import torch
import triton
from triton._C.libtriton.gluon_ir import make_cga_layout
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm
from triton.tools.mxfp import MXFP4Tensor


REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from third_party.amd.python.examples.gluon.mxfp_gemm_cdna5 import (  # noqa: E402
    MXScaleTensor,
    init_data,
    torch_gemm_mxfp,
)


BLOCK_M = 1024
BLOCK_N = 1024
# K depth differs by format because packed E2M1 stores two values per byte and
# scaled WMMA consumes a different logical K span than ordinary BF16 WMMA.
BF16_BLOCK_K = 128
MXFP8_BLOCK_K = 256
MXFP4_BLOCK_K = 512
# These are retained launch invariants for the peak 4x4-cluster paths.
NUM_WARPS = 8
NUM_BUFFERS = 2
CTA_M = 4
CTA_N = 4
GROUP_SIZE_M = 4
NUM_XCDS = 8
SCALE_BLOCK = 32
PRESHUFFLE_FACTOR = 128
AGPR_ATTRS = (("amdgpu-agpr-alloc", "0,0"),)
KERNEL_NAMES = ("bf16", "mxfp8", "fp8_mxfp4", "mxfp4")


# ---------------------------------------------------------------------------
# Shared grid mapping and output epilogue.
#
# Programs are first spread across eight XCDs, then grouped along one output
# dimension to improve operand reuse. The default mapping reuses B while the
# ``reuse_a`` form swaps grouping axes for the mixed FP8 x MXFP4 kernel.
#
# The output helper converts FP32 accumulators to the requested output type in
# distributed LDS, then performs one cluster-wide TDM store. Eight elements of
# row padding avoid the bank conflicts of an unpadded identity layout.

@gluon.jit
def snapshot_get_xcd_swizzled_pids(
        M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr,
        GROUP_SIZE_M: gl.constexpr):
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    if NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = (tall_xcds * pids_per_xcd
                   + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@gluon.jit
def snapshot_get_xcd_swizzled_pids_reuse_a(
        M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr,
        GROUP_SIZE_N: gl.constexpr):
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    if NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = (tall_xcds * pids_per_xcd
                   + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid)
    num_pid_in_group = GROUP_SIZE_N * num_pid_m
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
    pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size_n)
    pid_m = (pid % num_pid_in_group) // group_size_n
    return pid_m, pid_n


@gluon.jit
def snapshot_cluster_wait():
    gl.amd.gfx1250.cluster.arrive()
    gl.amd.gfx1250.cluster.wait()


@gluon.jit
def snapshot_tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        STORE_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, CTA_N: gl.constexpr):
    c_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N // CTA_N, 8]], [BLOCK_M, BLOCK_N], [1, 0],
        STORE_LAYOUT.cga_layout)
    c_shared = gl.allocate_shared_memory(
        c_ptr.type.element_ty, [BLOCK_M, BLOCK_N], c_shared_layout)
    c_shared.store(acc.to(c_ptr.type.element_ty))
    c_desc = tdm.make_tensor_descriptor(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        block_shape=(BLOCK_M, BLOCK_N), layout=c_shared_layout)
    tdm.async_store(
        c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], c_shared)
    tdm.async_wait(0)


# MXFP8 uses a half-width output stage. Two serial stores reduce accumulator
# partition width and let the output allocation reuse the dead operand arena.
@gluon.jit
def mxfp8_tdm_store_serial_n2(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc):
    cta_m: gl.constexpr = 256
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    cga_layout: gl.constexpr = (
        (1, 0, 0, 0), (2, 0, 0, 0),
        (0, 1, 0, 0), (0, 2, 0, 0))
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[half_n, 8]], [4, 4, cta_m, half_n],
        [3, 2, 1, 0], cga_layout)
    shared = gl.allocate_shared_memory(
        c_ptr.type.element_ty, [4, 4, cta_m, half_n], shared_layout)
    acc4 = acc.reshape((4, cta_m, 4, cta_n)).permute((0, 2, 1, 3))
    output0 = gl.amd.slice(
        acc4, [4, 4, cta_m, half_n], [0, 0, 0, 0])
    output1 = gl.amd.slice(
        acc4, [4, 4, cta_m, half_n], [0, 0, 0, half_n])
    desc = tdm.make_tensor_descriptor(
        base=c_ptr,
        shape=(M // cta_m, N // cta_n, cta_m, cta_n),
        strides=(cta_m * stride_cm, cta_n * stride_cn, stride_cm, stride_cn),
        block_shape=(4, 4, cta_m, half_n), layout=shared_layout)
    base_m = pid_m * 4
    base_n = pid_n * 4
    shared.store(output0.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, 0], shared)
    tdm.async_wait(0)
    shared.store(output1.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, half_n], shared)
    tdm.async_wait(0)


# ---------------------------------------------------------------------------
# BF16 x BF16.
#
# Aggregate tile: 1024x1024x128 across a 4x4 cluster.
# Pipeline: two BK128 LDS slots; each tile is consumed as two K64 halves.
# The refill-side K64 is further split into two K32 WMMAs so cluster wait and
# the next TDM refill sit between useful tensor operations. This precise
# arrive -> first K32 WMMA -> wait/refill -> second K32 WMMA cadence is retained.

@gluon.jit
def bf16_issue_loads(a_desc, b_desc, a_dst, b_dst, BLOCK_K: gl.constexpr):
    tdm.async_load(a_desc, [0, 0], a_dst, warp_used_hint=0b00001111)
    tdm.async_load(b_desc, [0, 0], b_dst, warp_used_hint=0b00001111)
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K])
    return a_desc, b_desc


@gluon.jit
def bf16_consume_slot(
        a_buf, b_buf, slot, acc, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        BLOCK_K: gl.constexpr):
    half_k: gl.constexpr = BLOCK_K // 2
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(0, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(
            0, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(
            half_k, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(
            half_k, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    return acc


@gluon.jit
def bf16_cluster_consume_and_refill(
        a_buf, b_buf, slot, a_desc, b_desc, acc,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        BLOCK_K: gl.constexpr):
    half_k: gl.constexpr = BLOCK_K // 2
    wmma_k: gl.constexpr = half_k // 2
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(0, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(
            0, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(
            half_k, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(
            half_k, half_k, 1).permute([1, 0]).load(layout=DOT_B)
        gl.amd.gfx1250.cluster.arrive()
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        # Retained refill-side split: the second K64 is two K32 WMMAs.
        a0 = gl.amd.slice(a, [BLOCK_M, wmma_k], [0, 0])
        b0 = gl.amd.slice(b, [wmma_k, BLOCK_N], [0, 0])
        a1 = gl.amd.slice(a, [BLOCK_M, wmma_k], [0, wmma_k])
        b1 = gl.amd.slice(b, [wmma_k, BLOCK_N], [wmma_k, 0])
        acc = gl.amd.gfx1250.wmma(a0, b0, acc)
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc = bf16_issue_loads(
            a_desc, b_desc, a_buf.index(slot), b_buf.index(slot), BLOCK_K)
        acc = gl.amd.gfx1250.wmma(a1, b1, acc)
    return a_desc, b_desc, acc


@gluon.jit
def bf16_kernelc_2pf_bk128_fused_cluster_4x4_gfx1250(
        a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk,
        stride_bn, stride_cm, stride_cn, GRID_MN: gl.constexpr,
        SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr,
        WMMA_LAYOUT: gl.constexpr):
    block_m: gl.constexpr = 1024
    block_n: gl.constexpr = 1024
    block_k: gl.constexpr = 128
    gl.static_assert(
        a_ptr.type.element_ty.is_bf16() and b_ptr.type.element_ty.is_bf16())
    gl.static_assert(gl.num_ctas() == 16)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
        M, N, block_m, block_n, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 8)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * block_m * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(block_m, block_k),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * block_n * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(block_n, block_k),
        layout=SHARED_LAYOUT_B)
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, block_m, block_k], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, block_n, block_k], SHARED_LAYOUT_B)
    a_desc, b_desc = bf16_issue_loads(
        a_desc, b_desc, a_buf.index(0), b_buf.index(0), block_k)
    a_desc, b_desc = bf16_issue_loads(
        a_desc, b_desc, a_buf.index(1), b_buf.index(1), block_k)
    tdm.async_wait(2)
    acc = gl.zeros((block_m, block_n), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, block_k)
    gl.assume(iter_max >= 2)
    for _ in range(0, (iter_max - 2) // 2):
        a_desc, b_desc, acc = bf16_cluster_consume_and_refill(
            a_buf, b_buf, 0, a_desc, b_desc, acc, dot_a, dot_b,
            block_m, block_n, block_k)
        tdm.async_wait(2)
        a_desc, b_desc, acc = bf16_cluster_consume_and_refill(
            a_buf, b_buf, 1, a_desc, b_desc, acc, dot_a, dot_b,
            block_m, block_n, block_k)
        tdm.async_wait(2)
    for tail_idx in range(iter_max - 2, iter_max - 1):
        slot = tail_idx % 2
        acc = bf16_consume_slot(
            a_buf, b_buf, slot, acc, dot_a, dot_b, block_k)
        tdm.async_wait(0)
        slot = (tail_idx + 1) % 2
        acc = bf16_consume_slot(
            a_buf, b_buf, slot, acc, dot_a, dot_b, block_k)
    snapshot_tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        WMMA_LAYOUT, block_m, block_n, 4)


def build_bf16_layouts():
    """Build matching distributed LDS and accumulator layouts for BF16.

    Layout construction is intentionally two-pass. First derive the local
    256x256-per-CTA WMMA ownership, then graft its operand CGA bases onto the
    padded shared layouts. This makes every CTA load the LDS partition consumed
    by its accumulator partition while retaining a cluster-wide 1024x1024 view.
    """
    cga_layout_c = make_cga_layout([4, 4], [4, 4], [0, 1])
    slice_m = BLOCK_M // 4
    slice_n = BLOCK_N // 4
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[BF16_BLOCK_K, 8]], [BLOCK_M, BF16_BLOCK_K], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[BF16_BLOCK_K, 8]], [BLOCK_N, BF16_BLOCK_K], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        BLOCK_M, BLOCK_N, local_a, local_b, NUM_WARPS, [16, 16, 32],
        a_transposed=False, b_transposed=True, slice_m=slice_m,
        slice_n=slice_n, transposed=True)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga_layout_c)
    dot_a = gl.DotOperandLayout(0, wmma, 8)
    dot_b = gl.DotOperandLayout(1, wmma, 8)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[BF16_BLOCK_K, 8]], [BLOCK_M, BF16_BLOCK_K], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[BF16_BLOCK_K, 8]], [BLOCK_N, BF16_BLOCK_K], [1, 0], cga_b)
    # One group per physical partition is the retained partitioned layout.
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        slice_m, slice_n, padded_a, padded_b, NUM_WARPS, [16, 16, 32],
        a_transposed=False, b_transposed=True, slice_m=slice_m,
        slice_n=slice_n, transposed=True)
    return shared_a, shared_b, wmma


# ---------------------------------------------------------------------------
# MXFP8 x MXFP8.
#
# Aggregate tile: 1024x1024x256 across a 4x4 cluster.
# Encoding: E4M3 values with one E8M0 scale per 32 logical K elements.
# Pipeline: two BK256 slots, each consumed as two K128 scaled WMMAs. A data uses
# producer warps 0-1; the four B layout pieces use warps 2-3 and 6-7. Scale
# transfers retain the two-warp A/B split across warps 0-3.
# The measured steady-state order is cluster wait -> data refill -> scale
# refill -> final K128 WMMA. Reordering this source changes performance.
# The final two BK256 slots retain explicit stage0/stage1 warp-pipeline
# annotations while the TDM ring drains.
# The epilogue serializes two 128-column stores. Besides halving the output
# staging tile, its 128-column accumulator partition removes the old spill.

@gluon.jit
def mxfp8_load_scale(
        scale_buffer, slot, start_k: gl.constexpr, LAYOUT: gl.constexpr,
        BLOCK_NONK: gl.constexpr, BK_SCALE: gl.constexpr,
        SUBTILE_SCALE_K: gl.constexpr):
    scale_slice = scale_buffer.index(slot).reshape(
        (BLOCK_NONK // 128, BK_SCALE // 4, 32, 4, 4)
    ).permute((0, 3, 2, 1, 4)).reshape((BLOCK_NONK, BK_SCALE))
    return scale_slice.slice(0, BLOCK_NONK, 0).slice(
        start_k, SUBTILE_SCALE_K, 1).load(layout=LAYOUT)


@gluon.jit
def mxfp8_issue_leading4_data(
        a_desc, b_desc, a_buf, b_buf, tile_idx, slot):
    tile_k = tile_idx * 256
    a_load = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, tile_k])
    b_load = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, tile_k])
    tdm.async_load_fused([
        (a_load, a_buf.index(slot), 0b00000011),
        # The 128-column accumulator partition creates four logical B pieces.
        (b_load, b_buf.index(slot), 0b11001100),
    ])


@gluon.jit
def mxfp8_issue_leading4_scale(
        as_desc, bs_desc, as_buf, bs_buf, tile_idx, slot):
    scale_k = tile_idx * (256 // 32) * 128
    as_load = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, scale_k])
    bs_load = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, scale_k])
    tdm.async_load_fused([
        (as_load, as_buf.index(slot), 0b00000011),
        (bs_load, bs_buf.index(slot), 0b00001100),
    ])


@gluon.jit
def mxfp8_consume_tile(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, DOT_A: gl.constexpr,
        DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr):
    a0 = a_buf.index(slot).slice(0, 1024, 0).slice(
        0, 128, 1).load(layout=DOT_A)
    as0 = mxfp8_load_scale(
        as_buf, slot, 0, SCALE_A_LAYOUT, 1024, 8, 4)
    b0 = b_buf.index(slot).slice(0, 1024, 0).slice(
        0, 128, 1).permute([1, 0]).load(layout=DOT_B)
    bs0 = mxfp8_load_scale(
        bs_buf, slot, 0, SCALE_B_LAYOUT, 1024, 8, 4)
    acc = gl.amd.gfx1250.wmma_scaled(
        a0, as0, "e4m3", b0, bs0, "e4m3", acc)
    a1 = a_buf.index(slot).slice(0, 1024, 0).slice(
        128, 128, 1).load(layout=DOT_A)
    as1 = mxfp8_load_scale(
        as_buf, slot, 4, SCALE_A_LAYOUT, 1024, 8, 4)
    b1 = b_buf.index(slot).slice(0, 1024, 0).slice(
        128, 128, 1).permute([1, 0]).load(layout=DOT_B)
    bs1 = mxfp8_load_scale(
        bs_buf, slot, 4, SCALE_B_LAYOUT, 1024, 8, 4)
    return gl.amd.gfx1250.wmma_scaled(
        a1, as1, "e4m3", b1, bs1, "e4m3", acc)


@gluon.jit
def mxfp8_consume_pipelined_tail(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, DOT_A: gl.constexpr,
        DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr):
    with gl.amd.warp_pipeline_stage("stage0_tail", priority=0):
        a0 = a_buf.index(slot).slice(0, 1024, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        as0 = mxfp8_load_scale(
            as_buf, slot, 0, SCALE_A_LAYOUT, 1024, 8, 4)
        b0 = b_buf.index(slot).slice(0, 1024, 0).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs0 = mxfp8_load_scale(
            bs_buf, slot, 0, SCALE_B_LAYOUT, 1024, 8, 4)
    with gl.amd.warp_pipeline_stage("stage1_tail", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0, bs0, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("stage0_tail", priority=0):
        a1 = a_buf.index(slot).slice(0, 1024, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        as1 = mxfp8_load_scale(
            as_buf, slot, 4, SCALE_A_LAYOUT, 1024, 8, 4)
        b1 = b_buf.index(slot).slice(0, 1024, 0).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs1 = mxfp8_load_scale(
            bs_buf, slot, 4, SCALE_B_LAYOUT, 1024, 8, 4)
    with gl.amd.warp_pipeline_stage("stage1_tail", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1, bs1, "e4m3", acc)
    return acc


@gluon.jit
def mxfp8_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, slot, a_desc, b_desc, as_desc,
        bs_desc, refill_idx, acc, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr):
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a0 = a_buf.index(slot).slice(0, 1024, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        as0 = mxfp8_load_scale(
            as_buf, slot, 0, SCALE_A_LAYOUT, 1024, 8, 4)
        b0 = b_buf.index(slot).slice(0, 1024, 0).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs0 = mxfp8_load_scale(
            bs_buf, slot, 0, SCALE_B_LAYOUT, 1024, 8, 4)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0, bs0, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a1 = a_buf.index(slot).slice(0, 1024, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        as1 = mxfp8_load_scale(
            as_buf, slot, 4, SCALE_A_LAYOUT, 1024, 8, 4)
        b1 = b_buf.index(slot).slice(0, 1024, 0).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs1 = mxfp8_load_scale(
            bs_buf, slot, 4, SCALE_B_LAYOUT, 1024, 8, 4)
        gl.amd.gfx1250.cluster.arrive()
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        # The measured schedule is fixed here: wait, data refill, scale refill,
        # then (and only then) consume the final staged K128.
        gl.amd.gfx1250.cluster.wait()
        mxfp8_issue_leading4_data(
            a_desc, b_desc, a_buf, b_buf, refill_idx, slot)
        mxfp8_issue_leading4_scale(
            as_desc, bs_desc, as_buf, bs_buf, refill_idx, slot)
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1, bs1, "e4m3", acc)
    return acc


@gluon.jit
def fp8_scaled_cluster_bf16_style_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr):
    gl.static_assert(gl.num_ctas() == 16)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
        M, N, 1024, 1024, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [1024, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [1024, 4])
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, 1024, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, 1024, 256], SHARED_LAYOUT_B)
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty, [2, 8, 1024], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty, [2, 8, 1024], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * 1024 * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(1024, 256),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * 1024 * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(1024, 256),
        layout=SHARED_LAYOUT_B)
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + (pid_m * 1024) // 128 * stride_scale,
        shape=(M // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(8, 1024), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * 1024) // 128 * stride_scale,
        shape=(N // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(8, 1024), layout=SHARED_SCALE_B)
    for prefetch_idx in gl.static_range(2):
        mxfp8_issue_leading4_data(
            a_desc, b_desc, a_buf, b_buf, prefetch_idx, prefetch_idx)
        mxfp8_issue_leading4_scale(
            as_desc, bs_desc, as_buf, bs_buf, prefetch_idx, prefetch_idx)
    tdm.async_wait(2)
    snapshot_cluster_wait()
    acc = gl.zeros((1024, 1024), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 256)
    gl.assume(iter_max >= 2)
    for tile_idx in range(0, iter_max - 2):
        slot = tile_idx % 2
        acc = mxfp8_consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, slot, a_desc, b_desc, as_desc,
            bs_desc, tile_idx + 2, acc, dot_a, dot_b, scale_a_layout,
            scale_b_layout)
        tdm.async_wait(2)
    penultimate_slot = (iter_max - 2) % 2
    acc = mxfp8_consume_pipelined_tail(
        a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout)
    tdm.async_wait(0)
    snapshot_cluster_wait()
    last_slot = (iter_max - 1) % 2
    acc = mxfp8_consume_pipelined_tail(
        a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout)
    # The serial output stage aliases cluster-distributed operand LDS. All CTAs
    # must finish the final operand reads before any CTA repurposes that arena.
    snapshot_cluster_wait()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    mxfp8_tdm_store_serial_n2(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc)


def build_mxfp8_layouts():
    """Build the 4x4-cluster data, scale, and WMMA layouts for MXFP8.

    Data operands use the K128 scaled-WMMA instruction shape. Scale layouts
    reuse the corresponding A/B CGA bases, so data and its block-32 scales are
    owned by the same CTA partition. The 128-column N slice matches the serial
    epilogue and reduces accumulator register pressure enough to avoid spills.
    """
    cga_c = make_cga_layout([4, 4], [4, 4], [0, 1])
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        1024, 1024, local_a, local_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=128)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga_c)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        256, 256, padded_a, padded_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=128)
    shared_as = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [8, 1024], [1, 0], cga_a)
    shared_bs = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [8, 1024], [1, 0], cga_b)
    return shared_a, shared_b, shared_as, shared_bs, wmma


# ---------------------------------------------------------------------------
# FP8 x MXFP4.
#
# A is ordinary E4M3, so its scale operand is ``None`` (mathematically one).
# B is packed E2M1 with block-32 E8M0 scales. One BK256 logical tile therefore
# occupies 256 A columns but only 128 packed B bytes.
#
# The default launch uses a three-slot TDM ring, a single-slot prologue, a
# warp-pipelined drain, and A-reusing output-tile order. Unlike the other peak
# paths, cluster width, ring depth, and reuse direction remain CLI-selectable.

@gluon.jit
def fp8_mxfp4_load_scale(
        scale_buffer, slot, start_k: gl.constexpr, LAYOUT: gl.constexpr,
        BLOCK_N: gl.constexpr):
    scale_slice = scale_buffer.index(slot).reshape(
        (BLOCK_N // 128, 2, 32, 4, 4)
    ).permute((0, 3, 2, 1, 4)).reshape((BLOCK_N, 8))
    return scale_slice.slice(0, BLOCK_N, 0).slice(
        start_k, 4, 1).load(layout=LAYOUT)


@gluon.jit
def fp8_mxfp4_issue_leading4_data(
        a_desc, b_desc, a_buf, b_buf, tile_idx, slot,
        B_WARP_HINT: gl.constexpr):
    a_load = tdm.update_tensor_descriptor(
        a_desc, add_offsets=[0, tile_idx * 256])
    b_load = tdm.update_tensor_descriptor(
        b_desc, add_offsets=[0, tile_idx * 128])
    tdm.async_load_fused([
        (a_load, a_buf.index(slot), 0b00000011),
        (b_load, b_buf.index(slot), B_WARP_HINT),
    ])


@gluon.jit
def fp8_mxfp4_issue_leading4_scale(
        bs_desc, bs_buf, tile_idx, slot):
    scale_k = tile_idx * (256 // 32) * 128
    bs_load = tdm.update_tensor_descriptor(
        bs_desc, add_offsets=[0, scale_k])
    tdm.async_load(
        bs_load, [0, 0], bs_buf.index(slot), warp_used_hint=0b00001100)


@gluon.jit
def fp8_mxfp4_consume_tile(
        a_buf, b_buf, bs_buf, slot, acc, DOT_A: gl.constexpr,
        DOT_B: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
        0, 128, 1).load(layout=DOT_A)
    b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
        0, 64, 1).permute([1, 0]).load(layout=DOT_B)
    bs0 = fp8_mxfp4_load_scale(
        bs_buf, slot, 0, SCALE_B_LAYOUT, BLOCK_N)
    acc = gl.amd.gfx1250.wmma_scaled(
        a0, None, "e4m3", b0, bs0, "e2m1", acc)
    a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
        128, 128, 1).load(layout=DOT_A)
    b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
        64, 64, 1).permute([1, 0]).load(layout=DOT_B)
    bs1 = fp8_mxfp4_load_scale(
        bs_buf, slot, 4, SCALE_B_LAYOUT, BLOCK_N)
    return gl.amd.gfx1250.wmma_scaled(
        a1, None, "e4m3", b1, bs1, "e2m1", acc)


@gluon.jit
def fp8_mxfp4_consume_pipelined_tile(
        a_buf, b_buf, bs_buf, slot, acc, DOT_A: gl.constexpr,
        DOT_B: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    with gl.amd.warp_pipeline_stage("stage0_tail", priority=0):
        a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            0, 64, 1).permute([1, 0]).load(layout=DOT_B)
        bs0 = fp8_mxfp4_load_scale(
            bs_buf, slot, 0, SCALE_B_LAYOUT, BLOCK_N)
    with gl.amd.warp_pipeline_stage("stage1_tail", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, None, "e4m3", b0, bs0, "e2m1", acc)
    with gl.amd.warp_pipeline_stage("stage0_tail", priority=0):
        a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            64, 64, 1).permute([1, 0]).load(layout=DOT_B)
        bs1 = fp8_mxfp4_load_scale(
            bs_buf, slot, 4, SCALE_B_LAYOUT, BLOCK_N)
    with gl.amd.warp_pipeline_stage("stage1_tail", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, None, "e4m3", b1, bs1, "e2m1", acc)
    return acc


@gluon.jit
def fp8_mxfp4_consume_and_refill(
        a_buf, b_buf, bs_buf, slot, a_desc, b_desc, bs_desc, refill_idx,
        acc, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr, B_WARP_HINT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        CLUSTERED: gl.constexpr):
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            0, 64, 1).permute([1, 0]).load(layout=DOT_B)
        bs0 = fp8_mxfp4_load_scale(
            bs_buf, slot, 0, SCALE_B_LAYOUT, BLOCK_N)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, None, "e4m3", b0, bs0, "e2m1", acc)
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            64, 64, 1).permute([1, 0]).load(layout=DOT_B)
        bs1 = fp8_mxfp4_load_scale(
            bs_buf, slot, 4, SCALE_B_LAYOUT, BLOCK_N)
        if CLUSTERED:
            gl.amd.gfx1250.cluster.arrive()
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        if CLUSTERED:
            gl.amd.gfx1250.cluster.wait()
        fp8_mxfp4_issue_leading4_data(
            a_desc, b_desc, a_buf, b_buf, refill_idx, slot, B_WARP_HINT)
        fp8_mxfp4_issue_leading4_scale(
            bs_desc, bs_buf, refill_idx, slot)
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, None, "e4m3", b1, bs1, "e2m1", acc)
    return acc


@gluon.jit
def fp8_mxfp4_cluster_gemm_gfx1250(
        a_ptr, b_ptr, c_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr,
        SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
        PIPELINE_TAIL: gl.constexpr, SINGLE_SLOT_PROLOGUE: gl.constexpr,
        NUM_BUFFERS: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, CTA_M: gl.constexpr, CTA_N: gl.constexpr,
        REUSE_A: gl.constexpr, GROUP_SIZE: gl.constexpr):
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    gl.static_assert(BLOCK_M // CTA_M == 256)
    gl.static_assert(BLOCK_N // CTA_N == 256)
    clustered: gl.constexpr = CTA_M * CTA_N > 1
    if REUSE_A:
        pid_m, pid_n = snapshot_get_xcd_swizzled_pids_reuse_a(
            M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, GROUP_SIZE)
    else:
        pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
            M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, GROUP_SIZE)
    wmma_packed: gl.constexpr = gl.amd.AMDWMMALayout(
        3, WMMA_LAYOUT.transposed, WMMA_LAYOUT.warp_bases,
        WMMA_LAYOUT.reg_bases, [16, 16, 64], WMMA_LAYOUT.cga_layout)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma_packed, 16)
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [BLOCK_N, 4])

    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [NUM_BUFFERS, BLOCK_M, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [NUM_BUFFERS, BLOCK_N, 128], SHARED_LAYOUT_B)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_N // 128, 1024], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, 256),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K // 2),
        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, 128),
        layout=SHARED_LAYOUT_B)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * BLOCK_N) // 128 * stride_scale,
        shape=(N // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(BLOCK_N // 128, 1024), layout=SHARED_SCALE_B)
    b_warp_hint: gl.constexpr = 0b00001100

    if SINGLE_SLOT_PROLOGUE:
        fp8_mxfp4_issue_leading4_data(
            a_desc, b_desc, a_buf, b_buf, 0, 0, b_warp_hint)
        fp8_mxfp4_issue_leading4_scale(
            bs_desc, bs_buf, 0, 0)
        tdm.async_wait(0)
        if clustered:
            snapshot_cluster_wait()
        for prefetch_idx in gl.static_range(1, NUM_BUFFERS):
            fp8_mxfp4_issue_leading4_data(
                a_desc, b_desc, a_buf, b_buf, prefetch_idx, prefetch_idx,
                b_warp_hint)
            fp8_mxfp4_issue_leading4_scale(
                bs_desc, bs_buf, prefetch_idx, prefetch_idx)
    else:
        for prefetch_idx in gl.static_range(NUM_BUFFERS):
            fp8_mxfp4_issue_leading4_data(
                a_desc, b_desc, a_buf, b_buf, prefetch_idx, prefetch_idx,
                b_warp_hint)
            fp8_mxfp4_issue_leading4_scale(
                bs_desc, bs_buf, prefetch_idx, prefetch_idx)
        tdm.async_wait(2 * (NUM_BUFFERS - 1))
        if clustered:
            snapshot_cluster_wait()

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 256)
    gl.assume(iter_max >= NUM_BUFFERS)
    for tile_idx in range(0, iter_max - NUM_BUFFERS):
        slot = tile_idx % NUM_BUFFERS
        acc = fp8_mxfp4_consume_and_refill(
            a_buf, b_buf, bs_buf, slot, a_desc, b_desc, bs_desc,
            tile_idx + NUM_BUFFERS, acc, dot_a, dot_b, scale_b_layout,
            b_warp_hint, BLOCK_M, BLOCK_N, clustered)
        tdm.async_wait(2 * (NUM_BUFFERS - 1))

    for drain_idx in gl.static_range(NUM_BUFFERS):
        slot = (iter_max - NUM_BUFFERS + drain_idx) % NUM_BUFFERS
        if PIPELINE_TAIL:
            acc = fp8_mxfp4_consume_pipelined_tile(
                a_buf, b_buf, bs_buf, slot, acc, dot_a, dot_b,
                scale_b_layout, BLOCK_M, BLOCK_N)
        else:
            acc = fp8_mxfp4_consume_tile(
                a_buf, b_buf, bs_buf, slot, acc, dot_a, dot_b,
                scale_b_layout, BLOCK_M, BLOCK_N)
        if drain_idx < NUM_BUFFERS - 1:
            tdm.async_wait(2 * (NUM_BUFFERS - 2 - drain_idx))
            if clustered:
                snapshot_cluster_wait()
    if NUM_BUFFERS == 3 and clustered:
        snapshot_cluster_wait()
    a_buf._keep_alive()
    b_buf._keep_alive()
    bs_buf._keep_alive()
    snapshot_tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        WMMA_LAYOUT, BLOCK_M, BLOCK_N, CTA_N)


def build_fp8_mxfp4_layouts(cluster_width=4):
    """Build mixed-format layouts for a square CTA cluster.

    The E4M3 A tile has 256 stored K elements while packed E2M1 B has 128
    stored bytes for the same logical K extent. ``wmma_packed`` changes only
    B's instruction K shape; output ownership remains compatible with A.
    """
    block_m = 256 * cluster_width
    block_n = 256 * cluster_width
    cga = make_cga_layout(
        [cluster_width, cluster_width], [cluster_width, cluster_width], [0, 1])
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_m, 256], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[128, 16]], [block_n, 128], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, block_n, local_a, local_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=256)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga)
    wmma_packed = gl.amd.AMDWMMALayout(
        3, wmma.transposed, wmma.warp_bases, wmma.reg_bases,
        [16, 16, 64], cga)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma_packed, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_m, 256], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[128, 16]], [block_n, 128], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        256, 256, padded_a, padded_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=256)
    shared_bs = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_n // 128, 1024], [1, 0], cga_b)
    return shared_a, shared_b, shared_bs, wmma


# ---------------------------------------------------------------------------
# MXFP4 x MXFP4.
#
# Aggregate tile: 1024x1024x512 logical values across a 4x4 cluster. Packed A/B
# storage reduces that K extent to 256 bytes. Each slot is consumed as two K256
# groups, and each group contains two K128 scaled WMMAs.
#
# Only the second K256 group overlaps cluster arrive/wait with its two WMMAs.
# Keeping the original grouping and barrier placement avoids scheduler bubbles.

@gluon.jit
def mxfp4_load_scale(
        scale_buffer, slot, start_k: gl.constexpr, LAYOUT: gl.constexpr):
    scale_slice = scale_buffer.index(slot).reshape(
        (8, 4, 32, 4, 4)).permute((0, 3, 2, 1, 4)).reshape((1024, 16))
    return scale_slice.slice(0, 1024, 0).slice(
        start_k, 4, 1).load(layout=LAYOUT)


@gluon.jit
def mxfp4_issue_refill(
        a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
        tile_idx, slot):
    packed_k = tile_idx * 256
    a_load = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, packed_k])
    b_load = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, packed_k])
    tdm.async_load_fused([
        (a_load, a_buf.index(slot), 0b00000011),
        (b_load, b_buf.index(slot), 0b00001100),
    ])
    scale_k = tile_idx * 2048
    as_load = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, scale_k])
    bs_load = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, scale_k])
    tdm.async_load_fused([
        (as_load, as_buf.index(slot), 0b00000011),
        (bs_load, bs_buf.index(slot), 0b00001100),
    ])


@gluon.jit
def mxfp4_consume_k256(
        a_buf, b_buf, as_buf, bs_buf, slot, acc,
        START_K_PACKED: gl.constexpr, START_SCALE_K: gl.constexpr,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        OVERLAP_CLUSTER: gl.constexpr):
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a0 = a_buf.index(slot).slice(0, 1024, 0).slice(
            START_K_PACKED, 64, 1).load(layout=DOT_A)
        as0 = mxfp4_load_scale(
            as_buf, slot, START_SCALE_K, SCALE_A_LAYOUT)
        b0 = b_buf.index(slot).slice(0, 1024, 0).slice(
            START_K_PACKED, 64, 1).permute([1, 0]).load(layout=DOT_B)
        bs0 = mxfp4_load_scale(
            bs_buf, slot, START_SCALE_K, SCALE_B_LAYOUT)
        a1 = a_buf.index(slot).slice(0, 1024, 0).slice(
            START_K_PACKED + 64, 64, 1).load(layout=DOT_A)
        as1 = mxfp4_load_scale(
            as_buf, slot, START_SCALE_K + 4, SCALE_A_LAYOUT)
        b1 = b_buf.index(slot).slice(0, 1024, 0).slice(
            START_K_PACKED + 64, 64, 1).permute([1, 0]).load(layout=DOT_B)
        bs1 = mxfp4_load_scale(
            bs_buf, slot, START_SCALE_K + 4, SCALE_B_LAYOUT)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e2m1", b0, bs0, "e2m1", acc)
        if OVERLAP_CLUSTER:
            gl.amd.gfx1250.cluster.arrive()
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e2m1", b1, bs1, "e2m1", acc)
        if OVERLAP_CLUSTER:
            gl.amd.gfx1250.cluster.wait()
    return acc


@gluon.jit
def mxfp4_consume_tile(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, DOT_A: gl.constexpr,
        DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr, OVERLAP_CLUSTER: gl.constexpr):
    # The retained path deliberately keeps the original K256 grouping/order.
    acc = mxfp4_consume_k256(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, 0, 0, DOT_A, DOT_B,
        SCALE_A_LAYOUT, SCALE_B_LAYOUT, False)
    return mxfp4_consume_k256(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, 128, 8, DOT_A, DOT_B,
        SCALE_A_LAYOUT, SCALE_B_LAYOUT, OVERLAP_CLUSTER)


@gluon.jit
def mxfp4_bk512_warp_pipeline_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr):
    gl.static_assert(gl.num_ctas() == 16)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
        M, N, 1024, 1024, GRID_MN, 8, 4)
    wmma_packed: gl.constexpr = gl.amd.AMDWMMALayout(
        3, WMMA_LAYOUT.transposed, WMMA_LAYOUT.warp_bases,
        WMMA_LAYOUT.reg_bases, [32, 16, 64], WMMA_LAYOUT.cga_layout)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma_packed, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma_packed, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [1024, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [1024, 4])
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, 1024, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, 1024, 256], SHARED_LAYOUT_B)
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty, [2, 8, 2048], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty, [2, 8, 2048], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * 1024 * stride_am, shape=(M, K // 2),
        strides=(stride_am, stride_ak), block_shape=(1024, 256),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * 1024 * stride_bn, shape=(N, K // 2),
        strides=(stride_bn, stride_bk), block_shape=(1024, 256),
        layout=SHARED_LAYOUT_B)
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + (pid_m * 1024) // 128 * stride_scale,
        shape=(M // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(8, 2048), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * 1024) // 128 * stride_scale,
        shape=(N // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(8, 2048), layout=SHARED_SCALE_B)
    for prefetch_idx in gl.static_range(2):
        mxfp4_issue_refill(
            a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
            prefetch_idx, prefetch_idx)
    tdm.async_wait(2)
    snapshot_cluster_wait()
    acc = gl.zeros((1024, 1024), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 512)
    gl.assume(iter_max >= 2)
    for tile_idx in range(0, iter_max - 2):
        slot = tile_idx % 2
        acc = mxfp4_consume_tile(
            a_buf, b_buf, as_buf, bs_buf, slot, acc, dot_a, dot_b,
            scale_a_layout, scale_b_layout, True)
        with gl.amd.warp_pipeline_stage("tdm", priority=1):
            mxfp4_issue_refill(
                a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                bs_buf, tile_idx + 2, slot)
        tdm.async_wait(2)
    penultimate_slot = (iter_max - 2) % 2
    acc = mxfp4_consume_tile(
        a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout, False)
    tdm.async_wait(0)
    snapshot_cluster_wait()
    last_slot = (iter_max - 1) % 2
    acc = mxfp4_consume_tile(
        a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout, False)
    snapshot_tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        WMMA_LAYOUT, 1024, 1024, 4)


def build_mxfp4_layouts():
    """Build packed-data and scale layouts for the BK512 MXFP4 kernel.

    Both operands use packed K64 fragments and block-32 scales. Their CGA bases
    are mirrored because B is loaded in [N,K] order and transposed for WMMA.
    """
    cga = make_cga_layout([4, 4], [4, 4], [0, 1])
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        1024, 1024, local_a, local_b, 8, [32, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=256)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        256, 256, padded_a, padded_b, 8, [32, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=256)
    shared_as = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [8, 2048], [1, 0], cga_a)
    shared_bs = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [8, 2048], [1, 0], cga_b)
    return shared_a, shared_b, shared_as, shared_bs, wmma


# ---------------------------------------------------------------------------
# Host-side input, launch, correctness, and benchmark glue.
#
# Storage conventions:
# * B is passed to kernels as an [N, K] contiguous transpose.
# * MXFP4 operands are packed along K before transfer to the GPU.
# * E8M0 scales are preshuffled in groups of 128 non-K values so LDS fragments
#   match ``get_wmma_scale_layout`` without a runtime transpose.
# * Every retained kernel accumulates in FP32 and writes BF16 by default.
#
# Benchmarking performs an event-timed probe, captures repeated launches in a
# CUDA/HIP graph, and reports the average latency over all graph replays.
# ``gpu-lock`` only serializes access; it does not pin clocks or disable DVFS.

def make_trig_tensor(shape, dtype, cosine):
    """Create a CPU tensor filled with sin(i) or cos(i) in flat-index order."""
    result = torch.empty(shape, dtype=dtype)
    flat = result.view(-1)
    chunk = 8 * 1024 * 1024
    for begin in range(0, flat.numel(), chunk):
        end = min(begin + chunk, flat.numel())
        angle = torch.arange(begin, end, dtype=torch.float64)
        values = torch.cos(angle) if cosine else torch.sin(angle)
        flat[begin:end].copy_(values.float())
    return result


def pack_scale(scale, scale_kwidth):
    """Preshuffle [non-K, K-block] E8M0 scales for scaled-WMMA LDS loads.

    The factor-128 permutation groups the four scale lanes consumed by each
    WMMA scale fragment. The returned 2D tensor is the global-memory format
    expected by the scale TDM descriptors in this module.
    """
    non_k, k_scale = scale.shape
    scale = scale.view(
        non_k // 128, 4, 32, k_scale // scale_kwidth, scale_kwidth)
    scale = scale.permute(0, 3, 2, 1, 4).contiguous()
    return scale.view(non_k // 128, k_scale * 128)


def make_bf16_case(args):
    """Build BF16 launch/check closures while keeping their GPU tensors alive."""
    if args.M % 1024 or args.N % 1024 or args.K % 128:
        raise ValueError("BF16 requires M/N divisible by 1024 and K by 128")
    if args.K // 128 < 2 or (args.K // 128) % 2:
        raise ValueError("BF16 requires an even number of at least two BK128 tiles")
    mode = args.input_mode or "trig"
    if mode == "random":
        torch.manual_seed(args.seed)
        a = torch.randn((args.M, args.K), dtype=torch.bfloat16)
        b = torch.randn((args.K, args.N), dtype=torch.bfloat16).T.contiguous()
        a = a.cuda()
        b = b.cuda()
    else:
        device = torch.device("cuda")
        a = torch.empty(
            (args.M, args.K), dtype=torch.bfloat16, device=device)
        b = torch.empty(
            (args.N, args.K), dtype=torch.bfloat16, device=device)
        chunk = 8 * 1024 * 1024
        for output, cosine in ((a, False), (b, True)):
            flat = output.view(-1)
            for begin in range(0, flat.numel(), chunk):
                end = min(begin + chunk, flat.numel())
                angle = torch.arange(
                    begin, end, dtype=torch.float64, device=device)
                value = torch.cos(angle) if cosine else torch.sin(angle)
                flat[begin:end].copy_(value.float())
    output_dtype = torch.bfloat16 if args.bf16_output else torch.float32
    c = torch.zeros((args.M, args.N), dtype=output_dtype, device="cuda")
    layouts = build_bf16_layouts()
    grid = (triton.cdiv(args.M, 1024) * triton.cdiv(args.N, 1024), 1)

    def launch():
        shared_a, shared_b, wmma = layouts
        return bf16_kernelc_2pf_bk128_fused_cluster_4x4_gfx1250[grid](
            a, b, c, args.M, args.N, args.K, a.stride(0), a.stride(1),
            b.stride(1), b.stride(0), c.stride(0), c.stride(1),
            GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, WMMA_LAYOUT=wmma,
            num_warps=8, waves_per_eu=2, num_ctas=16)

    def check():
        c.zero_()
        launch()
        torch.cuda.synchronize()
        reference = (a.cpu().float() @ b.cpu().T.float()).to(output_dtype)
        rtol = 1e-2 if output_dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(c.cpu(), reference, rtol=rtol, atol=1e-2)
        print("result verified", flush=True)

    return launch, check


def make_mxfp8_case(args):
    """Build the fixed E4M3/E8M0 block-32 4x4-cluster benchmark case."""
    if args.M % 1024 or args.N % 1024 or args.K % 256:
        raise ValueError("MXFP8 requires M/N divisible by 1024 and K by 256")
    if args.K // 256 < 2:
        raise ValueError("MXFP8 requires at least two BK256 tiles")
    if args.M % 128 or args.N % 128 or (args.K // 32) % 4:
        raise ValueError("MXFP8 dimensions are incompatible with scale packing")
    torch.manual_seed(args.seed)
    mode = args.input_mode or "trig"
    if mode == "trig":
        a = make_trig_tensor(
            (args.M, args.K), torch.float8_e4m3fn, False)
        b = make_trig_tensor(
            (args.K, args.N), torch.float8_e4m3fn, True)
    else:
        a = init_data("float8_e4m3", args.M, args.K)
        b = init_data("float8_e4m3", args.K, args.N)
    scale_k = args.K // 32
    a_scale_obj = MXScaleTensor(
        size=(args.M, scale_k)).random(low=1.0, high=32.0)
    b_scale_obj = MXScaleTensor(
        size=(args.N, scale_k)).random(low=1.0, high=32.0)
    c_ref = None
    if args.check:
        c_ref = torch_gemm_mxfp(
            a, b, a_scale_obj, b_scale_obj, 32,
            args.M, args.N, args.K).to(torch.bfloat16)
    a_scale = pack_scale(a_scale_obj.data, 4)
    b_scale = pack_scale(b_scale_obj.data, 4)
    a_d = a.contiguous().cuda()
    b_d = b.T.contiguous().cuda()
    as_d = a_scale.cuda()
    bs_d = b_scale.cuda()
    c_d = torch.empty(
        (args.M, args.N), dtype=torch.bfloat16, device="cuda")
    layouts = build_mxfp8_layouts()
    grid = (triton.cdiv(args.M, 1024) * triton.cdiv(args.N, 1024), 1)

    def launch():
        shared_a, shared_b, shared_as, shared_bs, wmma = layouts
        return fp8_scaled_cluster_bf16_style_kernel_gfx1250[grid](
            a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
            a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1), bs_d.stride(0),
            GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
            SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
            num_warps=8, waves_per_eu=2, num_ctas=16,
            llvm_fn_attrs=AGPR_ATTRS)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            c_d.cpu(), c_ref, rtol=1e-2, atol=5e-1)
        print("result verified", flush=True)

    return launch, check


def make_fp8_mxfp4_case(args):
    """Build the configurable E4M3 x packed-E2M1 benchmark case.

    Only B is block-scaled. The scaled-WMMA A-scale operand is ``None``, which
    represents a unit scale for the ordinary FP8 A matrix.
    """
    cluster_width = args.fp8_mxfp4_cluster_width
    block_m = 256 * cluster_width
    block_n = 256 * cluster_width
    if args.M % block_m or args.N % block_n or args.K % 256:
        raise ValueError(
            f"FP8xMXFP4 requires M/N divisible by {block_m} and K by 256")
    if args.K // 256 < args.fp8_mxfp4_num_buffers:
        raise ValueError(
            "FP8xMXFP4 requires at least one BK256 tile per buffer")
    if args.M % 128 or args.N % 128 or (args.K // 32) % 4:
        raise ValueError(
            "FP8xMXFP4 dimensions are incompatible with scale packing")
    torch.manual_seed(args.seed)
    mode = args.input_mode or "random"
    if mode == "trig":
        a = make_trig_tensor(
            (args.M, args.K), torch.float8_e4m3fn, False)
        b = MXFP4Tensor(size=(args.K, args.N))
        encoded = torch.empty(args.K * args.N, dtype=torch.uint8)
        chunk = 8 * 1024 * 1024
        for begin in range(0, encoded.numel(), chunk):
            end = min(begin + chunk, encoded.numel())
            angle = torch.arange(begin, end, dtype=torch.float64)
            encoded[begin:end].copy_(
                MXFP4Tensor(data=torch.cos(angle).float()).data)
        b.data = encoded.view(args.K, args.N)
    else:
        a = init_data("float8_e4m3", args.M, args.K)
        b = init_data("float4", args.K, args.N)

    scale_k = args.K // 32
    b_scale_obj = MXScaleTensor(
        size=(args.N, scale_k)).random(low=1.0, high=32.0)
    c_ref = None
    if args.check:
        a_scale_obj = MXScaleTensor(data=torch.ones(
            (args.M, scale_k), dtype=torch.float32))
        c_ref = torch_gemm_mxfp(
            a, b, a_scale_obj, b_scale_obj, 32,
            args.M, args.N, args.K).to(torch.bfloat16)

    a_d = a.contiguous().cuda()
    b_d = b.to_packed_tensor(dim=0).data.T.contiguous().cuda()
    bs_d = pack_scale(b_scale_obj.data, 4).cuda()
    c_d = torch.empty(
        (args.M, args.N), dtype=torch.bfloat16, device="cuda")
    layouts = build_fp8_mxfp4_layouts(cluster_width)
    grid = (
        triton.cdiv(args.M, block_m) * triton.cdiv(args.N, block_n), 1)
    group_size = args.fp8_mxfp4_group_size
    if group_size is None:
        group_size = 8 if args.K <= 8192 else 4

    def launch():
        shared_a, shared_b, shared_bs, wmma = layouts
        return fp8_mxfp4_cluster_gemm_gfx1250[grid](
            a_d, b_d, c_d, bs_d, args.M, args.N, args.K,
            a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1), bs_d.stride(0),
            GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, SHARED_SCALE_B=shared_bs,
            WMMA_LAYOUT=wmma,
            PIPELINE_TAIL=True, SINGLE_SLOT_PROLOGUE=True,
            NUM_BUFFERS=args.fp8_mxfp4_num_buffers,
            BLOCK_M=block_m, BLOCK_N=block_n,
            CTA_M=cluster_width, CTA_N=cluster_width,
            REUSE_A=args.fp8_mxfp4_reuse_order == "a",
            GROUP_SIZE=group_size,
            num_warps=8, waves_per_eu=2,
            num_ctas=cluster_width * cluster_width,
            llvm_fn_attrs=AGPR_ATTRS)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            c_d.cpu(), c_ref, rtol=1e-2, atol=5e-1)
        print("result verified", flush=True)

    return launch, check


def make_mxfp4_case(args):
    """Build the fixed packed-E2M1/E8M0 block-32 4x4-cluster case."""
    if args.M % 1024 or args.N % 1024 or args.K % 512:
        raise ValueError("MXFP4 requires M/N divisible by 1024 and K by 512")
    if args.K // 512 < 2:
        raise ValueError("MXFP4 requires at least two BK512 tiles")
    if args.M % 128 or args.N % 128 or (args.K // 32) % 4:
        raise ValueError("MXFP4 dimensions are incompatible with scale packing")
    torch.manual_seed(args.seed)
    mode = args.input_mode or "random"
    if mode == "trig":
        def make_mxfp4_trig(shape, cosine):
            result = MXFP4Tensor(size=shape)
            encoded = torch.empty(shape[0] * shape[1], dtype=torch.uint8)
            chunk = 8 * 1024 * 1024
            for begin in range(0, encoded.numel(), chunk):
                end = min(begin + chunk, encoded.numel())
                angle = torch.arange(begin, end, dtype=torch.float64)
                values = torch.cos(angle) if cosine else torch.sin(angle)
                encoded[begin:end].copy_(
                    MXFP4Tensor(data=values.float()).data)
            result.data = encoded.view(shape)
            return result

        a = make_mxfp4_trig((args.M, args.K), False)
        b = make_mxfp4_trig((args.K, args.N), True)
    else:
        a = init_data("float4", args.M, args.K)
        b = init_data("float4", args.K, args.N)
    scale_k = args.K // 32
    a_scale_obj = MXScaleTensor(
        size=(args.M, scale_k)).random(low=1.0, high=32.0)
    b_scale_obj = MXScaleTensor(
        size=(args.N, scale_k)).random(low=1.0, high=32.0)
    c_ref = None
    if args.check:
        c_ref = torch_gemm_mxfp(
            a, b, a_scale_obj, b_scale_obj, 32,
            args.M, args.N, args.K).to(torch.bfloat16)
    as_d = pack_scale(a_scale_obj.data, 4).cuda()
    bs_d = pack_scale(b_scale_obj.data, 4).cuda()
    a_d = a.to_packed_tensor(dim=1).data.contiguous().cuda()
    b_d = b.to_packed_tensor(dim=0).data.T.contiguous().cuda()
    c_d = torch.empty(
        (args.M, args.N), dtype=torch.bfloat16, device="cuda")
    layouts = build_mxfp4_layouts()
    grid = (triton.cdiv(args.M, 1024) * triton.cdiv(args.N, 1024), 1)

    def launch():
        shared_a, shared_b, shared_as, shared_bs, wmma = layouts
        return mxfp4_bk512_warp_pipeline_gfx1250[grid](
            a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
            a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1), bs_d.stride(0),
            GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
            SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
            num_warps=8, waves_per_eu=2, num_ctas=16,
            llvm_fn_attrs=AGPR_ATTRS)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            c_d.cpu(), c_ref.cpu(), rtol=1e-2, atol=5e-1)
        print("result verified", flush=True)

    return launch, check


def event_probe(fn, iters):
    """Measure a short per-launch GPU-event latency used to size the graph."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def capture_graph(fn, n_per_graph):
    """Capture ``n_per_graph`` launches after warming a side stream."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
        side.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            for _ in range(n_per_graph):
                fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    return graph


def run_benchmark(launch, M, N, K, args):
    """Time graph replays and report nominal GEMM throughput.

    Throughput uses ``2*M*N*K`` operations for every format. Packing and scale
    preparation happen before timing; only the GEMM launch is captured.
    """
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    probe_ms = event_probe(launch, args.probe_iters)
    n_per_graph = (
        max(1, int(args.graph_ms / max(probe_ms, 1e-6)))
        if args.iters_per_graph is None else args.iters_per_graph)
    if n_per_graph <= 0:
        raise ValueError("--iters-per-graph must be positive")
    graph = capture_graph(launch, n_per_graph)
    total_iters = args.replays * n_per_graph
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.replays):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    per_iter = elapsed / total_iters
    print(f"probe per-iter   : {probe_ms * 1e3:.2f} us")
    print(f"iters per graph  : {n_per_graph}")
    print(f"replays          : {args.replays}")
    print(f"total iters      : {total_iters}")
    print(f"total elapsed    : {elapsed:.6f} s")
    print(f"per-iter         : {per_iter * 1e6:.2f} us")
    print(f"TFLOPS           : {(2 * M * N * K) / per_iter / 1e12:.3f}")


def run_one(name, args):
    """Create, optionally verify, benchmark, and release one selected case."""
    print(f"\n=== {name.upper()} ===", flush=True)
    makers = {
        "bf16": make_bf16_case,
        "mxfp8": make_mxfp8_case,
        "fp8_mxfp4": make_fp8_mxfp4_case,
        "mxfp4": make_mxfp4_case,
    }
    launch, check = makers[name](args)
    if args.check:
        check()
    if args.benchmark:
        run_benchmark(launch, args.M, args.N, args.K, args)
    del launch, check
    gc.collect()
    torch.cuda.empty_cache()


def parse_args():
    """Parse the standalone correctness/benchmark command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run the four canonical performance-retained GFX1250 GEMMs")
    parser.add_argument(
        "--kernel", choices=(*KERNEL_NAMES, "all"), default="all")
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-K", type=int, default=65536)
    parser.add_argument(
        "--input-mode", choices=("random", "trig"),
        help=("input value pattern; defaults to trig for BF16/MXFP8 and random "
              "for FP8xMXFP4/MXFP4"))
    parser.add_argument(
        "--bf16-fp32-output", dest="bf16_output", action="store_false",
        help="Store BF16 kernel results as FP32 instead of the BF16 default")
    parser.set_defaults(bf16_output=True)
    parser.add_argument(
        "--fp8-mxfp4-num-buffers", type=int, choices=(2, 3), default=3,
        help="number of FP8xMXFP4 TDM ring slots")
    parser.add_argument(
        "--fp8-mxfp4-cluster-width", type=int, choices=(1, 2, 4), default=4,
        help="square FP8xMXFP4 CTA-cluster width")
    parser.add_argument(
        "--fp8-mxfp4-reuse-order", choices=("a", "b"), default="a",
        help="group output tiles to reuse A or B between cluster waves")
    parser.add_argument(
        "--fp8-mxfp4-group-size", type=int, choices=(1, 2, 4, 8),
        help="same-operand PID group size (default: 8 for K<=8192, else 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--probe-iters", type=int, default=20)
    parser.add_argument("--graph-ms", type=float, default=100.0)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--iters-per-graph", type=int)
    args = parser.parse_args()
    if not args.check and not args.benchmark:
        parser.error("select --check and/or --benchmark")
    if args.probe_iters <= 0 or args.replays <= 0 or args.warmup < 0:
        parser.error("benchmark iteration counts must be positive")
    return args


def main():
    args = parse_args()
    names = KERNEL_NAMES if args.kernel == "all" else (args.kernel,)
    for name in names:
        run_one(name, args)


if __name__ == "__main__":
    main()
