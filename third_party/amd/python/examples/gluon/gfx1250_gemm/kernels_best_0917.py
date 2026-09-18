"""0917 GFX1250 GEMM experiments.

BF16 issues refill before compute, arrives after the high LDS loads, and waits
after the high WMMA. MXFP8 uses an eight-stage variant with arrive in stage 5
and wait/refill in stage 7. MXFP4 applies the same eight-stage barrier/refill
placement. The remaining kernels start from the standalone 0913 set.
"""

import argparse
import gc

import numpy as np
import torch
import triton
from triton._C.libtriton.gluon_ir import make_cga_layout
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm

from pathlib import Path
import sys
import time

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


BF16_BLOCK_K = 128


NUM_WARPS = 8


CTA_M = 4


CTA_N = 4


GROUP_SIZE_M = 4


NUM_XCDS = 8


AGPR_ATTRS = (("amdgpu-agpr-alloc", "0,0"),)


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


@gluon.jit
def mxfp8_tdm_store_split_n2(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1,
        OUTPUT_CGA_LAYOUT: gl.constexpr, CTA_M: gl.constexpr,
        CTA_N: gl.constexpr):
    cta_m: gl.constexpr = 256
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[half_n, 8]], [CTA_M, CTA_N, cta_m, half_n],
        [3, 2, 1, 0], OUTPUT_CGA_LAYOUT)
    shared = gl.allocate_shared_memory(
        c_ptr.type.element_ty, [CTA_M, CTA_N, cta_m, half_n], shared_layout)
    output0 = acc0.reshape(
        (CTA_M, cta_m, CTA_N, half_n)).permute((0, 2, 1, 3))
    output1 = acc1.reshape(
        (CTA_M, cta_m, CTA_N, half_n)).permute((0, 2, 1, 3))
    desc = tdm.make_tensor_descriptor(
        base=c_ptr,
        shape=(M // cta_m, N // cta_n, cta_m, cta_n),
        strides=(cta_m * stride_cm, cta_n * stride_cn, stride_cm, stride_cn),
        block_shape=(CTA_M, CTA_N, cta_m, half_n), layout=shared_layout)
    base_m = pid_m * CTA_M
    base_n = pid_n * CTA_N
    shared.store(output0.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, 0], shared)
    tdm.async_wait(0)
    shared.store(output1.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, half_n], shared)
    tdm.async_wait(0)


@gluon.jit
def mxfp8_tdm_store_split_n2_pipelined(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1,
        OUTPUT_CGA_LAYOUT: gl.constexpr, CTA_M: gl.constexpr,
        CTA_N: gl.constexpr):
    cta_m: gl.constexpr = 256
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[half_n, 8]], [CTA_M, CTA_N, cta_m, half_n],
        [3, 2, 1, 0], OUTPUT_CGA_LAYOUT)
    shared0 = gl.allocate_shared_memory(
        c_ptr.type.element_ty,
        [CTA_M, CTA_N, cta_m, half_n], shared_layout)
    shared1 = gl.allocate_shared_memory(
        c_ptr.type.element_ty,
        [CTA_M, CTA_N, cta_m, half_n], shared_layout)
    output0 = acc0.reshape(
        (CTA_M, cta_m, CTA_N, half_n)).permute((0, 2, 1, 3))
    output1 = acc1.reshape(
        (CTA_M, cta_m, CTA_N, half_n)).permute((0, 2, 1, 3))
    desc = tdm.make_tensor_descriptor(
        base=c_ptr,
        shape=(M // cta_m, N // cta_n, cta_m, cta_n),
        strides=(cta_m * stride_cm, cta_n * stride_cn, stride_cm, stride_cn),
        block_shape=(CTA_M, CTA_N, cta_m, half_n), layout=shared_layout)
    base_m = pid_m * CTA_M
    base_n = pid_n * CTA_N
    shared0.store(output0.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, 0], shared0)
    shared1.store(output1.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, half_n], shared1)
    tdm.async_wait(0)


@gluon.jit
def mxfp8_tdm_store_split_m2(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1,
        OUTPUT_CGA_LAYOUT: gl.constexpr, CTA_M: gl.constexpr,
        CTA_N: gl.constexpr):
    cta_m: gl.constexpr = 256
    cta_n: gl.constexpr = 256
    half_m: gl.constexpr = 128
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[cta_n, 8]], [CTA_M, CTA_N, half_m, cta_n],
        [3, 2, 1, 0], OUTPUT_CGA_LAYOUT)
    shared = gl.allocate_shared_memory(
        c_ptr.type.element_ty,
        [CTA_M, CTA_N, half_m, cta_n], shared_layout)
    output0 = acc0.reshape(
        (CTA_M, half_m, CTA_N, cta_n)).permute((0, 2, 1, 3))
    output1 = acc1.reshape(
        (CTA_M, half_m, CTA_N, cta_n)).permute((0, 2, 1, 3))
    desc = tdm.make_tensor_descriptor(
        base=c_ptr,
        shape=(M // cta_m, N // cta_n, cta_m, cta_n),
        strides=(cta_m * stride_cm, cta_n * stride_cn, stride_cm, stride_cn),
        block_shape=(CTA_M, CTA_N, half_m, cta_n), layout=shared_layout)
    base_m = pid_m * CTA_M
    base_n = pid_n * CTA_N
    shared.store(output0.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, 0], shared)
    tdm.async_wait(0)
    shared.store(output1.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, half_m, 0], shared)
    tdm.async_wait(0)


@gluon.jit
def bf16_issue_loads(a_desc, b_desc, a_dst, b_dst, BLOCK_K: gl.constexpr):
    tdm.async_load(
        a_desc, [0, 0], a_dst, warp_used_hint=0b00001111)
    tdm.async_load(
        b_desc, [0, 0], b_dst, warp_used_hint=0b00001111)
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K])
    return a_desc, b_desc


@gluon.jit
def bf16_issue_loads_without_update(a_desc, b_desc, a_dst, b_dst):
    tdm.async_load(
        a_desc, [0, 0], a_dst, warp_used_hint=0b00001111)
    tdm.async_load(
        b_desc, [0, 0], b_dst, warp_used_hint=0b00001111)


@gluon.jit
def bf16_consume_slot(
        a_buf, b_buf, slot, acc, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        BLOCK_K: gl.constexpr):
    half_k: gl.constexpr = BLOCK_K // 2
    with gl.amd.warp_pipeline_stage("stage0"):
        a = a_buf.index(slot).slice(0, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(
            0, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1"):
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    with gl.amd.warp_pipeline_stage("stage0"):
        a = a_buf.index(slot).slice(
            half_k, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(
            half_k, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1"):
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    return acc


@gluon.jit
def bf16_cluster_consume_and_refill(
        a_buf, b_buf, slot, refill_slot, a_desc, b_desc, acc,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        BLOCK_K: gl.constexpr):
    half_k: gl.constexpr = BLOCK_K // 2
    tdm.async_wait(0)
    with gl.amd.warp_pipeline_stage("stage0"):
        bf16_issue_loads_without_update(
            a_desc, b_desc, a_buf.index(refill_slot),
            b_buf.index(refill_slot))
        a = a_buf.index(slot).slice(0, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(
            0, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1"):
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    with gl.amd.warp_pipeline_stage("stage0"):
        a = a_buf.index(slot).slice(
            half_k, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(
            half_k, half_k, 1).permute([1, 0]).load(layout=DOT_B)
        a_desc = tdm.update_tensor_descriptor(
            a_desc, add_offsets=[0, BLOCK_K])
        b_desc = tdm.update_tensor_descriptor(
            b_desc, add_offsets=[0, BLOCK_K])
        gl.amd.gfx1250.cluster.arrive()
    with gl.amd.warp_pipeline_stage("stage1"):
        acc = gl.amd.gfx1250.wmma(a, b, acc)
        gl.amd.gfx1250.cluster.wait()
    return a_desc, b_desc, acc


@gluon.jit
def bf16_8stage_split_b(
        b, DOT_B: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_N: gl.constexpr):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    b_grid = b.reshape((64, CTA_N, cta_n))
    b0 = gl.convert_layout(gl.amd.slice(
        b_grid, [64, CTA_N, half_n], [0, 0, 0]
    ).reshape((64, packed_n)), DOT_B, assert_trivial=True)
    b1 = gl.convert_layout(gl.amd.slice(
        b_grid, [64, CTA_N, half_n], [0, 0, half_n]
    ).reshape((64, packed_n)), DOT_B, assert_trivial=True)
    return b0, b1


@gluon.jit
def bf16_kernelc_2pf_bk128_fused_cluster_4x4_gfx1250(
        a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk,
        stride_bn, stride_cm, stride_cn, GRID_MN: gl.constexpr,
        SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr,
        WMMA_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, CTA_M: gl.constexpr, CTA_N: gl.constexpr):
    block_m: gl.constexpr = BLOCK_M
    block_n: gl.constexpr = BLOCK_N
    block_k: gl.constexpr = 128
    gl.static_assert(
        a_ptr.type.element_ty.is_bf16() and b_ptr.type.element_ty.is_bf16())
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    gl.static_assert(BLOCK_M == CTA_M * 256)
    gl.static_assert(BLOCK_N == CTA_N * 256)
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
    # Only slot 0 is prefetched; tile t's refill supplies tile t + 1.
    a_desc, b_desc = bf16_issue_loads(
        a_desc, b_desc, a_buf.index(0), b_buf.index(0), block_k)
    acc = gl.zeros((block_m, block_n), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, block_k)
    gl.assume(iter_max >= 2)
    gl.assume((iter_max % 2) == 0)
    # Tiles 0 .. iter_max - 2 refill; the unrolled body walks them in pairs and
    # the last refilling tile is peeled so the final tile can skip its refill.
    for _ in range(0, (iter_max - 2) // 2):
        a_desc, b_desc, acc = bf16_cluster_consume_and_refill(
            a_buf, b_buf, 0, 1, a_desc, b_desc, acc, dot_a, dot_b, block_k)
        a_desc, b_desc, acc = bf16_cluster_consume_and_refill(
            a_buf, b_buf, 1, 0, a_desc, b_desc, acc, dot_a, dot_b, block_k)
    a_desc, b_desc, acc = bf16_cluster_consume_and_refill(
        a_buf, b_buf, 0, 1, a_desc, b_desc, acc, dot_a, dot_b, block_k)
    tdm.async_wait(0)
    acc = bf16_consume_slot(
        a_buf, b_buf, 1, acc, dot_a, dot_b, block_k)
    snapshot_tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        WMMA_LAYOUT, block_m, block_n, CTA_N)


def build_bf16_layouts(cluster_m=4, cluster_n=4):
    """Build matching distributed LDS and accumulator layouts for BF16.

    Layout construction is intentionally two-pass. First derive the local
    256x256-per-CTA WMMA ownership, then graft its operand CGA bases onto the
    padded shared layouts. This makes every CTA load the LDS partition consumed
    by its accumulator partition.
    """
    block_m = 256 * cluster_m
    block_n = 256 * cluster_n
    cga_layout_c = make_cga_layout(
        [cluster_m, cluster_n], [cluster_m, cluster_n], [0, 1])
    slice_m = block_m // cluster_m
    slice_n = block_n // cluster_n
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[BF16_BLOCK_K, 8]], [block_m, BF16_BLOCK_K], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[BF16_BLOCK_K, 8]], [block_n, BF16_BLOCK_K], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, block_n, local_a, local_b, NUM_WARPS, [16, 16, 32],
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
        [[BF16_BLOCK_K, 8]], [block_m, BF16_BLOCK_K], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[BF16_BLOCK_K, 8]], [block_n, BF16_BLOCK_K], [1, 0], cga_b)
    # One group per physical partition is the retained partitioned layout.
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        slice_m, slice_n, padded_a, padded_b, NUM_WARPS, [16, 16, 32],
        a_transposed=False, b_transposed=True, slice_m=slice_m,
        slice_n=slice_n, transposed=True)
    return shared_a, shared_b, wmma


MXFP8_NUM_SLOTS = gl.constexpr(2)


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
def mxfp8_load_a_scale_b128(
        scale_buffer, slot, LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr):
    # The width-eight host packing makes K[0:8] contiguous before the M/32
    # register repetition. Together they form one 16-byte fragment per lane.
    scale_slice = scale_buffer.index(slot).reshape(
        (BLOCK_M // 128, 1, 32, 4, 8)
    ).permute((0, 3, 2, 1, 4)).reshape((BLOCK_M, 8))
    return scale_slice.load(layout=LAYOUT)


@gluon.jit
def mxfp8_load_b_scale(
        scale_buffer, slot, start_k: gl.constexpr, LAYOUT: gl.constexpr,
        BLOCK_N: gl.constexpr, CONTIGUOUS: gl.constexpr):
    if CONTIGUOUS:
        return scale_buffer.index(slot).slice(
            start_k, 4, 1).load(layout=LAYOUT)
    return mxfp8_load_scale(
        scale_buffer, slot, start_k, LAYOUT, BLOCK_N, 8, 4)


@gluon.jit
def mxfp8_split_a_scale(
        scale, SCALE_A_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr):
    scale0 = gl.convert_layout(
        gl.amd.slice(scale, [BLOCK_M, 4], [0, 0]),
        SCALE_A_LAYOUT, assert_trivial=True)
    scale1 = gl.convert_layout(
        gl.amd.slice(scale, [BLOCK_M, 4], [0, 4]),
        SCALE_A_LAYOUT, assert_trivial=True)
    return scale0, scale1


@gluon.jit
def mxfp8_issue_leading4_data(
        a_desc, b_desc, a_buf, b_buf, slot):
    # Both standalone copies use all four leading warps. Narrowing A to two
    # warps is legal, but the two-descriptor experiment that fused A with the
    # scales measured slower than this three-descriptor schedule.
    tdm.async_load(
        a_desc, dest=a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(
        b_desc, dest=b_buf.index(slot), warp_used_hint=0b00001111)
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 256])
    return a_desc, b_desc


@gluon.jit
def mxfp8_issue_leading4_data_without_update(
        a_desc, b_desc, a_buf, b_buf, slot):
    tdm.async_load(
        a_desc, dest=a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(
        b_desc, dest=b_buf.index(slot), warp_used_hint=0b00001111)


@gluon.jit
def mxfp8_issue_leading4_scale(
        as_desc, bs_desc, as_buf, bs_buf, slot):
    tdm.async_load_fused([
        (as_desc, as_buf.index(slot), 0b00000011),
        (bs_desc, bs_buf.index(slot), 0b00001100),
    ])
    as_desc = tdm.update_tensor_descriptor(
        as_desc, add_offsets=[0, 1024])
    bs_desc = tdm.update_tensor_descriptor(
        bs_desc, add_offsets=[0, 1024])
    return as_desc, bs_desc


@gluon.jit
def mxfp8_issue_leading4_scale_without_update(
        as_desc, bs_desc, as_buf, bs_buf, slot):
    tdm.async_load_fused([
        (as_desc, as_buf.index(slot), 0b00000011),
        (bs_desc, bs_buf.index(slot), 0b00001100),
    ])


@gluon.jit
def mxfp8_issue_pair_scale(
        as_desc, bs_desc, as_buf, bs_buf, pair_slot):
    tdm.async_load_fused([
        (as_desc, as_buf.index(pair_slot), 0b00000011),
        (bs_desc, bs_buf.index(pair_slot), 0b00001100),
    ])
    as_desc = tdm.update_tensor_descriptor(
        as_desc, add_offsets=[0, 2048])
    bs_desc = tdm.update_tensor_descriptor(
        bs_desc, add_offsets=[0, 2048])
    return as_desc, bs_desc


@gluon.jit
def mxfp8_issue_two_tdm(
        a_desc, b_desc, as_desc, bs_desc,
        a_buf, b_buf, as_buf, bs_buf, slot,
        B_SCALE_CONTIGUOUS: gl.constexpr):
    tdm.async_load(
        b_desc, dest=b_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load_fused([
        (a_desc, a_buf.index(slot), 0b00000011),
        (as_desc, as_buf.index(slot), 0b00000100),
        (bs_desc, bs_buf.index(slot), 0b00001000),
    ])
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 256])
    as_desc = tdm.update_tensor_descriptor(
        as_desc, add_offsets=[0, 1024])
    bs_desc = tdm.update_tensor_descriptor(
        bs_desc, add_offsets=[0, 8 if B_SCALE_CONTIGUOUS else 1024])
    return a_desc, b_desc, as_desc, bs_desc


@gluon.jit
def mxfp8_issue_leading4_without_update(
        a_desc, b_desc, as_desc, bs_desc,
        a_buf, b_buf, as_buf, bs_buf, slot):
    tdm.async_load(
        a_desc, dest=a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(
        b_desc, dest=b_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load_fused([
        (as_desc, as_buf.index(slot), 0b00000011),
        (bs_desc, bs_buf.index(slot), 0b00001100),
    ])


@gluon.jit
def mxfp8_advance_descriptors(
        a_desc, b_desc, as_desc, bs_desc,
        DATA_STEP: gl.constexpr, SCALE_STEP: gl.constexpr):
    a_desc = tdm.update_tensor_descriptor(
        a_desc, add_offsets=[0, DATA_STEP])
    b_desc = tdm.update_tensor_descriptor(
        b_desc, add_offsets=[0, DATA_STEP])
    as_desc = tdm.update_tensor_descriptor(
        as_desc, add_offsets=[0, SCALE_STEP])
    bs_desc = tdm.update_tensor_descriptor(
        bs_desc, add_offsets=[0, SCALE_STEP])
    return a_desc, b_desc, as_desc, bs_desc


@gluon.jit
def mxfp8_bk256_opt_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, slot, refill_slot, a_desc, b_desc,
        as_desc, bs_desc, acc0, acc1, DOT_A: gl.constexpr,
        DOT_B: gl.constexpr, DOT_B_LOAD: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_A_LOAD: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr,
        SCALE_B_LOAD: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, CTA_N: gl.constexpr,
        STAGE8_REFILL: gl.constexpr, TWO_TDM: gl.constexpr,
        B_SCALE_CONTIGUOUS: gl.constexpr,
        A_SCALE_COMBINED: gl.constexpr,
        DESC_UPDATE_STAGE5: gl.constexpr,
        SCALE_LOAD_STAGE1: gl.constexpr,
        B_LOAD_STAGE1: gl.constexpr,
        STAGE6_WAIT_REFILL_FIRST: gl.constexpr,
        EARLY_CLUSTER_SYNC: gl.constexpr,
        STAGE7_CLUSTER_SYNC: gl.constexpr,
        SEPARATE_SLOT_DESCRIPTORS: gl.constexpr,
        STAGE5_ADVANCE_WAIT_REFILL_FIRST: gl.constexpr,
        ARRIVE_STAGE1_WAIT_STAGE7: gl.constexpr):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    gl.static_assert(
        not DESC_UPDATE_STAGE5 or (STAGE8_REFILL and not TWO_TDM))
    gl.static_assert(
        not STAGE6_WAIT_REFILL_FIRST
        or (STAGE8_REFILL and not TWO_TDM and not DESC_UPDATE_STAGE5))
    gl.static_assert(
        not EARLY_CLUSTER_SYNC
        or (STAGE8_REFILL and not STAGE7_CLUSTER_SYNC
            and not STAGE5_ADVANCE_WAIT_REFILL_FIRST))
    gl.static_assert(not STAGE7_CLUSTER_SYNC or STAGE6_WAIT_REFILL_FIRST)
    gl.static_assert(not (EARLY_CLUSTER_SYNC and STAGE7_CLUSTER_SYNC))
    gl.static_assert(
        not SEPARATE_SLOT_DESCRIPTORS
        or (STAGE8_REFILL and not TWO_TDM and not DESC_UPDATE_STAGE5
            and not STAGE6_WAIT_REFILL_FIRST))
    gl.static_assert(
        not STAGE5_ADVANCE_WAIT_REFILL_FIRST
        or (STAGE8_REFILL and DESC_UPDATE_STAGE5 and not TWO_TDM
            and not STAGE6_WAIT_REFILL_FIRST and not EARLY_CLUSTER_SYNC
            and not STAGE7_CLUSTER_SYNC and not SEPARATE_SLOT_DESCRIPTORS))
    gl.static_assert(
        not ARRIVE_STAGE1_WAIT_STAGE7
        or (STAGE8_REFILL and not EARLY_CLUSTER_SYNC
            and not STAGE7_CLUSTER_SYNC and not STAGE6_WAIT_REFILL_FIRST
            and not STAGE5_ADVANCE_WAIT_REFILL_FIRST))
    gl.static_assert(not SCALE_LOAD_STAGE1 or not A_SCALE_COMBINED)
    tdm.async_wait(2 if TWO_TDM else 3)
    with gl.amd.warp_pipeline_stage(
            "stage0_load_k0", phase_gap=2):
        a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        if not SCALE_LOAD_STAGE1:
            if A_SCALE_COMBINED:
                as_full = mxfp8_load_a_scale_b128(
                    as_buf, slot, SCALE_A_LOAD, BLOCK_M)
                as0, as1 = mxfp8_split_a_scale(
                    as_full, SCALE_A_LAYOUT, BLOCK_M)
            else:
                as0 = mxfp8_load_scale(
                    as_buf, slot, 0, SCALE_A_LAYOUT, BLOCK_M, 8, 4)
        if not B_LOAD_STAGE1:
            b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
                0, 128, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
            b0_grid = b0.reshape((128, CTA_N, cta_n))
            b0_lo = gl.convert_layout(gl.amd.slice(
                b0_grid, [128, CTA_N, half_n], [0, 0, 0]
            ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
            b0_hi = gl.convert_layout(gl.amd.slice(
                b0_grid, [128, CTA_N, half_n], [0, 0, half_n]
            ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        if not SCALE_LOAD_STAGE1:
            bs0 = mxfp8_load_b_scale(
                bs_buf, slot, 0, SCALE_B_LOAD, BLOCK_N, B_SCALE_CONTIGUOUS)
            bs0_grid = bs0.reshape((CTA_N, cta_n, 4))
            bs0_lo = gl.convert_layout(gl.amd.slice(
                bs0_grid, [CTA_N, half_n, 4], [0, 0, 0]
            ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
            bs0_hi = gl.convert_layout(gl.amd.slice(
                bs0_grid, [CTA_N, half_n, 4], [0, half_n, 0]
            ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
    with gl.amd.warp_pipeline_stage("stage1_bubble"):
        if SCALE_LOAD_STAGE1:
            as0 = mxfp8_load_scale(
                as_buf, slot, 0, SCALE_A_LAYOUT, BLOCK_M, 8, 4)
            bs0 = mxfp8_load_b_scale(
                bs_buf, slot, 0, SCALE_B_LOAD, BLOCK_N, B_SCALE_CONTIGUOUS)
            bs0_grid = bs0.reshape((CTA_N, cta_n, 4))
            bs0_lo = gl.convert_layout(gl.amd.slice(
                bs0_grid, [CTA_N, half_n, 4], [0, 0, 0]
            ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
            bs0_hi = gl.convert_layout(gl.amd.slice(
                bs0_grid, [CTA_N, half_n, 4], [0, half_n, 0]
            ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        if B_LOAD_STAGE1:
            b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
                0, 128, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
            b0_grid = b0.reshape((128, CTA_N, cta_n))
            b0_lo = gl.convert_layout(gl.amd.slice(
                b0_grid, [128, CTA_N, half_n], [0, 0, 0]
            ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
            b0_hi = gl.convert_layout(gl.amd.slice(
                b0_grid, [128, CTA_N, half_n], [0, 0, half_n]
            ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        if EARLY_CLUSTER_SYNC or ARRIVE_STAGE1_WAIT_STAGE7:
            gl.amd.gfx1250.cluster.arrive()
    with gl.amd.warp_pipeline_stage("stage2_compute_k0_n0"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0_lo, bs0_lo, "e4m3", acc0)
    with gl.amd.warp_pipeline_stage("stage3_compute_k0_n1"):
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0_hi, bs0_hi, "e4m3", acc1)
    with gl.amd.warp_pipeline_stage("stage4_load_k1"):
        a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        if not A_SCALE_COMBINED and not SCALE_LOAD_STAGE1:
            as1 = mxfp8_load_scale(
                as_buf, slot, 4, SCALE_A_LAYOUT, BLOCK_M, 8, 4)
        b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        b1_grid = b1.reshape((128, CTA_N, cta_n))
        b1_lo = gl.convert_layout(gl.amd.slice(
            b1_grid, [128, CTA_N, half_n], [0, 0, 0]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        b1_hi = gl.convert_layout(gl.amd.slice(
            b1_grid, [128, CTA_N, half_n], [0, 0, half_n]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        if not SCALE_LOAD_STAGE1:
            bs1 = mxfp8_load_b_scale(
                bs_buf, slot, 4, SCALE_B_LOAD, BLOCK_N, B_SCALE_CONTIGUOUS)
            bs1_grid = bs1.reshape((CTA_N, cta_n, 4))
            bs1_lo = gl.convert_layout(gl.amd.slice(
                bs1_grid, [CTA_N, half_n, 4], [0, 0, 0]
            ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
            bs1_hi = gl.convert_layout(gl.amd.slice(
                bs1_grid, [CTA_N, half_n, 4], [0, half_n, 0]
            ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
    next_a_desc = a_desc
    next_b_desc = b_desc
    next_as_desc = as_desc
    next_bs_desc = bs_desc
    with gl.amd.warp_pipeline_stage("stage5_bubble"):
        if SCALE_LOAD_STAGE1:
            as1 = mxfp8_load_scale(
                as_buf, slot, 4, SCALE_A_LAYOUT, BLOCK_M, 8, 4)
            bs1 = mxfp8_load_b_scale(
                bs_buf, slot, 4, SCALE_B_LOAD, BLOCK_N, B_SCALE_CONTIGUOUS)
            bs1_grid = bs1.reshape((CTA_N, cta_n, 4))
            bs1_lo = gl.convert_layout(gl.amd.slice(
                bs1_grid, [CTA_N, half_n, 4], [0, 0, 0]
            ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
            bs1_hi = gl.convert_layout(gl.amd.slice(
                bs1_grid, [CTA_N, half_n, 4], [0, half_n, 0]
            ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        if EARLY_CLUSTER_SYNC:
            gl.amd.gfx1250.cluster.wait()
        elif (STAGE8_REFILL and not STAGE7_CLUSTER_SYNC
                and not ARRIVE_STAGE1_WAIT_STAGE7):
            gl.amd.gfx1250.cluster.arrive()
        if DESC_UPDATE_STAGE5:
            next_a_desc = tdm.update_tensor_descriptor(
                a_desc, add_offsets=[0, 256])
            next_b_desc = tdm.update_tensor_descriptor(
                b_desc, add_offsets=[0, 256])
            next_as_desc = tdm.update_tensor_descriptor(
                as_desc, add_offsets=[0, 1024])
            next_bs_desc = tdm.update_tensor_descriptor(
                bs_desc, add_offsets=[0, 1024])
        if STAGE5_ADVANCE_WAIT_REFILL_FIRST:
            gl.amd.gfx1250.cluster.wait()
    with gl.amd.warp_pipeline_stage("stage6_compute_k1_n0"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1_lo, bs1_lo, "e4m3", acc0)
        if (STAGE6_WAIT_REFILL_FIRST
                and not EARLY_CLUSTER_SYNC and not STAGE7_CLUSTER_SYNC):
            gl.amd.gfx1250.cluster.wait()
    if STAGE8_REFILL:
        with gl.amd.warp_pipeline_stage(
                "stage7_arrive_compute_k1_n1"):
            if STAGE5_ADVANCE_WAIT_REFILL_FIRST:
                mxfp8_issue_leading4_without_update(
                    a_desc, b_desc, as_desc, bs_desc,
                    a_buf, b_buf, as_buf, bs_buf, refill_slot)
                acc1 = gl.amd.gfx1250.wmma_scaled(
                    a1, as1, "e4m3", b1_hi, bs1_hi, "e4m3", acc1)
                a_desc = next_a_desc
                b_desc = next_b_desc
                as_desc = next_as_desc
                bs_desc = next_bs_desc
            elif STAGE6_WAIT_REFILL_FIRST:
                if STAGE7_CLUSTER_SYNC:
                    gl.amd.gfx1250.cluster.arrive()
                    gl.amd.gfx1250.cluster.wait()
                mxfp8_issue_leading4_without_update(
                    a_desc, b_desc, as_desc, bs_desc,
                    a_buf, b_buf, as_buf, bs_buf, refill_slot)
                acc1 = gl.amd.gfx1250.wmma_scaled(
                    a1, as1, "e4m3", b1_hi, bs1_hi, "e4m3", acc1)
                a_desc = tdm.update_tensor_descriptor(
                    a_desc, add_offsets=[0, 256])
                b_desc = tdm.update_tensor_descriptor(
                    b_desc, add_offsets=[0, 256])
                as_desc = tdm.update_tensor_descriptor(
                    as_desc, add_offsets=[0, 1024])
                bs_desc = tdm.update_tensor_descriptor(
                    bs_desc, add_offsets=[0, 1024])
            else:
                acc1 = gl.amd.gfx1250.wmma_scaled(
                    a1, as1, "e4m3", b1_hi, bs1_hi, "e4m3", acc1)
                if not EARLY_CLUSTER_SYNC:
                    gl.amd.gfx1250.cluster.wait()
                if TWO_TDM:
                    a_desc, b_desc, as_desc, bs_desc = mxfp8_issue_two_tdm(
                        a_desc, b_desc, as_desc, bs_desc,
                        a_buf, b_buf, as_buf, bs_buf, refill_slot,
                        B_SCALE_CONTIGUOUS)
                elif DESC_UPDATE_STAGE5:
                    mxfp8_issue_leading4_without_update(
                        a_desc, b_desc, as_desc, bs_desc,
                        a_buf, b_buf, as_buf, bs_buf, refill_slot)
                    a_desc = next_a_desc
                    b_desc = next_b_desc
                    as_desc = next_as_desc
                    bs_desc = next_bs_desc
                else:
                    if SEPARATE_SLOT_DESCRIPTORS:
                        mxfp8_issue_leading4_without_update(
                            a_desc, b_desc, as_desc, bs_desc,
                            a_buf, b_buf, as_buf, bs_buf, refill_slot)
                        a_desc, b_desc, as_desc, bs_desc = (
                            mxfp8_advance_descriptors(
                                a_desc, b_desc, as_desc, bs_desc, 512, 2048))
                    else:
                        a_desc, b_desc = mxfp8_issue_leading4_data(
                            a_desc, b_desc, a_buf, b_buf, refill_slot)
                        as_desc, bs_desc = mxfp8_issue_leading4_scale(
                            as_desc, bs_desc, as_buf, bs_buf, refill_slot)
    else:
        with gl.amd.warp_pipeline_stage(
                "stage7_arrive_compute_k1_n1_wait_refill"):
            gl.amd.gfx1250.cluster.arrive()
            acc1 = gl.amd.gfx1250.wmma_scaled(
                a1, as1, "e4m3", b1_hi, bs1_hi, "e4m3", acc1)
            gl.amd.gfx1250.cluster.wait()
            if TWO_TDM:
                a_desc, b_desc, as_desc, bs_desc = mxfp8_issue_two_tdm(
                    a_desc, b_desc, as_desc, bs_desc,
                    a_buf, b_buf, as_buf, bs_buf, refill_slot,
                    B_SCALE_CONTIGUOUS)
            else:
                a_desc, b_desc = mxfp8_issue_leading4_data(
                    a_desc, b_desc, a_buf, b_buf, refill_slot)
                as_desc, bs_desc = mxfp8_issue_leading4_scale(
                    as_desc, bs_desc, as_buf, bs_buf, refill_slot)
    return a_desc, b_desc, as_desc, bs_desc, acc0, acc1


@gluon.jit
def mxfp8_bk256_consume_and_refill_paired_scales(
        a_buf, b_buf, as_buf, bs_buf, data_slot, scale_slot,
        next_scale_slot, scale_start, a_desc, b_desc, as_desc, bs_desc,
        acc0, acc1, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        DOT_B_LOAD: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
        SCALE_A_LOAD: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        SCALE_B_LOAD: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, CTA_N: gl.constexpr,
        WAITCNT: gl.constexpr, REFILL_SCALES: gl.constexpr,
        SCALE_TILE_WIDTH: gl.constexpr,
        REFILL_DATA_BEFORE_WMMA: gl.constexpr,
        STAGE5_ADVANCE_REFILL_ALL_BEFORE_WMMA: gl.constexpr,
        STAGE5_ADVANCE_ORIGINAL_STAGE7_ORDER: gl.constexpr):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    tdm.async_wait(WAITCNT)
    with gl.amd.warp_pipeline_stage(
            "pair_stage0_load_k0", phase_gap=2):
        a0 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        as0 = mxfp8_load_scale(
            as_buf, scale_slot, scale_start,
            SCALE_A_LOAD, BLOCK_M, SCALE_TILE_WIDTH, 4)
        as0 = gl.convert_layout(as0, SCALE_A_LAYOUT, assert_trivial=True)
        bs0 = mxfp8_load_scale(
            bs_buf, scale_slot, scale_start,
            SCALE_B_LOAD, BLOCK_N, SCALE_TILE_WIDTH, 4)
        bs0_grid = bs0.reshape((CTA_N, cta_n, 4))
        bs0_lo = gl.convert_layout(gl.amd.slice(
            bs0_grid, [CTA_N, half_n, 4], [0, 0, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        bs0_hi = gl.convert_layout(gl.amd.slice(
            bs0_grid, [CTA_N, half_n, 4], [0, half_n, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
    with gl.amd.warp_pipeline_stage("pair_stage1_load_b_k0"):
        b0 = b_buf.index(data_slot).slice(0, BLOCK_N, 0).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        b0_grid = b0.reshape((128, CTA_N, cta_n))
        b0_lo = gl.convert_layout(gl.amd.slice(
            b0_grid, [128, CTA_N, half_n], [0, 0, 0]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        b0_hi = gl.convert_layout(gl.amd.slice(
            b0_grid, [128, CTA_N, half_n], [0, 0, half_n]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
    with gl.amd.warp_pipeline_stage(
            "pair_stage2_compute_k0_n0"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0_lo, bs0_lo, "e4m3", acc0)
    with gl.amd.warp_pipeline_stage(
            "pair_stage3_compute_k0_n1"):
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0_hi, bs0_hi, "e4m3", acc1)
    with gl.amd.warp_pipeline_stage(
            "pair_stage4_load_k1"):
        a1 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        as1 = mxfp8_load_scale(
            as_buf, scale_slot, scale_start + 4,
            SCALE_A_LOAD, BLOCK_M, SCALE_TILE_WIDTH, 4)
        as1 = gl.convert_layout(as1, SCALE_A_LAYOUT, assert_trivial=True)
        b1 = b_buf.index(data_slot).slice(0, BLOCK_N, 0).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        b1_grid = b1.reshape((128, CTA_N, cta_n))
        b1_lo = gl.convert_layout(gl.amd.slice(
            b1_grid, [128, CTA_N, half_n], [0, 0, 0]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        b1_hi = gl.convert_layout(gl.amd.slice(
            b1_grid, [128, CTA_N, half_n], [0, 0, half_n]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        bs1 = mxfp8_load_scale(
            bs_buf, scale_slot, scale_start + 4,
            SCALE_B_LOAD, BLOCK_N, SCALE_TILE_WIDTH, 4)
        bs1_grid = bs1.reshape((CTA_N, cta_n, 4))
        bs1_lo = gl.convert_layout(gl.amd.slice(
            bs1_grid, [CTA_N, half_n, 4], [0, 0, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        bs1_hi = gl.convert_layout(gl.amd.slice(
            bs1_grid, [CTA_N, half_n, 4], [0, half_n, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
    next_a_desc = a_desc
    next_b_desc = b_desc
    next_as_desc = as_desc
    next_bs_desc = bs_desc
    with gl.amd.warp_pipeline_stage("pair_stage5_arrive"):
        gl.amd.gfx1250.cluster.arrive()
        if (STAGE5_ADVANCE_REFILL_ALL_BEFORE_WMMA
                or STAGE5_ADVANCE_ORIGINAL_STAGE7_ORDER):
            next_a_desc = tdm.update_tensor_descriptor(
                a_desc, add_offsets=[0, 256])
            next_b_desc = tdm.update_tensor_descriptor(
                b_desc, add_offsets=[0, 256])
            if REFILL_SCALES:
                next_as_desc = tdm.update_tensor_descriptor(
                    as_desc, add_offsets=[0, 1024])
                next_bs_desc = tdm.update_tensor_descriptor(
                    bs_desc, add_offsets=[0, 1024])
    with gl.amd.warp_pipeline_stage(
            "pair_stage6_compute_k1_n0"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1_lo, bs1_lo, "e4m3", acc0)
    with gl.amd.warp_pipeline_stage(
            "pair_stage7_compute_k1_n1_refill"):
        if STAGE5_ADVANCE_REFILL_ALL_BEFORE_WMMA:
            gl.amd.gfx1250.cluster.wait()
            mxfp8_issue_leading4_data_without_update(
                a_desc, b_desc, a_buf, b_buf, data_slot)
            if REFILL_SCALES:
                mxfp8_issue_leading4_scale_without_update(
                    as_desc, bs_desc, as_buf, bs_buf, next_scale_slot)
            acc1 = gl.amd.gfx1250.wmma_scaled(
                a1, as1, "e4m3", b1_hi, bs1_hi, "e4m3", acc1)
            a_desc = next_a_desc
            b_desc = next_b_desc
            as_desc = next_as_desc
            bs_desc = next_bs_desc
        elif STAGE5_ADVANCE_ORIGINAL_STAGE7_ORDER:
            acc1 = gl.amd.gfx1250.wmma_scaled(
                a1, as1, "e4m3", b1_hi, bs1_hi, "e4m3", acc1)
            gl.amd.gfx1250.cluster.wait()
            mxfp8_issue_leading4_data_without_update(
                a_desc, b_desc, a_buf, b_buf, data_slot)
            if REFILL_SCALES:
                mxfp8_issue_leading4_scale_without_update(
                    as_desc, bs_desc, as_buf, bs_buf, next_scale_slot)
            a_desc = next_a_desc
            b_desc = next_b_desc
            as_desc = next_as_desc
            bs_desc = next_bs_desc
        elif REFILL_DATA_BEFORE_WMMA:
            gl.amd.gfx1250.cluster.wait()
            mxfp8_issue_leading4_data_without_update(
                a_desc, b_desc, a_buf, b_buf, data_slot)
            acc1 = gl.amd.gfx1250.wmma_scaled(
                a1, as1, "e4m3", b1_hi, bs1_hi, "e4m3", acc1)
            a_desc = tdm.update_tensor_descriptor(
                a_desc, add_offsets=[0, 256])
            b_desc = tdm.update_tensor_descriptor(
                b_desc, add_offsets=[0, 256])
        else:
            acc1 = gl.amd.gfx1250.wmma_scaled(
                a1, as1, "e4m3", b1_hi, bs1_hi, "e4m3", acc1)
            gl.amd.gfx1250.cluster.wait()
            a_desc, b_desc = mxfp8_issue_leading4_data(
                a_desc, b_desc, a_buf, b_buf, data_slot)
        if (REFILL_SCALES
                and not STAGE5_ADVANCE_REFILL_ALL_BEFORE_WMMA
                and not STAGE5_ADVANCE_ORIGINAL_STAGE7_ORDER):
            if SCALE_TILE_WIDTH == 16:
                as_desc, bs_desc = mxfp8_issue_pair_scale(
                    as_desc, bs_desc, as_buf, bs_buf, next_scale_slot)
            else:
                as_desc, bs_desc = mxfp8_issue_leading4_scale(
                    as_desc, bs_desc, as_buf, bs_buf, next_scale_slot)
    return a_desc, b_desc, as_desc, bs_desc, acc0, acc1


@gluon.jit
def mxfp8_bk256_consume_tail_paired_scales(
        a_buf, b_buf, as_buf, bs_buf, data_slot, scale_slot, scale_start,
        acc0, acc1, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        DOT_B_LOAD: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
        SCALE_A_LOAD: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        SCALE_B_LOAD: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, CTA_N: gl.constexpr,
        SCALE_TILE_WIDTH: gl.constexpr):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    with gl.amd.warp_pipeline_stage("pair_tail_load_k0"):
        a0 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        as0_load = mxfp8_load_scale(
            as_buf, scale_slot, scale_start,
            SCALE_A_LOAD, BLOCK_M, SCALE_TILE_WIDTH, 4)
        as0 = gl.convert_layout(
            as0_load, SCALE_A_LAYOUT, assert_trivial=True)
        b0 = b_buf.index(data_slot).slice(0, BLOCK_N, 0).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        bs0 = mxfp8_load_scale(
            bs_buf, scale_slot, scale_start,
            SCALE_B_LOAD, BLOCK_N, SCALE_TILE_WIDTH, 4)
        b0_grid = b0.reshape((128, CTA_N, cta_n))
        b0_lo = gl.convert_layout(gl.amd.slice(
            b0_grid, [128, CTA_N, half_n], [0, 0, 0]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        b0_hi = gl.convert_layout(gl.amd.slice(
            b0_grid, [128, CTA_N, half_n], [0, 0, half_n]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        bs0_grid = bs0.reshape((CTA_N, cta_n, 4))
        bs0_lo = gl.convert_layout(gl.amd.slice(
            bs0_grid, [CTA_N, half_n, 4], [0, 0, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        bs0_hi = gl.convert_layout(gl.amd.slice(
            bs0_grid, [CTA_N, half_n, 4], [0, half_n, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
    with gl.amd.warp_pipeline_stage("pair_tail_compute_k0"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0_lo, bs0_lo, "e4m3", acc0)
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0_hi, bs0_hi, "e4m3", acc1)
    with gl.amd.warp_pipeline_stage("pair_tail_load_k1"):
        a1 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        as1_load = mxfp8_load_scale(
            as_buf, scale_slot, scale_start + 4,
            SCALE_A_LOAD, BLOCK_M, SCALE_TILE_WIDTH, 4)
        as1 = gl.convert_layout(
            as1_load, SCALE_A_LAYOUT, assert_trivial=True)
        b1 = b_buf.index(data_slot).slice(0, BLOCK_N, 0).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        bs1 = mxfp8_load_scale(
            bs_buf, scale_slot, scale_start + 4,
            SCALE_B_LOAD, BLOCK_N, SCALE_TILE_WIDTH, 4)
        b1_grid = b1.reshape((128, CTA_N, cta_n))
        b1_lo = gl.convert_layout(gl.amd.slice(
            b1_grid, [128, CTA_N, half_n], [0, 0, 0]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        b1_hi = gl.convert_layout(gl.amd.slice(
            b1_grid, [128, CTA_N, half_n], [0, 0, half_n]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        bs1_grid = bs1.reshape((CTA_N, cta_n, 4))
        bs1_lo = gl.convert_layout(gl.amd.slice(
            bs1_grid, [CTA_N, half_n, 4], [0, 0, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        bs1_hi = gl.convert_layout(gl.amd.slice(
            bs1_grid, [CTA_N, half_n, 4], [0, half_n, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
    with gl.amd.warp_pipeline_stage("pair_tail_compute_k1"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1_lo, bs1_lo, "e4m3", acc0)
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1_hi, bs1_hi, "e4m3", acc1)
    return acc0, acc1


@gluon.jit
def mxfp8_bk256_opt_consume_tail_split_n(
        a_buf, b_buf, as_buf, bs_buf, slot, acc0, acc1,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        DOT_B_LOAD: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
        SCALE_A_LOAD: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr, SCALE_B_LOAD: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_N: gl.constexpr, B_SCALE_CONTIGUOUS: gl.constexpr,
        A_SCALE_COMBINED: gl.constexpr):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    with gl.amd.warp_pipeline_stage("gap2_tail_load_k0"):
        a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        if A_SCALE_COMBINED:
            as_full = mxfp8_load_a_scale_b128(
                as_buf, slot, SCALE_A_LOAD, BLOCK_M)
            as0, as1 = mxfp8_split_a_scale(
                as_full, SCALE_A_LAYOUT, BLOCK_M)
        else:
            as0 = mxfp8_load_scale(
                as_buf, slot, 0, SCALE_A_LAYOUT, BLOCK_M, 8, 4)
        b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        bs0 = mxfp8_load_b_scale(
            bs_buf, slot, 0, SCALE_B_LOAD, BLOCK_N, B_SCALE_CONTIGUOUS)
        b0_grid = b0.reshape((128, CTA_N, cta_n))
        b0_lo = gl.convert_layout(gl.amd.slice(
            b0_grid, [128, CTA_N, half_n], [0, 0, 0]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        b0_hi = gl.convert_layout(gl.amd.slice(
            b0_grid, [128, CTA_N, half_n], [0, 0, half_n]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        bs0_grid = bs0.reshape((CTA_N, cta_n, 4))
        bs0_lo = gl.convert_layout(gl.amd.slice(
            bs0_grid, [CTA_N, half_n, 4], [0, 0, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        bs0_hi = gl.convert_layout(gl.amd.slice(
            bs0_grid, [CTA_N, half_n, 4], [0, half_n, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
    with gl.amd.warp_pipeline_stage("gap2_tail_compute_k0"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0_lo, bs0_lo, "e4m3", acc0)
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0_hi, bs0_hi, "e4m3", acc1)
    with gl.amd.warp_pipeline_stage("gap2_tail_load_k1"):
        a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        if not A_SCALE_COMBINED:
            as1 = mxfp8_load_scale(
                as_buf, slot, 4, SCALE_A_LAYOUT, BLOCK_M, 8, 4)
        b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        bs1 = mxfp8_load_b_scale(
            bs_buf, slot, 4, SCALE_B_LOAD, BLOCK_N, B_SCALE_CONTIGUOUS)
        b1_grid = b1.reshape((128, CTA_N, cta_n))
        b1_lo = gl.convert_layout(gl.amd.slice(
            b1_grid, [128, CTA_N, half_n], [0, 0, 0]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        b1_hi = gl.convert_layout(gl.amd.slice(
            b1_grid, [128, CTA_N, half_n], [0, 0, half_n]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        bs1_grid = bs1.reshape((CTA_N, cta_n, 4))
        bs1_lo = gl.convert_layout(gl.amd.slice(
            bs1_grid, [CTA_N, half_n, 4], [0, 0, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        bs1_hi = gl.convert_layout(gl.amd.slice(
            bs1_grid, [CTA_N, half_n, 4], [0, half_n, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
    with gl.amd.warp_pipeline_stage("gap2_tail_compute_k1"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1_lo, bs1_lo, "e4m3", acc0)
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1_hi, bs1_hi, "e4m3", acc1)
    return acc0, acc1


@gluon.jit
def mxfp8_split_a_m(
        a, DOT_A: gl.constexpr, BLOCK_M: gl.constexpr,
        CTA_M: gl.constexpr):
    cta_m: gl.constexpr = 256
    half_m: gl.constexpr = 128
    packed_m: gl.constexpr = BLOCK_M // 2
    a_grid = a.reshape((CTA_M, cta_m, 128))
    a_lo = gl.convert_layout(gl.amd.slice(
        a_grid, [CTA_M, half_m, 128], [0, 0, 0]
    ).reshape((packed_m, 128)), DOT_A, assert_trivial=True)
    a_hi = gl.convert_layout(gl.amd.slice(
        a_grid, [CTA_M, half_m, 128], [0, half_m, 0]
    ).reshape((packed_m, 128)), DOT_A, assert_trivial=True)
    return a_lo, a_hi


@gluon.jit
def mxfp8_split_a_scale_m(
        scale, SCALE_A_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
        CTA_M: gl.constexpr):
    cta_m: gl.constexpr = 256
    half_m: gl.constexpr = 128
    packed_m: gl.constexpr = BLOCK_M // 2
    scale_grid = scale.reshape((CTA_M, cta_m, 4))
    scale_lo = gl.convert_layout(gl.amd.slice(
        scale_grid, [CTA_M, half_m, 4], [0, 0, 0]
    ).reshape((packed_m, 4)), SCALE_A_LAYOUT, assert_trivial=True)
    scale_hi = gl.convert_layout(gl.amd.slice(
        scale_grid, [CTA_M, half_m, 4], [0, half_m, 0]
    ).reshape((packed_m, 4)), SCALE_A_LAYOUT, assert_trivial=True)
    return scale_lo, scale_hi


@gluon.jit
def mxfp8_bk256_consume_and_refill_split_m(
        a_buf, b_buf, as_buf, bs_buf, slot, a_desc, b_desc, as_desc, bs_desc,
        acc0, acc1, DOT_A: gl.constexpr, DOT_A_LOAD: gl.constexpr,
        DOT_B: gl.constexpr, DOT_B_LOAD: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_A_LOAD: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr, SCALE_B_LOAD: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_M: gl.constexpr, WAIT_REFILL_FIRST: gl.constexpr):
    tdm.async_wait(3)
    with gl.amd.warp_pipeline_stage(
            "msplit_stage0_load_k0", phase_gap=2):
        a0_full = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            0, 128, 1).load(layout=DOT_A_LOAD)
        a0_lo, a0_hi = mxfp8_split_a_m(
            a0_full, DOT_A, BLOCK_M, CTA_M)
        as0_full = mxfp8_load_scale(
            as_buf, slot, 0, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as0_lo, as0_hi = mxfp8_split_a_scale_m(
            as0_full, SCALE_A_LAYOUT, BLOCK_M, CTA_M)
        bs0_load = mxfp8_load_scale(
            bs_buf, slot, 0, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs0 = gl.convert_layout(
            bs0_load, SCALE_B_LAYOUT, assert_trivial=True)
    with gl.amd.warp_pipeline_stage("msplit_stage1_load_b_k0"):
        b0_load = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        b0 = gl.convert_layout(b0_load, DOT_B, assert_trivial=True)
    with gl.amd.warp_pipeline_stage(
            "msplit_stage2_compute_k0_m0"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a0_lo, as0_lo, "e4m3", b0, bs0, "e4m3", acc0)
    with gl.amd.warp_pipeline_stage(
            "msplit_stage3_compute_k0_m1"):
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a0_hi, as0_hi, "e4m3", b0, bs0, "e4m3", acc1)
    with gl.amd.warp_pipeline_stage(
            "msplit_stage4_load_k1"):
        a1_full = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            128, 128, 1).load(layout=DOT_A_LOAD)
        a1_lo, a1_hi = mxfp8_split_a_m(
            a1_full, DOT_A, BLOCK_M, CTA_M)
        as1_full = mxfp8_load_scale(
            as_buf, slot, 4, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as1_lo, as1_hi = mxfp8_split_a_scale_m(
            as1_full, SCALE_A_LAYOUT, BLOCK_M, CTA_M)
        b1_load = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        b1 = gl.convert_layout(b1_load, DOT_B, assert_trivial=True)
        bs1_load = mxfp8_load_scale(
            bs_buf, slot, 4, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs1 = gl.convert_layout(
            bs1_load, SCALE_B_LAYOUT, assert_trivial=True)
    with gl.amd.warp_pipeline_stage("msplit_stage5_arrive"):
        gl.amd.gfx1250.cluster.arrive()
    with gl.amd.warp_pipeline_stage(
            "msplit_stage6_compute_k1_m0"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a1_lo, as1_lo, "e4m3", b1, bs1, "e4m3", acc0)
        if WAIT_REFILL_FIRST:
            gl.amd.gfx1250.cluster.wait()
    with gl.amd.warp_pipeline_stage(
            "msplit_stage7_compute_k1_m1_refill"):
        if WAIT_REFILL_FIRST:
            mxfp8_issue_leading4_without_update(
                a_desc, b_desc, as_desc, bs_desc,
                a_buf, b_buf, as_buf, bs_buf, slot)
            acc1 = gl.amd.gfx1250.wmma_scaled(
                a1_hi, as1_hi, "e4m3", b1, bs1, "e4m3", acc1)
            a_desc, b_desc, as_desc, bs_desc = mxfp8_advance_descriptors(
                a_desc, b_desc, as_desc, bs_desc, 256, 1024)
        else:
            acc1 = gl.amd.gfx1250.wmma_scaled(
                a1_hi, as1_hi, "e4m3", b1, bs1, "e4m3", acc1)
            gl.amd.gfx1250.cluster.wait()
            a_desc, b_desc = mxfp8_issue_leading4_data(
                a_desc, b_desc, a_buf, b_buf, slot)
            as_desc, bs_desc = mxfp8_issue_leading4_scale(
                as_desc, bs_desc, as_buf, bs_buf, slot)
    return a_desc, b_desc, as_desc, bs_desc, acc0, acc1


@gluon.jit
def mxfp8_bk256_consume_tail_split_m(
        a_buf, b_buf, as_buf, bs_buf, slot, acc0, acc1,
        DOT_A: gl.constexpr, DOT_A_LOAD: gl.constexpr,
        DOT_B: gl.constexpr, DOT_B_LOAD: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_A_LOAD: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr, SCALE_B_LOAD: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_M: gl.constexpr):
    with gl.amd.warp_pipeline_stage("msplit_tail_load_k0"):
        a0_full = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            0, 128, 1).load(layout=DOT_A_LOAD)
        a0_lo, a0_hi = mxfp8_split_a_m(
            a0_full, DOT_A, BLOCK_M, CTA_M)
        as0_full = mxfp8_load_scale(
            as_buf, slot, 0, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as0_lo, as0_hi = mxfp8_split_a_scale_m(
            as0_full, SCALE_A_LAYOUT, BLOCK_M, CTA_M)
        b0_load = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        b0 = gl.convert_layout(b0_load, DOT_B, assert_trivial=True)
        bs0_load = mxfp8_load_scale(
            bs_buf, slot, 0, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs0 = gl.convert_layout(
            bs0_load, SCALE_B_LAYOUT, assert_trivial=True)
    with gl.amd.warp_pipeline_stage("msplit_tail_compute_k0"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a0_lo, as0_lo, "e4m3", b0, bs0, "e4m3", acc0)
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a0_hi, as0_hi, "e4m3", b0, bs0, "e4m3", acc1)
    with gl.amd.warp_pipeline_stage("msplit_tail_load_k1"):
        a1_full = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            128, 128, 1).load(layout=DOT_A_LOAD)
        a1_lo, a1_hi = mxfp8_split_a_m(
            a1_full, DOT_A, BLOCK_M, CTA_M)
        as1_full = mxfp8_load_scale(
            as_buf, slot, 4, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as1_lo, as1_hi = mxfp8_split_a_scale_m(
            as1_full, SCALE_A_LAYOUT, BLOCK_M, CTA_M)
        b1_load = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        b1 = gl.convert_layout(b1_load, DOT_B, assert_trivial=True)
        bs1_load = mxfp8_load_scale(
            bs_buf, slot, 4, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs1 = gl.convert_layout(
            bs1_load, SCALE_B_LAYOUT, assert_trivial=True)
    with gl.amd.warp_pipeline_stage("msplit_tail_compute_k1"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a1_lo, as1_lo, "e4m3", b1, bs1, "e4m3", acc0)
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a1_hi, as1_hi, "e4m3", b1, bs1, "e4m3", acc1)
    return acc0, acc1




@gluon.jit
def fp8_scaled_cluster_bk256_opt_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale_a, stride_scale_b, GRID_MN: gl.constexpr,
        SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
        LOAD_WMMA_LAYOUT: gl.constexpr, OUTPUT_CGA_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_M: gl.constexpr, CTA_N: gl.constexpr,
        PID_REUSE_A: gl.constexpr, PID_GROUP_SIZE: gl.constexpr,
        STAGE8_REFILL: gl.constexpr, TWO_TDM: gl.constexpr,
        B_SCALE_CONTIGUOUS: gl.constexpr,
        A_SCALE_COMBINED: gl.constexpr,
        DESC_UPDATE_STAGE5: gl.constexpr,
        SCALE_LOAD_STAGE1: gl.constexpr,
        B_LOAD_STAGE1: gl.constexpr,
        STAGE6_WAIT_REFILL_FIRST: gl.constexpr,
        EARLY_CLUSTER_SYNC: gl.constexpr,
        STAGE7_CLUSTER_SYNC: gl.constexpr,
        UNROLL_TWO_CONST_SLOTS: gl.constexpr,
        UNROLL_FOUR_CONST_SLOTS: gl.constexpr,
        SEPARATE_SLOT_DESCRIPTORS: gl.constexpr,
        STAGE5_ADVANCE_WAIT_REFILL_FIRST: gl.constexpr,
        ARRIVE_STAGE1_WAIT_STAGE7: gl.constexpr,
        PAIRED_SCALE_TDM: gl.constexpr,
        TRIPLE_SCALE_RING: gl.constexpr,
        FOUR_SCALE_RING: gl.constexpr,
        TRIPLE_REFILL_DATA_BEFORE_WMMA: gl.constexpr,
        TRIPLE_STAGE5_ADVANCE_REFILL_ALL_BEFORE_WMMA: gl.constexpr,
        TRIPLE_STAGE5_ADVANCE_ORIGINAL_STAGE7_ORDER: gl.constexpr,
        PIPELINED_OUTPUT_STORE: gl.constexpr):
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    gl.static_assert(BLOCK_M // CTA_M == 256)
    gl.static_assert(BLOCK_N // CTA_N == 256)
    gl.static_assert(
        not SEPARATE_SLOT_DESCRIPTORS or UNROLL_TWO_CONST_SLOTS)
    gl.static_assert(
        not (UNROLL_TWO_CONST_SLOTS and UNROLL_FOUR_CONST_SLOTS))
    gl.static_assert(
        not ARRIVE_STAGE1_WAIT_STAGE7 or UNROLL_TWO_CONST_SLOTS)
    gl.static_assert(
        not PAIRED_SCALE_TDM
        or (UNROLL_TWO_CONST_SLOTS and not TWO_TDM
            and not SEPARATE_SLOT_DESCRIPTORS
            and not STAGE5_ADVANCE_WAIT_REFILL_FIRST
            and not STAGE6_WAIT_REFILL_FIRST and not EARLY_CLUSTER_SYNC
            and not STAGE7_CLUSTER_SYNC and not ARRIVE_STAGE1_WAIT_STAGE7))
    gl.static_assert(
        not TRIPLE_SCALE_RING
        or (UNROLL_TWO_CONST_SLOTS and not PAIRED_SCALE_TDM
            and not TWO_TDM and not SEPARATE_SLOT_DESCRIPTORS
            and not STAGE5_ADVANCE_WAIT_REFILL_FIRST
            and not STAGE6_WAIT_REFILL_FIRST and not EARLY_CLUSTER_SYNC
            and not STAGE7_CLUSTER_SYNC and not ARRIVE_STAGE1_WAIT_STAGE7))
    gl.static_assert(
        not FOUR_SCALE_RING
        or (UNROLL_TWO_CONST_SLOTS and not PAIRED_SCALE_TDM
            and not TRIPLE_SCALE_RING and not TWO_TDM
            and not SEPARATE_SLOT_DESCRIPTORS
            and not STAGE5_ADVANCE_WAIT_REFILL_FIRST
            and not STAGE6_WAIT_REFILL_FIRST and not EARLY_CLUSTER_SYNC
            and not STAGE7_CLUSTER_SYNC and not ARRIVE_STAGE1_WAIT_STAGE7))
    gl.static_assert(
        not TRIPLE_REFILL_DATA_BEFORE_WMMA or TRIPLE_SCALE_RING)
    gl.static_assert(
        not TRIPLE_STAGE5_ADVANCE_REFILL_ALL_BEFORE_WMMA
        or (TRIPLE_SCALE_RING and not TRIPLE_REFILL_DATA_BEFORE_WMMA))
    gl.static_assert(
        not TRIPLE_STAGE5_ADVANCE_ORIGINAL_STAGE7_ORDER
        or (TRIPLE_SCALE_RING
            and not TRIPLE_REFILL_DATA_BEFORE_WMMA
            and not TRIPLE_STAGE5_ADVANCE_REFILL_ALL_BEFORE_WMMA))
    if PID_REUSE_A:
        pid_m, pid_n = snapshot_get_xcd_swizzled_pids_reuse_a(
            M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, PID_GROUP_SIZE)
    else:
        pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
            M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, PID_GROUP_SIZE)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    dot_b_load: gl.constexpr = gl.DotOperandLayout(
        1, LOAD_WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [BLOCK_M, 4])
    scale_a_load: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [BLOCK_M, 8])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [BLOCK_N // 2, 4])
    scale_b_load: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b_load, [BLOCK_N, 4])
    slots: gl.constexpr = MXFP8_NUM_SLOTS
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [slots, BLOCK_M, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [slots, BLOCK_N, 256], SHARED_LAYOUT_B)
    if PAIRED_SCALE_TDM:
        as_buf = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty,
            [2, BLOCK_M // 128, 2048], SHARED_SCALE_A)
        bs_buf = gl.allocate_shared_memory(
            b_scale_ptr.type.element_ty,
            [2, BLOCK_N // 128, 2048], SHARED_SCALE_B)
    elif TRIPLE_SCALE_RING or FOUR_SCALE_RING:
        scale_slots: gl.constexpr = 4 if FOUR_SCALE_RING else 3
        as_buf = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty,
            [scale_slots, BLOCK_M // 128, 1024], SHARED_SCALE_A)
        bs_buf = gl.allocate_shared_memory(
            b_scale_ptr.type.element_ty,
            [scale_slots, BLOCK_N // 128, 1024], SHARED_SCALE_B)
    else:
        as_buf = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty,
            [slots, BLOCK_M // 128, 1024], SHARED_SCALE_A)
    if (B_SCALE_CONTIGUOUS
            and not PAIRED_SCALE_TDM
            and not TRIPLE_SCALE_RING and not FOUR_SCALE_RING):
        bs_buf = gl.allocate_shared_memory(
            b_scale_ptr.type.element_ty, [slots, BLOCK_N, 8], SHARED_SCALE_B)
    elif (not PAIRED_SCALE_TDM
            and not TRIPLE_SCALE_RING and not FOUR_SCALE_RING):
        bs_buf = gl.allocate_shared_memory(
            b_scale_ptr.type.element_ty,
            [slots, BLOCK_N // 128, 1024], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, 256),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, 256),
        layout=SHARED_LAYOUT_B)
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + (pid_m * BLOCK_M) // 128 * stride_scale_a,
        shape=(M // 128, K // 32 * 128), strides=(stride_scale_a, 1),
        block_shape=(
            (BLOCK_M // 128, 2048) if PAIRED_SCALE_TDM
            else (BLOCK_M // 128, 1024)),
        layout=SHARED_SCALE_A)
    if B_SCALE_CONTIGUOUS:
        bs_desc = tdm.make_tensor_descriptor(
            base=b_scale_ptr + pid_n * BLOCK_N * stride_scale_b,
            shape=(N, K // 32), strides=(stride_scale_b, 1),
            block_shape=(BLOCK_N, 8), layout=SHARED_SCALE_B)
    else:
        bs_desc = tdm.make_tensor_descriptor(
            base=b_scale_ptr + (pid_n * BLOCK_N) // 128 * stride_scale_b,
            shape=(N // 128, K // 32 * 128), strides=(stride_scale_b, 1),
            block_shape=(
                (BLOCK_N // 128, 2048) if PAIRED_SCALE_TDM
                else (BLOCK_N // 128, 1024)),
            layout=SHARED_SCALE_B)
    if TRIPLE_SCALE_RING or FOUR_SCALE_RING:
        as_desc, bs_desc = mxfp8_issue_leading4_scale(
            as_desc, bs_desc, as_buf, bs_buf, 0)
        a_desc, b_desc = mxfp8_issue_leading4_data(
            a_desc, b_desc, a_buf, b_buf, 0)
        as_desc, bs_desc = mxfp8_issue_leading4_scale(
            as_desc, bs_desc, as_buf, bs_buf, 1)
        a_desc, b_desc = mxfp8_issue_leading4_data(
            a_desc, b_desc, a_buf, b_buf, 1)
        as_desc, bs_desc = mxfp8_issue_leading4_scale(
            as_desc, bs_desc, as_buf, bs_buf, 2)
    elif PAIRED_SCALE_TDM:
        a_desc, b_desc = mxfp8_issue_leading4_data(
            a_desc, b_desc, a_buf, b_buf, 0)
        as_desc, bs_desc = mxfp8_issue_pair_scale(
            as_desc, bs_desc, as_buf, bs_buf, 0)
        a_desc, b_desc = mxfp8_issue_leading4_data(
            a_desc, b_desc, a_buf, b_buf, 1)
    elif SEPARATE_SLOT_DESCRIPTORS:
        a_desc0 = a_desc
        b_desc0 = b_desc
        as_desc0 = as_desc
        bs_desc0 = bs_desc
        a_desc1, b_desc1, as_desc1, bs_desc1 = mxfp8_advance_descriptors(
            a_desc, b_desc, as_desc, bs_desc, 256, 1024)
        mxfp8_issue_leading4_without_update(
            a_desc0, b_desc0, as_desc0, bs_desc0,
            a_buf, b_buf, as_buf, bs_buf, 0)
        mxfp8_issue_leading4_without_update(
            a_desc1, b_desc1, as_desc1, bs_desc1,
            a_buf, b_buf, as_buf, bs_buf, 1)
        a_desc0, b_desc0, as_desc0, bs_desc0 = mxfp8_advance_descriptors(
            a_desc0, b_desc0, as_desc0, bs_desc0, 512, 2048)
        a_desc1, b_desc1, as_desc1, bs_desc1 = mxfp8_advance_descriptors(
            a_desc1, b_desc1, as_desc1, bs_desc1, 512, 2048)
    else:
        for prefetch_idx in gl.static_range(2):
            if TWO_TDM:
                a_desc, b_desc, as_desc, bs_desc = mxfp8_issue_two_tdm(
                    a_desc, b_desc, as_desc, bs_desc,
                    a_buf, b_buf, as_buf, bs_buf, prefetch_idx,
                    B_SCALE_CONTIGUOUS)
            else:
                a_desc, b_desc = mxfp8_issue_leading4_data(
                    a_desc, b_desc, a_buf, b_buf, prefetch_idx)
                as_desc, bs_desc = mxfp8_issue_leading4_scale(
                    as_desc, bs_desc, as_buf, bs_buf, prefetch_idx)
    acc0 = gl.zeros(
        (BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    acc1 = gl.zeros(
        (BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 256)
    gl.assume(iter_max >= 3)
    if UNROLL_FOUR_CONST_SLOTS:
        gl.assume(iter_max >= 4)
        gl.assume(iter_max % 4 == 0)
        for _ in range(0, (iter_max - 4) // 4):
            for inner in gl.static_range(4):
                a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
                    mxfp8_bk256_opt_consume_and_refill(
                        a_buf, b_buf, as_buf, bs_buf, inner % 2, inner % 2,
                        a_desc, b_desc, as_desc, bs_desc, acc0, acc1,
                        dot_a, dot_b, dot_b_load, scale_a_layout, scale_a_load,
                        scale_b_layout, scale_b_load,
                        BLOCK_M, BLOCK_N, CTA_N, STAGE8_REFILL, TWO_TDM,
                        B_SCALE_CONTIGUOUS, A_SCALE_COMBINED,
                        DESC_UPDATE_STAGE5, SCALE_LOAD_STAGE1, B_LOAD_STAGE1,
                        STAGE6_WAIT_REFILL_FIRST, EARLY_CLUSTER_SYNC,
                        STAGE7_CLUSTER_SYNC, False,
                        STAGE5_ADVANCE_WAIT_REFILL_FIRST,
                        ARRIVE_STAGE1_WAIT_STAGE7))
        for inner in gl.static_range(2):
            a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
                mxfp8_bk256_opt_consume_and_refill(
                    a_buf, b_buf, as_buf, bs_buf, inner, inner,
                    a_desc, b_desc, as_desc, bs_desc, acc0, acc1,
                    dot_a, dot_b, dot_b_load, scale_a_layout, scale_a_load,
                    scale_b_layout, scale_b_load,
                    BLOCK_M, BLOCK_N, CTA_N, STAGE8_REFILL, TWO_TDM,
                    B_SCALE_CONTIGUOUS, A_SCALE_COMBINED,
                    DESC_UPDATE_STAGE5, SCALE_LOAD_STAGE1, B_LOAD_STAGE1,
                    STAGE6_WAIT_REFILL_FIRST, EARLY_CLUSTER_SYNC,
                    STAGE7_CLUSTER_SYNC, False,
                    STAGE5_ADVANCE_WAIT_REFILL_FIRST,
                    ARRIVE_STAGE1_WAIT_STAGE7))
    elif UNROLL_TWO_CONST_SLOTS:
        gl.assume(iter_max >= 4)
        gl.assume(iter_max % 2 == 0)
        if TRIPLE_SCALE_RING or FOUR_SCALE_RING:
            scale_ring: gl.constexpr = 4 if FOUR_SCALE_RING else 3
            for pair_idx in range(0, (iter_max - 4) // 2):
                tile0 = pair_idx * 2
                scale_slot0 = tile0 % scale_ring
                scale_slot1 = (tile0 + 1) % scale_ring
                refill_scale_slot0 = (tile0 + 3) % scale_ring
                refill_scale_slot1 = (tile0 + 4) % scale_ring
                a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
                    mxfp8_bk256_consume_and_refill_paired_scales(
                        a_buf, b_buf, as_buf, bs_buf, 0, scale_slot0,
                        refill_scale_slot0, 0,
                        a_desc, b_desc, as_desc, bs_desc,
                        acc0, acc1, dot_a, dot_b, dot_b_load,
                        scale_a_layout, scale_a_load,
                        scale_b_layout, scale_b_load,
                        BLOCK_M, BLOCK_N, CTA_N, 4, True, 8,
                        TRIPLE_REFILL_DATA_BEFORE_WMMA,
                        TRIPLE_STAGE5_ADVANCE_REFILL_ALL_BEFORE_WMMA,
                        TRIPLE_STAGE5_ADVANCE_ORIGINAL_STAGE7_ORDER))
                a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
                    mxfp8_bk256_consume_and_refill_paired_scales(
                        a_buf, b_buf, as_buf, bs_buf, 1, scale_slot1,
                        refill_scale_slot1, 0,
                        a_desc, b_desc, as_desc, bs_desc,
                        acc0, acc1, dot_a, dot_b, dot_b_load,
                        scale_a_layout, scale_a_load,
                        scale_b_layout, scale_b_load,
                        BLOCK_M, BLOCK_N, CTA_N, 4, True, 8,
                        TRIPLE_REFILL_DATA_BEFORE_WMMA,
                        TRIPLE_STAGE5_ADVANCE_REFILL_ALL_BEFORE_WMMA,
                        TRIPLE_STAGE5_ADVANCE_ORIGINAL_STAGE7_ORDER))
            residual_tile = iter_max - 4
            residual_scale0 = residual_tile % scale_ring
            residual_scale1 = (residual_tile + 1) % scale_ring
            residual_refill_scale0 = (residual_tile + 3) % scale_ring
            a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
                mxfp8_bk256_consume_and_refill_paired_scales(
                    a_buf, b_buf, as_buf, bs_buf, 0, residual_scale0,
                    residual_refill_scale0, 0,
                    a_desc, b_desc, as_desc, bs_desc,
                    acc0, acc1, dot_a, dot_b, dot_b_load,
                    scale_a_layout, scale_a_load,
                    scale_b_layout, scale_b_load,
                    BLOCK_M, BLOCK_N, CTA_N, 4, True, 8,
                    TRIPLE_REFILL_DATA_BEFORE_WMMA,
                    TRIPLE_STAGE5_ADVANCE_REFILL_ALL_BEFORE_WMMA,
                    TRIPLE_STAGE5_ADVANCE_ORIGINAL_STAGE7_ORDER))
            a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
                mxfp8_bk256_consume_and_refill_paired_scales(
                    a_buf, b_buf, as_buf, bs_buf, 1, residual_scale1,
                    residual_scale1, 0, a_desc, b_desc, as_desc, bs_desc,
                    acc0, acc1, dot_a, dot_b, dot_b_load,
                    scale_a_layout, scale_a_load,
                    scale_b_layout, scale_b_load,
                    BLOCK_M, BLOCK_N, CTA_N, 4, False, 8,
                    TRIPLE_REFILL_DATA_BEFORE_WMMA,
                    TRIPLE_STAGE5_ADVANCE_REFILL_ALL_BEFORE_WMMA,
                    TRIPLE_STAGE5_ADVANCE_ORIGINAL_STAGE7_ORDER))
        elif PAIRED_SCALE_TDM:
            for pair_idx in range(0, (iter_max - 2) // 2):
                scale_slot = pair_idx % 2
                next_scale_slot = (pair_idx + 1) % 2
                a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
                    mxfp8_bk256_consume_and_refill_paired_scales(
                        a_buf, b_buf, as_buf, bs_buf, 0, scale_slot,
                        next_scale_slot, 0, a_desc, b_desc, as_desc, bs_desc,
                        acc0, acc1, dot_a, dot_b, dot_b_load,
                        scale_a_layout, scale_a_load,
                        scale_b_layout, scale_b_load,
                        BLOCK_M, BLOCK_N, CTA_N, 2, True, 16,
                        TRIPLE_REFILL_DATA_BEFORE_WMMA,
                        TRIPLE_STAGE5_ADVANCE_REFILL_ALL_BEFORE_WMMA,
                        TRIPLE_STAGE5_ADVANCE_ORIGINAL_STAGE7_ORDER))
                a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
                    mxfp8_bk256_consume_and_refill_paired_scales(
                        a_buf, b_buf, as_buf, bs_buf, 1, scale_slot,
                        next_scale_slot, 8, a_desc, b_desc, as_desc, bs_desc,
                        acc0, acc1, dot_a, dot_b, dot_b_load,
                        scale_a_layout, scale_a_load,
                        scale_b_layout, scale_b_load,
                        BLOCK_M, BLOCK_N, CTA_N, 3, False, 16,
                        TRIPLE_REFILL_DATA_BEFORE_WMMA,
                        TRIPLE_STAGE5_ADVANCE_REFILL_ALL_BEFORE_WMMA,
                        TRIPLE_STAGE5_ADVANCE_ORIGINAL_STAGE7_ORDER))
        elif SEPARATE_SLOT_DESCRIPTORS:
            for _ in range(0, (iter_max - 2) // 2):
                a_desc0, b_desc0, as_desc0, bs_desc0, acc0, acc1 = (
                    mxfp8_bk256_opt_consume_and_refill(
                        a_buf, b_buf, as_buf, bs_buf, 0, 0,
                        a_desc0, b_desc0, as_desc0, bs_desc0, acc0, acc1,
                        dot_a, dot_b, dot_b_load, scale_a_layout, scale_a_load,
                        scale_b_layout, scale_b_load,
                        BLOCK_M, BLOCK_N, CTA_N, STAGE8_REFILL, TWO_TDM,
                        B_SCALE_CONTIGUOUS, A_SCALE_COMBINED,
                        DESC_UPDATE_STAGE5, SCALE_LOAD_STAGE1, B_LOAD_STAGE1,
                        STAGE6_WAIT_REFILL_FIRST, EARLY_CLUSTER_SYNC,
                        STAGE7_CLUSTER_SYNC, True,
                        STAGE5_ADVANCE_WAIT_REFILL_FIRST,
                        ARRIVE_STAGE1_WAIT_STAGE7))
                a_desc1, b_desc1, as_desc1, bs_desc1, acc0, acc1 = (
                    mxfp8_bk256_opt_consume_and_refill(
                        a_buf, b_buf, as_buf, bs_buf, 1, 1,
                        a_desc1, b_desc1, as_desc1, bs_desc1, acc0, acc1,
                        dot_a, dot_b, dot_b_load, scale_a_layout, scale_a_load,
                        scale_b_layout, scale_b_load,
                        BLOCK_M, BLOCK_N, CTA_N, STAGE8_REFILL, TWO_TDM,
                        B_SCALE_CONTIGUOUS, A_SCALE_COMBINED,
                        DESC_UPDATE_STAGE5, SCALE_LOAD_STAGE1, B_LOAD_STAGE1,
                        STAGE6_WAIT_REFILL_FIRST, EARLY_CLUSTER_SYNC,
                        STAGE7_CLUSTER_SYNC, True,
                        STAGE5_ADVANCE_WAIT_REFILL_FIRST,
                        ARRIVE_STAGE1_WAIT_STAGE7))
        else:
            for _ in range(0, (iter_max - 2) // 2):
                a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
                    mxfp8_bk256_opt_consume_and_refill(
                        a_buf, b_buf, as_buf, bs_buf, 0, 0,
                        a_desc, b_desc, as_desc, bs_desc, acc0, acc1,
                        dot_a, dot_b, dot_b_load, scale_a_layout, scale_a_load,
                        scale_b_layout, scale_b_load,
                        BLOCK_M, BLOCK_N, CTA_N, STAGE8_REFILL, TWO_TDM,
                        B_SCALE_CONTIGUOUS, A_SCALE_COMBINED,
                        DESC_UPDATE_STAGE5, SCALE_LOAD_STAGE1, B_LOAD_STAGE1,
                        STAGE6_WAIT_REFILL_FIRST, EARLY_CLUSTER_SYNC,
                        STAGE7_CLUSTER_SYNC, False,
                        STAGE5_ADVANCE_WAIT_REFILL_FIRST,
                        ARRIVE_STAGE1_WAIT_STAGE7))
                a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
                    mxfp8_bk256_opt_consume_and_refill(
                        a_buf, b_buf, as_buf, bs_buf, 1, 1,
                        a_desc, b_desc, as_desc, bs_desc, acc0, acc1,
                        dot_a, dot_b, dot_b_load, scale_a_layout, scale_a_load,
                        scale_b_layout, scale_b_load,
                        BLOCK_M, BLOCK_N, CTA_N, STAGE8_REFILL, TWO_TDM,
                        B_SCALE_CONTIGUOUS, A_SCALE_COMBINED,
                        DESC_UPDATE_STAGE5, SCALE_LOAD_STAGE1, B_LOAD_STAGE1,
                        STAGE6_WAIT_REFILL_FIRST, EARLY_CLUSTER_SYNC,
                        STAGE7_CLUSTER_SYNC, False,
                        STAGE5_ADVANCE_WAIT_REFILL_FIRST,
                        ARRIVE_STAGE1_WAIT_STAGE7))
    else:
        for tile_idx in range(0, iter_max - 2):
            slot = tile_idx % 2
            a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
                mxfp8_bk256_opt_consume_and_refill(
                    a_buf, b_buf, as_buf, bs_buf, slot, slot,
                    a_desc, b_desc, as_desc, bs_desc, acc0, acc1,
                    dot_a, dot_b, dot_b_load, scale_a_layout, scale_a_load,
                    scale_b_layout, scale_b_load,
                    BLOCK_M, BLOCK_N, CTA_N, STAGE8_REFILL, TWO_TDM,
                    B_SCALE_CONTIGUOUS, A_SCALE_COMBINED,
                    DESC_UPDATE_STAGE5, SCALE_LOAD_STAGE1, B_LOAD_STAGE1,
                    STAGE6_WAIT_REFILL_FIRST, EARLY_CLUSTER_SYNC,
                    STAGE7_CLUSTER_SYNC, False,
                    STAGE5_ADVANCE_WAIT_REFILL_FIRST,
                    ARRIVE_STAGE1_WAIT_STAGE7))

    if TRIPLE_SCALE_RING or FOUR_SCALE_RING:
        tail_scale_ring: gl.constexpr = 4 if FOUR_SCALE_RING else 3
        tdm.async_wait(3)
        acc0, acc1 = mxfp8_bk256_consume_tail_paired_scales(
            a_buf, b_buf, as_buf, bs_buf, 0,
            (iter_max - 2) % tail_scale_ring, 0,
            acc0, acc1, dot_a, dot_b, dot_b_load,
            scale_a_layout, scale_a_load, scale_b_layout, scale_b_load,
            BLOCK_M, BLOCK_N, CTA_N, 8)
        tdm.async_wait(0)
        acc0, acc1 = mxfp8_bk256_consume_tail_paired_scales(
            a_buf, b_buf, as_buf, bs_buf, 1,
            (iter_max - 1) % tail_scale_ring, 0,
            acc0, acc1, dot_a, dot_b, dot_b_load,
            scale_a_layout, scale_a_load, scale_b_layout, scale_b_load,
            BLOCK_M, BLOCK_N, CTA_N, 8)
    elif PAIRED_SCALE_TDM:
        tail_scale_slot = ((iter_max - 2) // 2) % 2
        tdm.async_wait(2)
        acc0, acc1 = mxfp8_bk256_consume_tail_paired_scales(
            a_buf, b_buf, as_buf, bs_buf, 0, tail_scale_slot, 0,
            acc0, acc1, dot_a, dot_b, dot_b_load,
            scale_a_layout, scale_a_load, scale_b_layout, scale_b_load,
            BLOCK_M, BLOCK_N, CTA_N, 16)
        tdm.async_wait(0)
        acc0, acc1 = mxfp8_bk256_consume_tail_paired_scales(
            a_buf, b_buf, as_buf, bs_buf, 1, tail_scale_slot, 8,
            acc0, acc1, dot_a, dot_b, dot_b_load,
            scale_a_layout, scale_a_load, scale_b_layout, scale_b_load,
            BLOCK_M, BLOCK_N, CTA_N, 16)
    else:
        tdm.async_wait(2 if TWO_TDM else 3)
        penultimate_slot = (iter_max - 2) % 2
        acc0, acc1 = mxfp8_bk256_opt_consume_tail_split_n(
            a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc0, acc1,
            dot_a, dot_b, dot_b_load, scale_a_layout, scale_a_load,
            scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N,
            B_SCALE_CONTIGUOUS, A_SCALE_COMBINED)
        tdm.async_wait(0)
        last_slot = (iter_max - 1) % 2
        acc0, acc1 = mxfp8_bk256_opt_consume_tail_split_n(
            a_buf, b_buf, as_buf, bs_buf, last_slot, acc0, acc1, dot_a, dot_b,
            dot_b_load, scale_a_layout, scale_a_load, scale_b_layout,
            scale_b_load, BLOCK_M, BLOCK_N, CTA_N,
            B_SCALE_CONTIGUOUS, A_SCALE_COMBINED)
    snapshot_cluster_wait()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    if PIPELINED_OUTPUT_STORE:
        mxfp8_tdm_store_split_n2_pipelined(
            c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1,
            OUTPUT_CGA_LAYOUT, CTA_M, CTA_N)
    else:
        mxfp8_tdm_store_split_n2(
            c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1,
            OUTPUT_CGA_LAYOUT, CTA_M, CTA_N)


@gluon.jit
def fp8_scaled_cluster_bk256_msplit_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale_a, stride_scale_b, GRID_MN: gl.constexpr,
        SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr,
        SHARED_SCALE_A: gl.constexpr, SHARED_SCALE_B: gl.constexpr,
        WMMA_LAYOUT: gl.constexpr, LOAD_WMMA_LAYOUT: gl.constexpr,
        OUTPUT_CGA_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, CTA_M: gl.constexpr, CTA_N: gl.constexpr,
        WAIT_REFILL_FIRST: gl.constexpr):
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    gl.static_assert(BLOCK_M // CTA_M == 256)
    gl.static_assert(BLOCK_N // CTA_N == 256)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
        M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    dot_a_load: gl.constexpr = gl.DotOperandLayout(
        0, LOAD_WMMA_LAYOUT, 16)
    dot_b_load: gl.constexpr = gl.DotOperandLayout(
        1, LOAD_WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [BLOCK_M // 2, 4])
    scale_a_load: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a_load, [BLOCK_M, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [BLOCK_N, 4])
    scale_b_load: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b_load, [BLOCK_N, 4])
    slots: gl.constexpr = 2
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [slots, BLOCK_M, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [slots, BLOCK_N, 256], SHARED_LAYOUT_B)
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty,
        [slots, BLOCK_M // 128, 1024], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty,
        [slots, BLOCK_N // 128, 1024], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, 256),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, 256),
        layout=SHARED_LAYOUT_B)
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + (pid_m * BLOCK_M) // 128 * stride_scale_a,
        shape=(M // 128, K // 32 * 128), strides=(stride_scale_a, 1),
        block_shape=(BLOCK_M // 128, 1024), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * BLOCK_N) // 128 * stride_scale_b,
        shape=(N // 128, K // 32 * 128), strides=(stride_scale_b, 1),
        block_shape=(BLOCK_N // 128, 1024), layout=SHARED_SCALE_B)
    for prefetch_idx in gl.static_range(2):
        a_desc, b_desc = mxfp8_issue_leading4_data(
            a_desc, b_desc, a_buf, b_buf, prefetch_idx)
        as_desc, bs_desc = mxfp8_issue_leading4_scale(
            as_desc, bs_desc, as_buf, bs_buf, prefetch_idx)
    acc0 = gl.zeros(
        (BLOCK_M // 2, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    acc1 = gl.zeros(
        (BLOCK_M // 2, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 256)
    gl.assume(iter_max >= 4)
    gl.assume(iter_max % 2 == 0)
    for _ in range(0, (iter_max - 2) // 2):
        a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
            mxfp8_bk256_consume_and_refill_split_m(
                a_buf, b_buf, as_buf, bs_buf, 0,
                a_desc, b_desc, as_desc, bs_desc, acc0, acc1,
                dot_a, dot_a_load, dot_b, dot_b_load,
                scale_a_layout, scale_a_load, scale_b_layout, scale_b_load,
                BLOCK_M, BLOCK_N, CTA_M, WAIT_REFILL_FIRST))
        a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
            mxfp8_bk256_consume_and_refill_split_m(
                a_buf, b_buf, as_buf, bs_buf, 1,
                a_desc, b_desc, as_desc, bs_desc, acc0, acc1,
                dot_a, dot_a_load, dot_b, dot_b_load,
                scale_a_layout, scale_a_load, scale_b_layout, scale_b_load,
                BLOCK_M, BLOCK_N, CTA_M, WAIT_REFILL_FIRST))
    tdm.async_wait(3)
    acc0, acc1 = mxfp8_bk256_consume_tail_split_m(
        a_buf, b_buf, as_buf, bs_buf, 0, acc0, acc1,
        dot_a, dot_a_load, dot_b, dot_b_load,
        scale_a_layout, scale_a_load, scale_b_layout, scale_b_load,
        BLOCK_M, BLOCK_N, CTA_M)
    tdm.async_wait(0)
    acc0, acc1 = mxfp8_bk256_consume_tail_split_m(
        a_buf, b_buf, as_buf, bs_buf, 1, acc0, acc1,
        dot_a, dot_a_load, dot_b, dot_b_load,
        scale_a_layout, scale_a_load, scale_b_layout, scale_b_load,
        BLOCK_M, BLOCK_N, CTA_M)
    snapshot_cluster_wait()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    mxfp8_tdm_store_split_m2(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1,
        OUTPUT_CGA_LAYOUT, CTA_M, CTA_N)


def build_mxfp8_layouts(cluster_m=4, cluster_n=None):
    """Build cluster data, scale, and WMMA layouts for MXFP8.

    Data operands use the K128 scaled-WMMA instruction shape. Scale layouts
    reuse the corresponding A/B CGA bases, so data and its block-32 scales are
    owned by the same CTA partition. The 128-column N slice matches the serial
    epilogue and reduces accumulator register pressure enough to avoid spills.
    """
    cluster_n = cluster_m if cluster_n is None else cluster_n
    block_m = 256 * cluster_m
    block_n = 256 * cluster_n
    cga_c = make_cga_layout(
        [cluster_m, cluster_n], [cluster_m, cluster_n], [0, 1])
    output_cga = tuple(tuple(basis) + (0, 0) for basis in cga_c)
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_m, 256], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_n, 256], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, block_n, local_a, local_b, 8, [16, 16, 128],
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
        [[256, 16]], [block_m, 256], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_n, 256], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        256, 256, padded_a, padded_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=128)
    shared_as = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_m // 128, 1024], [1, 0], cga_a)
    shared_bs = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_n // 128, 1024], [1, 0], cga_b)
    return shared_a, shared_b, shared_as, shared_bs, wmma, output_cga


def build_mxfp8_bk256_opt_layouts(cluster_m=4, cluster_n=None):
    """Build BK256 operand layouts with a 128-column accumulator partition."""
    cluster_n = cluster_m if cluster_n is None else cluster_n
    block_m = 256 * cluster_m
    packed_n = 128 * cluster_n
    shared_a, shared_b, shared_as, shared_bs, load_wmma, output_cga = (
        build_mxfp8_layouts(cluster_m, cluster_n))
    cga_c = make_cga_layout(
        [cluster_m, cluster_n], [cluster_m, cluster_n], [0, 1])
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_m, 256], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [packed_n, 256], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, packed_n, local_a, local_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=128)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga_c)
    return (shared_a, shared_b, shared_as, shared_bs, wmma, load_wmma,
            output_cga)


def build_mxfp8_bk256_pair_scale_layouts(cluster_m=4, cluster_n=None):
    """Build unroll2 layouts with two BK256 scale tiles per pair buffer."""
    cluster_n = cluster_m if cluster_n is None else cluster_n
    block_m = 256 * cluster_m
    block_n = 256 * cluster_n
    shared_a, shared_b, shared_as, shared_bs, wmma, load_wmma, output_cga = (
        build_mxfp8_bk256_opt_layouts(cluster_m, cluster_n))
    shared_as_pair = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_m // 128, 2048], [1, 0],
        shared_as.cga_layout)
    shared_bs_pair = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_n // 128, 2048], [1, 0],
        shared_bs.cga_layout)
    return (shared_a, shared_b, shared_as_pair, shared_bs_pair,
            wmma, load_wmma, output_cga)


def build_mxfp8_bk256_msplit_layouts(cluster_m=4, cluster_n=None):
    """Build BK256 layouts with a local M128 accumulator partition."""
    cluster_n = cluster_m if cluster_n is None else cluster_n
    block_m = 256 * cluster_m
    block_n = 256 * cluster_n
    packed_m = 128 * cluster_m
    cga_c = make_cga_layout(
        [cluster_m, cluster_n], [cluster_m, cluster_n], [0, 1])
    output_cga = tuple(tuple(basis) + (0, 0) for basis in cga_c)

    local_a_load = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_m, 256], [1, 0])
    local_b_load = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_n, 256], [1, 0])
    _, _, local_load_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, block_n, local_a_load, local_b_load, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=128, slice_n=256)
    load_wmma = gl.amd.AMDWMMALayout(
        3, local_load_wmma.transposed, local_load_wmma.warp_bases,
        local_load_wmma.reg_bases, local_load_wmma.instr_shape, cga_c)
    load_dot_a = gl.DotOperandLayout(0, load_wmma, 16)
    load_dot_b = gl.DotOperandLayout(1, load_wmma, 16)
    cga_a = load_dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in load_dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_m, 256], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_n, 256], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        256, 256, padded_a, padded_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=128, slice_n=256)
    shared_as = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_m // 128, 1024], [1, 0], cga_a)
    shared_bs = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_n // 128, 1024], [1, 0], cga_b)

    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [packed_m, 256], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_n, 256], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        packed_m, block_n, local_a, local_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=128, slice_n=256)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga_c)
    return (shared_a, shared_b, shared_as, shared_bs, wmma, load_wmma,
            output_cga)


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
        a_desc, b_desc, a_buf, b_buf, slot, B_WARP_HINT: gl.constexpr):
    tdm.async_load_fused([
        (a_desc, a_buf.index(slot), 0b00000011),
        (b_desc, b_buf.index(slot), B_WARP_HINT),
    ])
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 128])
    return a_desc, b_desc


@gluon.jit
def fp8_mxfp4_issue_leading4_scale(
        bs_desc, bs_buf, slot):
    # Only B is quantized here, so this copy has no A-scale partner to be
    # disjoint from. Standalone copies stay separate by contract, so it takes
    # all four of the warps a TDM copy is allowed rather than the warps 2 and
    # 3 a fused pair would have left it.
    tdm.async_load(
        bs_desc, dest=bs_buf.index(slot), warp_used_hint=0b00001111)
    return tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, 1024])


@gluon.jit
def fp8_mxfp4_stage8_split_scale(
        bs, SCALE_B_LAYOUT: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_N: gl.constexpr):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    bs_grid = bs.reshape((CTA_N, cta_n, 4))
    bs_lo = gl.convert_layout(gl.amd.slice(
        bs_grid, [CTA_N, half_n, 4], [0, 0, 0]
    ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
    bs_hi = gl.convert_layout(gl.amd.slice(
        bs_grid, [CTA_N, half_n, 4], [0, half_n, 0]
    ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
    return bs_lo, bs_hi


@gluon.jit
def fp8_mxfp4_stage8_consume_and_refill(
        a_buf, b_buf, bs_buf, slot, a_desc, b_desc, bs_desc,
        acc0, acc1, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        DOT_B_LOAD: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        SCALE_B_LOAD: gl.constexpr, B_WARP_HINT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_N: gl.constexpr, WAITCNT: gl.constexpr,
        B_LOAD_STAGE1: gl.constexpr):
    tdm.async_wait(WAITCNT)
    with gl.amd.warp_pipeline_stage(
            "fp8mxfp4_stage0_load_k0", phase_gap=2):
        a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        bs0_full = fp8_mxfp4_load_scale(
            bs_buf, slot, 0, SCALE_B_LOAD, BLOCK_N)
        bs0, bs1 = fp8_mxfp4_stage8_split_scale(
            bs0_full, SCALE_B_LAYOUT, BLOCK_N, CTA_N)
        if not B_LOAD_STAGE1:
            b0_full = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
                0, 64, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
            b0, b1 = bf16_8stage_split_b(
                b0_full, DOT_B, BLOCK_N, CTA_N)
    with gl.amd.warp_pipeline_stage("fp8mxfp4_stage1_bubble"):
        if B_LOAD_STAGE1:
            b0_full = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
                0, 64, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
            b0, b1 = bf16_8stage_split_b(
                b0_full, DOT_B, BLOCK_N, CTA_N)
    with gl.amd.warp_pipeline_stage(
            "fp8mxfp4_stage2_compute_k0_n0"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a0, None, "e4m3", b0, bs0, "e2m1", acc0)
    with gl.amd.warp_pipeline_stage(
            "fp8mxfp4_stage3_compute_k0_n1"):
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a0, None, "e4m3", b1, bs1, "e2m1", acc1)
    with gl.amd.warp_pipeline_stage(
            "fp8mxfp4_stage4_load_k1"):
        a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        bs1_full = fp8_mxfp4_load_scale(
            bs_buf, slot, 4, SCALE_B_LOAD, BLOCK_N)
        bs2, bs3 = fp8_mxfp4_stage8_split_scale(
            bs1_full, SCALE_B_LAYOUT, BLOCK_N, CTA_N)
        b1_full = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            64, 64, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        b2, b3 = bf16_8stage_split_b(
            b1_full, DOT_B, BLOCK_N, CTA_N)
    with gl.amd.warp_pipeline_stage("fp8mxfp4_stage5_bubble"):
        pass
    with gl.amd.warp_pipeline_stage(
            "fp8mxfp4_stage6_compute_k1_n0"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a1, None, "e4m3", b2, bs2, "e2m1", acc0)
    with gl.amd.warp_pipeline_stage(
            "fp8mxfp4_stage7_arrive_compute_k1_n1"):
        gl.amd.gfx1250.cluster.arrive()
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a1, None, "e4m3", b3, bs3, "e2m1", acc1)
    with gl.amd.warp_pipeline_stage(
            "fp8mxfp4_stage8_wait_refill"):
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc = fp8_mxfp4_issue_leading4_data(
            a_desc, b_desc, a_buf, b_buf, slot, B_WARP_HINT)
        bs_desc = fp8_mxfp4_issue_leading4_scale(
            bs_desc, bs_buf, slot)
    return a_desc, b_desc, bs_desc, acc0, acc1


@gluon.jit
def fp8_mxfp4_stage8_consume_tail(
        a_buf, b_buf, bs_buf, slot, acc0, acc1,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        DOT_B_LOAD: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        SCALE_B_LOAD: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, CTA_N: gl.constexpr):
    with gl.amd.warp_pipeline_stage("fp8mxfp4_tail_load_k0"):
        a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        b0_full = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            0, 64, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        bs0_full = fp8_mxfp4_load_scale(
            bs_buf, slot, 0, SCALE_B_LOAD, BLOCK_N)
        b0, b1 = bf16_8stage_split_b(
            b0_full, DOT_B, BLOCK_N, CTA_N)
        bs0, bs1 = fp8_mxfp4_stage8_split_scale(
            bs0_full, SCALE_B_LAYOUT, BLOCK_N, CTA_N)
    with gl.amd.warp_pipeline_stage("fp8mxfp4_tail_compute_k0"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a0, None, "e4m3", b0, bs0, "e2m1", acc0)
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a0, None, "e4m3", b1, bs1, "e2m1", acc1)
    with gl.amd.warp_pipeline_stage("fp8mxfp4_tail_load_k1"):
        a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        b1_full = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            64, 64, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        bs1_full = fp8_mxfp4_load_scale(
            bs_buf, slot, 4, SCALE_B_LOAD, BLOCK_N)
        b2, b3 = bf16_8stage_split_b(
            b1_full, DOT_B, BLOCK_N, CTA_N)
        bs2, bs3 = fp8_mxfp4_stage8_split_scale(
            bs1_full, SCALE_B_LAYOUT, BLOCK_N, CTA_N)
    with gl.amd.warp_pipeline_stage("fp8mxfp4_tail_compute_k1"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a1, None, "e4m3", b2, bs2, "e2m1", acc0)
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a1, None, "e4m3", b3, bs3, "e2m1", acc1)
    return acc0, acc1


@gluon.jit
def fp8_mxfp4_stage8_gfx1250(
        a_ptr, b_ptr, c_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr,
        SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
        LOAD_WMMA_LAYOUT: gl.constexpr, OUTPUT_CGA_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_M: gl.constexpr, CTA_N: gl.constexpr,
        REUSE_A: gl.constexpr, GROUP_SIZE: gl.constexpr,
        WAITCNT: gl.constexpr, B_LOAD_STAGE1: gl.constexpr):
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    if REUSE_A:
        pid_m, pid_n = snapshot_get_xcd_swizzled_pids_reuse_a(
            M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, GROUP_SIZE)
    else:
        pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
            M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, GROUP_SIZE)
    wmma_packed: gl.constexpr = gl.amd.AMDWMMALayout(
        3, WMMA_LAYOUT.transposed, WMMA_LAYOUT.warp_bases,
        WMMA_LAYOUT.reg_bases, [16, 16, 64], WMMA_LAYOUT.cga_layout)
    load_wmma_packed: gl.constexpr = gl.amd.AMDWMMALayout(
        3, LOAD_WMMA_LAYOUT.transposed, LOAD_WMMA_LAYOUT.warp_bases,
        LOAD_WMMA_LAYOUT.reg_bases, [16, 16, 64],
        LOAD_WMMA_LAYOUT.cga_layout)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma_packed, 16)
    dot_b_load: gl.constexpr = gl.DotOperandLayout(
        1, load_wmma_packed, 16)
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [BLOCK_N // 2, 4])
    scale_b_load: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b_load, [BLOCK_N, 4])
    b_warp_hint: gl.constexpr = 0b00001100
    slots: gl.constexpr = 3
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [slots, BLOCK_M, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [slots, BLOCK_N, 128], SHARED_LAYOUT_B)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty,
        [slots, BLOCK_N // 128, 1024], SHARED_SCALE_B)
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
    for prefetch_idx in gl.static_range(slots):
        a_desc, b_desc = fp8_mxfp4_issue_leading4_data(
            a_desc, b_desc, a_buf, b_buf, prefetch_idx, b_warp_hint)
        bs_desc = fp8_mxfp4_issue_leading4_scale(
            bs_desc, bs_buf, prefetch_idx)
    acc0 = gl.zeros(
        (BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    acc1 = gl.zeros(
        (BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 256)
    gl.assume(iter_max >= slots + 1)
    for tile_idx in range(0, iter_max - slots):
        slot = tile_idx % slots
        a_desc, b_desc, bs_desc, acc0, acc1 = (
            fp8_mxfp4_stage8_consume_and_refill(
                a_buf, b_buf, bs_buf, slot, a_desc, b_desc, bs_desc,
                acc0, acc1, dot_a, dot_b, dot_b_load, scale_b_layout,
                scale_b_load, b_warp_hint, BLOCK_M, BLOCK_N, CTA_N,
                WAITCNT, B_LOAD_STAGE1))
    for drain_idx in gl.static_range(slots):
        tdm.async_wait(2 * (slots - 1 - drain_idx))
        slot = (iter_max - slots + drain_idx) % slots
        acc0, acc1 = fp8_mxfp4_stage8_consume_tail(
            a_buf, b_buf, bs_buf, slot, acc0, acc1,
            dot_a, dot_b, dot_b_load, scale_b_layout, scale_b_load,
            BLOCK_M, BLOCK_N, CTA_N)
    snapshot_cluster_wait()
    a_buf._keep_alive()
    b_buf._keep_alive()
    bs_buf._keep_alive()
    mxfp8_tdm_store_split_n2(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1,
        OUTPUT_CGA_LAYOUT, CTA_M, CTA_N)


def build_fp8_mxfp4_stage8_layouts(cluster_width=4):
    """Build retained mixed-format LDS layouts with half-N accumulators."""
    block_m = 256 * cluster_width
    block_n = 256 * cluster_width
    packed_n = 128 * cluster_width
    cga = make_cga_layout(
        [cluster_width, cluster_width], [cluster_width, cluster_width],
        [0, 1])
    output_cga = tuple(tuple(basis) + (0, 0) for basis in cga)
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_m, 256], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[128, 16]], [packed_n, 128], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, packed_n, local_a, local_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=128)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga)
    load_wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        tuple(local_wmma.reg_bases) + ((0, 8),),
        local_wmma.instr_shape, cga)
    load_wmma_packed = gl.amd.AMDWMMALayout(
        3, load_wmma.transposed, load_wmma.warp_bases,
        load_wmma.reg_bases, [16, 16, 64], cga)
    dot_a = gl.DotOperandLayout(0, load_wmma, 16)
    dot_b = gl.DotOperandLayout(1, load_wmma_packed, 16)
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
    return shared_a, shared_b, shared_bs, wmma, load_wmma, output_cga


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
        slot):
    # MXFP4 A and B each have two logical partition pieces, so two issuing
    # warps cover either full operand. This differs from MXFP8 B, whose two
    # partitions and two groups require four warps and force separate A/B
    # copies. Here the disjoint 0-1/2-3 hints are both sufficient and keep the
    # fused operation entirely on the leading wave group.
    tdm.async_load_fused([
        (a_desc, a_buf.index(slot), 0b00000011),
        (b_desc, b_buf.index(slot), 0b00001100),
    ])
    tdm.async_load_fused([
        (as_desc, as_buf.index(slot), 0b00000011),
        (bs_desc, bs_buf.index(slot), 0b00001100),
    ])
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 256])
    as_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, 2048])
    bs_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, 2048])
    return a_desc, b_desc, as_desc, bs_desc


@gluon.jit
def mxfp4_issue_data_refill(
        a_desc, b_desc, a_buf, b_buf, slot):
    tdm.async_load_fused([
        (a_desc, a_buf.index(slot), 0b00000011),
        (b_desc, b_buf.index(slot), 0b00001100),
    ])
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 256])
    return a_desc, b_desc


@gluon.jit
def mxfp4_issue_scale_refill(
        as_desc, bs_desc, as_buf, bs_buf, slot):
    tdm.async_load_fused([
        (as_desc, as_buf.index(slot), 0b00000011),
        (bs_desc, bs_buf.index(slot), 0b00001100),
    ])
    as_desc = tdm.update_tensor_descriptor(
        as_desc, add_offsets=[0, 2048])
    bs_desc = tdm.update_tensor_descriptor(
        bs_desc, add_offsets=[0, 2048])
    return as_desc, bs_desc


@gluon.jit
def mxfp4_consume_k256(
        a_buf, b_buf, as_buf, bs_buf, slot, acc,
        START_K_PACKED: gl.constexpr, START_SCALE_K: gl.constexpr,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr):
    with gl.amd.warp_pipeline_stage("stage0"):
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
    with gl.amd.warp_pipeline_stage("stage1"):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e2m1", b0, bs0, "e2m1", acc)
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e2m1", b1, bs1, "e2m1", acc)
    return acc


@gluon.jit
def mxfp4_consume_k256_split_slots(
        a_buf, b_buf, as_buf, bs_buf, data_slot, scale_slot, acc,
        START_K_PACKED: gl.constexpr, START_SCALE_K: gl.constexpr,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr):
    with gl.amd.warp_pipeline_stage("mxfp4_u2_tail_load"):
        a0 = a_buf.index(data_slot).slice(0, 1024, 0).slice(
            START_K_PACKED, 64, 1).load(layout=DOT_A)
        b0 = b_buf.index(data_slot).slice(0, 1024, 0).slice(
            START_K_PACKED, 64, 1).permute([1, 0]).load(layout=DOT_B)
        as0 = mxfp4_load_scale(
            as_buf, scale_slot, START_SCALE_K, SCALE_A_LAYOUT)
        bs0 = mxfp4_load_scale(
            bs_buf, scale_slot, START_SCALE_K, SCALE_B_LAYOUT)
        a1 = a_buf.index(data_slot).slice(0, 1024, 0).slice(
            START_K_PACKED + 64, 64, 1).load(layout=DOT_A)
        b1 = b_buf.index(data_slot).slice(0, 1024, 0).slice(
            START_K_PACKED + 64, 64, 1).permute([1, 0]).load(layout=DOT_B)
        as1 = mxfp4_load_scale(
            as_buf, scale_slot, START_SCALE_K + 4, SCALE_A_LAYOUT)
        bs1 = mxfp4_load_scale(
            bs_buf, scale_slot, START_SCALE_K + 4, SCALE_B_LAYOUT)
    with gl.amd.warp_pipeline_stage("mxfp4_u2_tail_compute"):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e2m1", b0, bs0, "e2m1", acc)
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e2m1", b1, bs1, "e2m1", acc)
    return acc


@gluon.jit
def mxfp4_consume_tile_split_slots(
        a_buf, b_buf, as_buf, bs_buf, data_slot, scale_slot, acc,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr):
    acc = mxfp4_consume_k256_split_slots(
        a_buf, b_buf, as_buf, bs_buf, data_slot, scale_slot, acc,
        0, 0, DOT_A, DOT_B, SCALE_A_LAYOUT, SCALE_B_LAYOUT)
    return mxfp4_consume_k256_split_slots(
        a_buf, b_buf, as_buf, bs_buf, data_slot, scale_slot, acc,
        128, 8, DOT_A, DOT_B, SCALE_A_LAYOUT, SCALE_B_LAYOUT)


@gluon.jit
def mxfp4_consume_tile(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, DOT_A: gl.constexpr,
        DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr):
    # The retained path deliberately keeps the original K256 grouping/order.
    acc = mxfp4_consume_k256(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, 0, 0, DOT_A, DOT_B,
        SCALE_A_LAYOUT, SCALE_B_LAYOUT)
    return mxfp4_consume_k256(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, 128, 8, DOT_A, DOT_B,
        SCALE_A_LAYOUT, SCALE_B_LAYOUT)


@gluon.jit
def mxfp4_stage8_k_split_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, slot, a_desc, b_desc, as_desc,
        bs_desc, acc, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        WAITCNT: gl.constexpr, B_LOAD_STAGE1: gl.constexpr):
    tdm.async_wait(WAITCNT)
    with gl.amd.warp_pipeline_stage(
            "mxfp4_stage0_load_k0", phase_gap=2):
        a0 = a_buf.index(slot).slice(0, 1024, 0).slice(
            0, 64, 1).load(layout=DOT_A)
        as0 = mxfp4_load_scale(as_buf, slot, 0, SCALE_A_LAYOUT)
        bs0 = mxfp4_load_scale(bs_buf, slot, 0, SCALE_B_LAYOUT)
        a1 = a_buf.index(slot).slice(0, 1024, 0).slice(
            64, 64, 1).load(layout=DOT_A)
        as1 = mxfp4_load_scale(as_buf, slot, 4, SCALE_A_LAYOUT)
        bs1 = mxfp4_load_scale(bs_buf, slot, 4, SCALE_B_LAYOUT)
        if not B_LOAD_STAGE1:
            b0 = b_buf.index(slot).slice(0, 1024, 0).slice(
                0, 64, 1).permute([1, 0]).load(layout=DOT_B)
            b1 = b_buf.index(slot).slice(0, 1024, 0).slice(
                64, 64, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("mxfp4_stage1_bubble"):
        if B_LOAD_STAGE1:
            b0 = b_buf.index(slot).slice(0, 1024, 0).slice(
                0, 64, 1).permute([1, 0]).load(layout=DOT_B)
            b1 = b_buf.index(slot).slice(0, 1024, 0).slice(
                64, 64, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage(
            "mxfp4_stage2_compute_k0"):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e2m1", b0, bs0, "e2m1", acc)
    with gl.amd.warp_pipeline_stage(
            "mxfp4_stage3_compute_k1"):
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e2m1", b1, bs1, "e2m1", acc)
    with gl.amd.warp_pipeline_stage(
            "mxfp4_stage4_load_k2"):
        a2 = a_buf.index(slot).slice(0, 1024, 0).slice(
            128, 64, 1).load(layout=DOT_A)
        as2 = mxfp4_load_scale(as_buf, slot, 8, SCALE_A_LAYOUT)
        bs2 = mxfp4_load_scale(bs_buf, slot, 8, SCALE_B_LAYOUT)
        a3 = a_buf.index(slot).slice(0, 1024, 0).slice(
            192, 64, 1).load(layout=DOT_A)
        as3 = mxfp4_load_scale(as_buf, slot, 12, SCALE_A_LAYOUT)
        bs3 = mxfp4_load_scale(bs_buf, slot, 12, SCALE_B_LAYOUT)
        b2 = b_buf.index(slot).slice(0, 1024, 0).slice(
            128, 64, 1).permute([1, 0]).load(layout=DOT_B)
        b3 = b_buf.index(slot).slice(0, 1024, 0).slice(
            192, 64, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("mxfp4_stage5_bubble"):
        gl.amd.gfx1250.cluster.arrive()
    with gl.amd.warp_pipeline_stage(
            "mxfp4_stage6_compute_k2"):
        acc = gl.amd.gfx1250.wmma_scaled(
            a2, as2, "e2m1", b2, bs2, "e2m1", acc)
    with gl.amd.warp_pipeline_stage(
            "mxfp4_stage7_arrive_compute_k3"):
        acc = gl.amd.gfx1250.wmma_scaled(
            a3, as3, "e2m1", b3, bs3, "e2m1", acc)
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc, as_desc, bs_desc = mxfp4_issue_refill(
            a_desc, b_desc, as_desc, bs_desc,
            a_buf, b_buf, as_buf, bs_buf, slot)
    return a_desc, b_desc, as_desc, bs_desc, acc


@gluon.jit
def mxfp4_unroll2_triple_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, data_slot, scale_slot,
        a_desc, b_desc, as_desc, bs_desc, acc,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        REFILL_SCALE: gl.constexpr):
    # Two TDM operations are issued per complete refill: fused A/B data and
    # fused A/B scales. Leaving three operations outstanding retires exactly
    # the current tile's data and scale pair while preserving the lookahead.
    tdm.async_wait(3)
    with gl.amd.warp_pipeline_stage(
            "mxfp4_u2_stage0_load_k0", phase_gap=2):
        a0 = a_buf.index(data_slot).slice(0, 1024, 0).slice(
            0, 64, 1).load(layout=DOT_A)
        as0 = mxfp4_load_scale(
            as_buf, scale_slot, 0, SCALE_A_LAYOUT)
        bs0 = mxfp4_load_scale(
            bs_buf, scale_slot, 0, SCALE_B_LAYOUT)
        a1 = a_buf.index(data_slot).slice(0, 1024, 0).slice(
            64, 64, 1).load(layout=DOT_A)
        as1 = mxfp4_load_scale(
            as_buf, scale_slot, 4, SCALE_A_LAYOUT)
        bs1 = mxfp4_load_scale(
            bs_buf, scale_slot, 4, SCALE_B_LAYOUT)
    with gl.amd.warp_pipeline_stage("mxfp4_u2_stage1_load_b_k0"):
        b0 = b_buf.index(data_slot).slice(0, 1024, 0).slice(
            0, 64, 1).permute([1, 0]).load(layout=DOT_B)
        b1 = b_buf.index(data_slot).slice(0, 1024, 0).slice(
            64, 64, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("mxfp4_u2_stage2_compute_k0"):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e2m1", b0, bs0, "e2m1", acc)
    with gl.amd.warp_pipeline_stage("mxfp4_u2_stage3_compute_k1"):
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e2m1", b1, bs1, "e2m1", acc)
    with gl.amd.warp_pipeline_stage("mxfp4_u2_stage4_load_k2"):
        a2 = a_buf.index(data_slot).slice(0, 1024, 0).slice(
            128, 64, 1).load(layout=DOT_A)
        b2 = b_buf.index(data_slot).slice(0, 1024, 0).slice(
            128, 64, 1).permute([1, 0]).load(layout=DOT_B)
        as2 = mxfp4_load_scale(
            as_buf, scale_slot, 8, SCALE_A_LAYOUT)
        bs2 = mxfp4_load_scale(
            bs_buf, scale_slot, 8, SCALE_B_LAYOUT)
        a3 = a_buf.index(data_slot).slice(0, 1024, 0).slice(
            192, 64, 1).load(layout=DOT_A)
        b3 = b_buf.index(data_slot).slice(0, 1024, 0).slice(
            192, 64, 1).permute([1, 0]).load(layout=DOT_B)
        as3 = mxfp4_load_scale(
            as_buf, scale_slot, 12, SCALE_A_LAYOUT)
        bs3 = mxfp4_load_scale(
            bs_buf, scale_slot, 12, SCALE_B_LAYOUT)
    with gl.amd.warp_pipeline_stage("mxfp4_u2_stage5_arrive"):
        gl.amd.gfx1250.cluster.arrive()
    with gl.amd.warp_pipeline_stage("mxfp4_u2_stage6_compute_k2"):
        acc = gl.amd.gfx1250.wmma_scaled(
            a2, as2, "e2m1", b2, bs2, "e2m1", acc)
    with gl.amd.warp_pipeline_stage(
            "mxfp4_u2_stage7_compute_k3_refill"):
        acc = gl.amd.gfx1250.wmma_scaled(
            a3, as3, "e2m1", b3, bs3, "e2m1", acc)
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc = mxfp4_issue_data_refill(
            a_desc, b_desc, a_buf, b_buf, data_slot)
        if REFILL_SCALE:
            as_desc, bs_desc = mxfp4_issue_scale_refill(
                as_desc, bs_desc, as_buf, bs_buf, scale_slot)
    return a_desc, b_desc, as_desc, bs_desc, acc


@gluon.jit
def mxfp4_bk512_stage8_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
        WAITCNT: gl.constexpr, B_LOAD_STAGE1: gl.constexpr):
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
        a_desc, b_desc, as_desc, bs_desc = mxfp4_issue_refill(
            a_desc, b_desc, as_desc, bs_desc,
            a_buf, b_buf, as_buf, bs_buf, prefetch_idx)
    acc = gl.zeros((1024, 1024), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 512)
    gl.assume(iter_max >= 3)
    for tile_idx in range(0, iter_max - 2):
        slot = tile_idx % 2
        a_desc, b_desc, as_desc, bs_desc, acc = (
            mxfp4_stage8_k_split_consume_and_refill(
                a_buf, b_buf, as_buf, bs_buf, slot,
                a_desc, b_desc, as_desc, bs_desc, acc, dot_a, dot_b,
                scale_a_layout, scale_b_layout, WAITCNT, B_LOAD_STAGE1))
    tdm.async_wait(WAITCNT)
    penultimate_slot = (iter_max - 2) % 2
    acc = mxfp4_consume_tile(
        a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout)
    tdm.async_wait(0)
    last_slot = (iter_max - 1) % 2
    acc = mxfp4_consume_tile(
        a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout)
    snapshot_cluster_wait()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    snapshot_tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        WMMA_LAYOUT, 1024, 1024, 4)


@gluon.jit
def mxfp4_bk512_unroll2_triple_scale_gfx1250(
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
        a_scale_ptr.type.element_ty, [3, 8, 2048], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty, [3, 8, 2048], SHARED_SCALE_B)
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

    # Queue D0,S0,D1,S1,S2. wait(3) therefore retires D0/S0 and leaves
    # exactly the three operations needed by the two/three-slot lookahead.
    a_desc, b_desc = mxfp4_issue_data_refill(
        a_desc, b_desc, a_buf, b_buf, 0)
    as_desc, bs_desc = mxfp4_issue_scale_refill(
        as_desc, bs_desc, as_buf, bs_buf, 0)
    a_desc, b_desc = mxfp4_issue_data_refill(
        a_desc, b_desc, a_buf, b_buf, 1)
    as_desc, bs_desc = mxfp4_issue_scale_refill(
        as_desc, bs_desc, as_buf, bs_buf, 1)
    as_desc, bs_desc = mxfp4_issue_scale_refill(
        as_desc, bs_desc, as_buf, bs_buf, 2)

    acc = gl.zeros((1024, 1024), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 512)
    gl.assume(iter_max >= 4)
    gl.assume(iter_max % 2 == 0)
    for pair_idx in range(0, (iter_max - 4) // 2):
        tile0 = pair_idx * 2
        scale_slot0 = tile0 % 3
        scale_slot1 = (tile0 + 1) % 3
        a_desc, b_desc, as_desc, bs_desc, acc = (
            mxfp4_unroll2_triple_consume_and_refill(
                a_buf, b_buf, as_buf, bs_buf, 0, scale_slot0,
                a_desc, b_desc, as_desc, bs_desc, acc, dot_a, dot_b,
                scale_a_layout, scale_b_layout, True))
        a_desc, b_desc, as_desc, bs_desc, acc = (
            mxfp4_unroll2_triple_consume_and_refill(
                a_buf, b_buf, as_buf, bs_buf, 1, scale_slot1,
                a_desc, b_desc, as_desc, bs_desc, acc, dot_a, dot_b,
                scale_a_layout, scale_b_layout, True))

    residual_tile = iter_max - 4
    residual_scale0 = residual_tile % 3
    residual_scale1 = (residual_tile + 1) % 3
    a_desc, b_desc, as_desc, bs_desc, acc = (
        mxfp4_unroll2_triple_consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, 0, residual_scale0,
            a_desc, b_desc, as_desc, bs_desc, acc, dot_a, dot_b,
            scale_a_layout, scale_b_layout, True))
    a_desc, b_desc, as_desc, bs_desc, acc = (
        mxfp4_unroll2_triple_consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, 1, residual_scale1,
            a_desc, b_desc, as_desc, bs_desc, acc, dot_a, dot_b,
            scale_a_layout, scale_b_layout, False))

    # The final data-only refill leaves S(i+2),D(i+2),S(i+3),D(i+3).
    # Retire the next data/scale pair while retaining the last pair.
    tdm.async_wait(2)
    acc = mxfp4_consume_tile_split_slots(
        a_buf, b_buf, as_buf, bs_buf, 0, (iter_max - 2) % 3, acc,
        dot_a, dot_b, scale_a_layout, scale_b_layout)
    tdm.async_wait(0)
    acc = mxfp4_consume_tile_split_slots(
        a_buf, b_buf, as_buf, bs_buf, 1, (iter_max - 1) % 3, acc,
        dot_a, dot_b, scale_a_layout, scale_b_layout)
    snapshot_cluster_wait()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
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


def _make_hipblaslt_mxfp8_trig_matrix(rows, k):
    """Port hipBLASLt's E4M3/E8M0 ``trig_float`` CPU generator.

    hipBLASLt uses 32 independently seeded ``std::mt19937`` streams over
    contiguous scale-block ranges. Each stream draws doubles with libstdc++'s
    ``uniform_real_distribution``, takes ``cos(2*pi*u)``, truncates each value
    to an individually scaled E4M3 value, then replaces the 32 individual
    E8M0 exponents with their rounded mean and adjusts the E4M3 exponents.
    """
    if k % 32:
        raise ValueError("MXFP8 trig generation requires K divisible by 32")

    block_size = 32
    num_threads = 32
    seed = 1713573849
    blocks = rows * k // block_size
    if blocks % num_threads:
        raise ValueError(
            "MXFP8 trig generation requires scale blocks divisible by 32")

    data = torch.empty((rows, k), dtype=torch.uint8)
    scales = torch.empty((rows, k // block_size), dtype=torch.uint8)
    data_flat = data.numpy().reshape(-1)
    scale_flat = scales.numpy().reshape(-1)
    blocks_per_thread = blocks // num_threads
    blocks_per_chunk = 32 * 1024
    output_block = 0

    for thread in range(num_threads):
        rng = np.random.RandomState(seed + thread)
        remaining = blocks_per_thread
        while remaining:
            chunk_blocks = min(remaining, blocks_per_chunk)
            count = chunk_blocks * block_size

            # libstdc++ generate_canonical<double, 53> consumes two MT19937
            # words, with the first word as the low 32 bits.
            raw = rng.randint(
                0, 2**32, size=(count, 2), dtype=np.uint32)
            uniform = (
                raw[:, 0].astype(np.float64)
                + raw[:, 1].astype(np.float64) * 2.0**32) * 2.0**-64
            values = np.cos(uniform * (2.0 * np.pi))
            bits = values.view(np.uint64)

            sign = ((bits >> 63) << 7).astype(np.uint8)
            exponent = (
                ((bits >> 52) & 0x7ff).astype(np.int32) - 1023)
            mantissa = (
                (bits & ((1 << 52) - 1)) >> 49).astype(np.uint8)

            # For |cos(angle)| <= 1, the initial E4M3 exponent is zero and
            # its E8M0 scale exponent is floor(log2(abs(value))).
            element_scales = np.maximum(exponent, -127) + 127
            element_scales = element_scales.reshape(chunk_blocks, block_size)
            block_scales = np.floor(
                element_scales.mean(axis=1) + 0.5).astype(np.int32)
            adjusted = element_scales - block_scales[:, None]
            sign = sign.reshape(chunk_blocks, block_size)
            mantissa = mantissa.reshape(chunk_blocks, block_size)

            encoded = np.zeros(
                (chunk_blocks, block_size), dtype=np.uint8)
            normal = adjusted >= -6
            encoded[normal] = (
                sign[normal]
                | ((adjusted[normal] + 7).astype(np.uint8) << 3)
                | mantissa[normal])

            subnormal = (adjusted >= -9) & (adjusted < -6)
            subnormal_mantissa = (mantissa[subnormal] >> 1) | 4
            subnormal_mantissa >>= (
                -7 - adjusted[subnormal]).astype(np.uint8)
            encoded[subnormal] = sign[subnormal] | subnormal_mantissa

            data_begin = output_block * block_size
            data_end = data_begin + count
            data_flat[data_begin:data_end] = encoded.reshape(-1)
            scale_flat[
                output_block:output_block + chunk_blocks
            ] = block_scales.astype(np.uint8)
            output_block += chunk_blocks
            remaining -= chunk_blocks

    return data, scales


def make_hipblaslt_mxfp8_trig_inputs(m, n, k):
    """Generate hipBLASLt-equivalent logical E4M3 data and E8M0 scales."""
    a_bits, a_scale_bits = _make_hipblaslt_mxfp8_trig_matrix(m, k)
    b_bits, b_scale_bits = _make_hipblaslt_mxfp8_trig_matrix(n, k)
    a_scale = MXScaleTensor(size=a_scale_bits.shape)
    b_scale = MXScaleTensor(size=b_scale_bits.shape)
    a_scale.data = a_scale_bits
    b_scale.data = b_scale_bits
    a = a_bits.view(torch.float8_e4m3fn)
    # hipBLASLt stores B as contiguous [N, K]; expose logical [K, N].
    b = b_bits.view(torch.float8_e4m3fn).T
    return a, b, a_scale, b_scale


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
        for _ in range(2):
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
    probe_ms = None
    if args.iters_per_graph is None:
        probe_ms = event_probe(launch, args.probe_iters)
        n_per_graph = max(1, int(args.graph_ms / max(probe_ms, 1e-6)))
    else:
        n_per_graph = args.iters_per_graph
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
    if probe_ms is not None:
        print(f"probe per-iter   : {probe_ms * 1e3:.2f} us")
    print(f"iters per graph  : {n_per_graph}")
    print(f"replays          : {args.replays}")
    print(f"total iters      : {total_iters}")
    print(f"total elapsed    : {elapsed:.6f} s")
    print(f"per-iter         : {per_iter * 1e6:.2f} us")
    print(f"TFLOPS           : {(2 * M * N * K) / per_iter / 1e12:.3f}")


BEST_KERNELS = {
    "bf16": bf16_kernelc_2pf_bk128_fused_cluster_4x4_gfx1250,
    "mxfp8": fp8_scaled_cluster_bk256_opt_kernel_gfx1250,
    "fp8_mxfp4": fp8_mxfp4_stage8_gfx1250,
    "mxfp4": mxfp4_bk512_stage8_gfx1250,
    "mxfp4_u2": mxfp4_bk512_unroll2_triple_scale_gfx1250,
}

TILED_CONFIGS = {
    "64x64x512": (64, 64, 512, False, False),
    "128x128x256": (128, 128, 256, False, True),
}


@gluon.jit
def tiled_issue_refill(
        a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
        slot, BLOCK_K: gl.constexpr, SCALE_STEP: gl.constexpr):
    tdm.async_load(
        a_desc, dest=a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(
        b_desc, dest=b_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load_fused([
        (as_desc, as_buf.index(slot), 0b00000011),
        (bs_desc, bs_buf.index(slot), 0b00001100),
    ])
    a_desc = tdm.update_tensor_descriptor(
        a_desc, add_offsets=[0, BLOCK_K])
    b_desc = tdm.update_tensor_descriptor(
        b_desc, add_offsets=[0, BLOCK_K])
    as_desc = tdm.update_tensor_descriptor(
        as_desc, add_offsets=[0, SCALE_STEP])
    bs_desc = tdm.update_tensor_descriptor(
        bs_desc, add_offsets=[0, SCALE_STEP])
    return a_desc, b_desc, as_desc, bs_desc


@gluon.jit
def tiled_load_scale(
        scale_buf, slot, start_k: gl.constexpr, layout: gl.constexpr):
    return scale_buf.index(slot).slice(
        start_k, 4, 1).load(layout=layout)


@gluon.jit
def tiled_load_scale_preshuffled(
        scale_buf, slot, start_k: gl.constexpr, layout: gl.constexpr,
        block_nonk: gl.constexpr, scale_k: gl.constexpr):
    return mxfp8_load_scale(
        scale_buf, slot, start_k, layout, block_nonk, scale_k, 4)


@gluon.jit
def tiled_split_n(
        b, scale, dot_b: gl.constexpr, scale_b_layout: gl.constexpr,
        block_n: gl.constexpr, cta_n_count: gl.constexpr,
        cta_tile_n: gl.constexpr):
    half_n: gl.constexpr = cta_tile_n // 2
    packed_n: gl.constexpr = block_n // 2
    b_grid = b.reshape((128, cta_n_count, cta_tile_n))
    b_lo = gl.convert_layout(gl.amd.slice(
        b_grid, [128, cta_n_count, half_n], [0, 0, 0]
    ).reshape((128, packed_n)), dot_b, assert_trivial=True)
    b_hi = gl.convert_layout(gl.amd.slice(
        b_grid, [128, cta_n_count, half_n], [0, 0, half_n]
    ).reshape((128, packed_n)), dot_b, assert_trivial=True)
    scale_grid = scale.reshape((cta_n_count, cta_tile_n, 4))
    scale_lo = gl.convert_layout(gl.amd.slice(
        scale_grid, [cta_n_count, half_n, 4], [0, 0, 0]
    ).reshape((packed_n, 4)), scale_b_layout, assert_trivial=True)
    scale_hi = gl.convert_layout(gl.amd.slice(
        scale_grid, [cta_n_count, half_n, 4], [0, half_n, 0]
    ).reshape((packed_n, 4)), scale_b_layout, assert_trivial=True)
    return b_lo, scale_lo, b_hi, scale_hi


@gluon.jit
def tiled_bk256_stage8(
        a_buf, b_buf, as_buf, bs_buf, slot,
        a_desc, b_desc, as_desc, bs_desc, acc0, acc1,
        dot_a: gl.constexpr, dot_b: gl.constexpr,
        dot_b_load: gl.constexpr, scale_a_layout: gl.constexpr,
        scale_b_layout: gl.constexpr, scale_b_load: gl.constexpr,
        block_m: gl.constexpr, block_n: gl.constexpr,
        cta_n_count: gl.constexpr, cta_tile_n: gl.constexpr):
    scale_k: gl.constexpr = 8
    tdm.async_wait(3)
    with gl.amd.warp_pipeline_stage(
            "tiled_stage0_load_k0", phase_gap=2):
        a0 = a_buf.index(slot).slice(0, 128, 1).load(layout=dot_a)
        as0 = tiled_load_scale_preshuffled(
            as_buf, slot, 0, scale_a_layout, block_m, scale_k)
        bs0_full = tiled_load_scale_preshuffled(
            bs_buf, slot, 0, scale_b_load, block_n, scale_k)
    with gl.amd.warp_pipeline_stage("tiled_stage1_load_b0"):
        b0_full = b_buf.index(slot).slice(
            0, 128, 1).permute([1, 0]).load(layout=dot_b_load)
        b0, bs0, b1, bs1 = tiled_split_n(
            b0_full, bs0_full, dot_b, scale_b_layout, block_n,
            cta_n_count, cta_tile_n)
    with gl.amd.warp_pipeline_stage(
            "tiled_stage2_compute_k0_n0"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0, bs0, "e4m3", acc0)
    with gl.amd.warp_pipeline_stage(
            "tiled_stage3_compute_k0_n1"):
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b1, bs1, "e4m3", acc1)
    with gl.amd.warp_pipeline_stage(
            "tiled_stage4_load_k1"):
        a1 = a_buf.index(slot).slice(128, 128, 1).load(layout=dot_a)
        as1 = tiled_load_scale_preshuffled(
            as_buf, slot, 4, scale_a_layout, block_m, scale_k)
        b1_full = b_buf.index(slot).slice(
            128, 128, 1).permute([1, 0]).load(layout=dot_b_load)
        bs1_full = tiled_load_scale_preshuffled(
            bs_buf, slot, 4, scale_b_load, block_n, scale_k)
        b2, bs2, b3, bs3 = tiled_split_n(
            b1_full, bs1_full, dot_b, scale_b_layout, block_n,
            cta_n_count, cta_tile_n)
    with gl.amd.warp_pipeline_stage("tiled_stage5_bubble"):
        pass
    with gl.amd.warp_pipeline_stage(
            "tiled_stage6_compute_k1_n0"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b2, bs2, "e4m3", acc0)
    with gl.amd.warp_pipeline_stage(
            "tiled_stage7_arrive_compute_k1_n1"):
        gl.amd.gfx1250.cluster.arrive()
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b3, bs3, "e4m3", acc1)
    with gl.amd.warp_pipeline_stage(
            "tiled_stage8_wait_refill"):
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc, as_desc, bs_desc = tiled_issue_refill(
            a_desc, b_desc, as_desc, bs_desc,
            a_buf, b_buf, as_buf, bs_buf, slot, 256, 1024)
    return a_desc, b_desc, as_desc, bs_desc, acc0, acc1


@gluon.jit
def tiled_bk256_tail(
        a_buf, b_buf, as_buf, bs_buf, slot, acc0, acc1,
        dot_a: gl.constexpr, dot_b: gl.constexpr,
        dot_b_load: gl.constexpr, scale_a_layout: gl.constexpr,
        scale_b_layout: gl.constexpr, scale_b_load: gl.constexpr,
        block_m: gl.constexpr, block_n: gl.constexpr,
        cta_n_count: gl.constexpr,
        cta_tile_n: gl.constexpr):
    scale_k: gl.constexpr = 8
    with gl.amd.warp_pipeline_stage("tiled_tail_load_k0"):
        a0 = a_buf.index(slot).slice(0, 128, 1).load(layout=dot_a)
        as0 = tiled_load_scale_preshuffled(
            as_buf, slot, 0, scale_a_layout, block_m, scale_k)
        b0_full = b_buf.index(slot).slice(
            0, 128, 1).permute([1, 0]).load(layout=dot_b_load)
        bs0_full = tiled_load_scale_preshuffled(
            bs_buf, slot, 0, scale_b_load, block_n, scale_k)
        b0, bs0, b1, bs1 = tiled_split_n(
            b0_full, bs0_full, dot_b, scale_b_layout, block_n,
            cta_n_count, cta_tile_n)
    with gl.amd.warp_pipeline_stage("tiled_tail_compute_k0"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0, bs0, "e4m3", acc0)
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b1, bs1, "e4m3", acc1)
    with gl.amd.warp_pipeline_stage("tiled_tail_load_k1"):
        a1 = a_buf.index(slot).slice(128, 128, 1).load(layout=dot_a)
        as1 = tiled_load_scale_preshuffled(
            as_buf, slot, 4, scale_a_layout, block_m, scale_k)
        b1_full = b_buf.index(slot).slice(
            128, 128, 1).permute([1, 0]).load(layout=dot_b_load)
        bs1_full = tiled_load_scale_preshuffled(
            bs_buf, slot, 4, scale_b_load, block_n, scale_k)
        b2, bs2, b3, bs3 = tiled_split_n(
            b1_full, bs1_full, dot_b, scale_b_layout, block_n,
            cta_n_count, cta_tile_n)
    with gl.amd.warp_pipeline_stage("tiled_tail_compute_k1"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b2, bs2, "e4m3", acc0)
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b3, bs3, "e4m3", acc1)
    return acc0, acc1


@gluon.jit
def tiled_bk256_single_stage8(
        a_buf, b_buf, as_buf, bs_buf, slot,
        a_desc, b_desc, as_desc, bs_desc, acc,
        dot_a: gl.constexpr, dot_b: gl.constexpr,
        scale_a_layout: gl.constexpr, scale_b_layout: gl.constexpr,
        block_m: gl.constexpr, block_n: gl.constexpr):
    scale_k: gl.constexpr = 8
    tdm.async_wait(3)
    with gl.amd.warp_pipeline_stage(
            "tiled_stage0_load_k0", phase_gap=2):
        a0 = a_buf.index(slot).slice(0, 128, 1).load(layout=dot_a)
        as0 = tiled_load_scale_preshuffled(
            as_buf, slot, 0, scale_a_layout, block_m, scale_k)
        b0 = b_buf.index(slot).slice(
            0, 128, 1).permute([1, 0]).load(layout=dot_b)
        bs0 = tiled_load_scale_preshuffled(
            bs_buf, slot, 0, scale_b_layout, block_n, scale_k)
    with gl.amd.warp_pipeline_stage("tiled_stage1_bubble"):
        pass
    with gl.amd.warp_pipeline_stage(
            "tiled_stage2_compute_k0"):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0, bs0, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("tiled_stage3_bubble"):
        pass
    with gl.amd.warp_pipeline_stage(
            "tiled_stage4_load_k1"):
        a1 = a_buf.index(slot).slice(128, 128, 1).load(layout=dot_a)
        as1 = tiled_load_scale_preshuffled(
            as_buf, slot, 4, scale_a_layout, block_m, scale_k)
        b1 = b_buf.index(slot).slice(
            128, 128, 1).permute([1, 0]).load(layout=dot_b)
        bs1 = tiled_load_scale_preshuffled(
            bs_buf, slot, 4, scale_b_layout, block_n, scale_k)
    with gl.amd.warp_pipeline_stage("tiled_stage5_bubble"):
        pass
    with gl.amd.warp_pipeline_stage("tiled_stage6_bubble"):
        pass
    with gl.amd.warp_pipeline_stage(
            "tiled_stage7_arrive_compute_k1"):
        gl.amd.gfx1250.cluster.arrive()
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1, bs1, "e4m3", acc)
    with gl.amd.warp_pipeline_stage(
            "tiled_stage8_wait_refill"):
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc, as_desc, bs_desc = tiled_issue_refill(
            a_desc, b_desc, as_desc, bs_desc,
            a_buf, b_buf, as_buf, bs_buf, slot, 256, 1024)
    return a_desc, b_desc, as_desc, bs_desc, acc


@gluon.jit
def tiled_bk256_single_tail(
        a_buf, b_buf, as_buf, bs_buf, slot, acc,
        dot_a: gl.constexpr, dot_b: gl.constexpr,
        scale_a_layout: gl.constexpr, scale_b_layout: gl.constexpr,
        block_m: gl.constexpr, block_n: gl.constexpr):
    scale_k: gl.constexpr = 8
    for k in gl.static_range(0, 256, 128):
        with gl.amd.warp_pipeline_stage("tiled_tail_load"):
            a = a_buf.index(slot).slice(k, 128, 1).load(layout=dot_a)
            a_scale = tiled_load_scale_preshuffled(
                as_buf, slot, k // 32, scale_a_layout, block_m, scale_k)
            b = b_buf.index(slot).slice(
                k, 128, 1).permute([1, 0]).load(layout=dot_b)
            b_scale = tiled_load_scale_preshuffled(
                bs_buf, slot, k // 32, scale_b_layout, block_n, scale_k)
        with gl.amd.warp_pipeline_stage("tiled_tail_compute"):
            acc = gl.amd.gfx1250.wmma_scaled(
                a, a_scale, "e4m3", b, b_scale, "e4m3", acc)
    return acc


@gluon.jit
def tiled_bk512_stage8(
        a_buf, b_buf, as_buf, bs_buf, slot,
        a_desc, b_desc, as_desc, bs_desc, acc,
        dot_a: gl.constexpr, dot_b: gl.constexpr,
        scale_a_layout: gl.constexpr, scale_b_layout: gl.constexpr,
        block_m: gl.constexpr, block_n: gl.constexpr):
    tdm.async_wait(3)
    with gl.amd.warp_pipeline_stage(
            "tiled_stage0_load_k0", phase_gap=2):
        a0 = a_buf.index(slot).slice(0, 128, 1).load(layout=dot_a)
        as0 = tiled_load_scale(as_buf, slot, 0, scale_a_layout)
        bs0 = tiled_load_scale(bs_buf, slot, 0, scale_b_layout)
        a1 = a_buf.index(slot).slice(128, 128, 1).load(layout=dot_a)
        as1 = tiled_load_scale(as_buf, slot, 4, scale_a_layout)
        bs1 = tiled_load_scale(bs_buf, slot, 4, scale_b_layout)
    with gl.amd.warp_pipeline_stage("tiled_stage1_load_b01"):
        b0 = b_buf.index(slot).slice(
            0, 128, 1).permute([1, 0]).load(layout=dot_b)
        b1 = b_buf.index(slot).slice(
            128, 128, 1).permute([1, 0]).load(layout=dot_b)
    with gl.amd.warp_pipeline_stage(
            "tiled_stage2_compute_k0"):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0, bs0, "e4m3", acc)
    with gl.amd.warp_pipeline_stage(
            "tiled_stage3_compute_k1"):
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1, bs1, "e4m3", acc)
    with gl.amd.warp_pipeline_stage(
            "tiled_stage4_load_k2"):
        a2 = a_buf.index(slot).slice(256, 128, 1).load(layout=dot_a)
        as2 = tiled_load_scale(as_buf, slot, 8, scale_a_layout)
        b2 = b_buf.index(slot).slice(
            256, 128, 1).permute([1, 0]).load(layout=dot_b)
        bs2 = tiled_load_scale(bs_buf, slot, 8, scale_b_layout)
        a3 = a_buf.index(slot).slice(384, 128, 1).load(layout=dot_a)
        as3 = tiled_load_scale(as_buf, slot, 12, scale_a_layout)
        b3 = b_buf.index(slot).slice(
            384, 128, 1).permute([1, 0]).load(layout=dot_b)
        bs3 = tiled_load_scale(bs_buf, slot, 12, scale_b_layout)
    with gl.amd.warp_pipeline_stage("tiled_stage5_bubble"):
        pass
    with gl.amd.warp_pipeline_stage(
            "tiled_stage6_compute_k2"):
        acc = gl.amd.gfx1250.wmma_scaled(
            a2, as2, "e4m3", b2, bs2, "e4m3", acc)
    with gl.amd.warp_pipeline_stage(
            "tiled_stage7_arrive_compute_k3"):
        gl.amd.gfx1250.cluster.arrive()
        acc = gl.amd.gfx1250.wmma_scaled(
            a3, as3, "e4m3", b3, bs3, "e4m3", acc)
    with gl.amd.warp_pipeline_stage(
            "tiled_stage8_wait_refill"):
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc, as_desc, bs_desc = tiled_issue_refill(
            a_desc, b_desc, as_desc, bs_desc,
            a_buf, b_buf, as_buf, bs_buf, slot, 512, 16)
    return a_desc, b_desc, as_desc, bs_desc, acc


@gluon.jit
def tiled_consume_tail(
        a_buf, b_buf, as_buf, bs_buf, slot, acc,
        dot_a: gl.constexpr, dot_b: gl.constexpr,
        scale_a_layout: gl.constexpr, scale_b_layout: gl.constexpr,
        block_m: gl.constexpr, block_n: gl.constexpr,
        block_k: gl.constexpr):
    for k in gl.static_range(0, block_k, 128):
        with gl.amd.warp_pipeline_stage("tiled_tail_load"):
            a = a_buf.index(slot).slice(k, 128, 1).load(layout=dot_a)
            a_scale = tiled_load_scale(
                as_buf, slot, k // 32, scale_a_layout)
            b = b_buf.index(slot).slice(
                k, 128, 1).permute([1, 0]).load(layout=dot_b)
            b_scale = tiled_load_scale(
                bs_buf, slot, k // 32, scale_b_layout)
        with gl.amd.warp_pipeline_stage("tiled_tail_compute"):
            acc = gl.amd.gfx1250.wmma_scaled(
                a, a_scale, "e4m3", b, b_scale, "e4m3", acc)
    return acc


@gluon.jit
def tiled_store_split_n2(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1,
        output_cga: gl.constexpr, cta_m_count: gl.constexpr,
        cta_n_count: gl.constexpr, cta_tile_m: gl.constexpr,
        cta_tile_n: gl.constexpr):
    half_n: gl.constexpr = cta_tile_n // 2
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[half_n, 8]],
        [cta_m_count, cta_n_count, cta_tile_m, half_n],
        [3, 2, 1, 0], output_cga)
    shared = gl.allocate_shared_memory(
        c_ptr.type.element_ty,
        [cta_m_count, cta_n_count, cta_tile_m, half_n], shared_layout)
    desc = tdm.make_tensor_descriptor(
        base=c_ptr,
        shape=(M // cta_tile_m, N // cta_tile_n, cta_tile_m, cta_tile_n),
        strides=(cta_tile_m * stride_cm, cta_tile_n * stride_cn,
                 stride_cm, stride_cn),
        block_shape=(cta_m_count, cta_n_count, cta_tile_m, half_n),
        layout=shared_layout)
    base_m = pid_m * cta_m_count
    base_n = pid_n * cta_n_count
    output0 = acc0.reshape(
        (cta_m_count, cta_tile_m, cta_n_count, half_n)
    ).permute((0, 2, 1, 3))
    output1 = acc1.reshape(
        (cta_m_count, cta_tile_m, cta_n_count, half_n)
    ).permute((0, 2, 1, 3))
    shared.store(output0.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, 0], shared)
    tdm.async_wait(0)
    shared.store(output1.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, half_n], shared)
    tdm.async_wait(0)


@gluon.jit
def mxfp8_stage8_tiled_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_as, stride_bs, GRID_MN: gl.constexpr,
        shared_a: gl.constexpr, shared_b: gl.constexpr,
        shared_as: gl.constexpr, shared_bs: gl.constexpr,
        wmma: gl.constexpr, load_wmma: gl.constexpr,
        output_cga: gl.constexpr, block_m: gl.constexpr,
        block_n: gl.constexpr, block_k: gl.constexpr,
        cta_m_count: gl.constexpr, cta_n_count: gl.constexpr,
        cta_tile_m: gl.constexpr, cta_tile_n: gl.constexpr,
        split_n: gl.constexpr, preshuffle_scales: gl.constexpr):
    gl.static_assert(gl.num_ctas() == cta_m_count * cta_n_count)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
        M, N, block_m, block_n, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [block_m, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [block_n if not split_n else block_n // 2, 4])
    if split_n:
        dot_b_load: gl.constexpr = gl.DotOperandLayout(1, load_wmma, 16)
        scale_b_load: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
            dot_b_load, [block_n, 4])
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, block_m, block_k], shared_a)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, block_n, block_k], shared_b)
    scale_k: gl.constexpr = block_k // 32
    scale_step: gl.constexpr = (
        scale_k * 128 if preshuffle_scales else scale_k)
    if preshuffle_scales:
        as_buf = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty,
            [2, block_m // 128, scale_step], shared_as)
        bs_buf = gl.allocate_shared_memory(
            b_scale_ptr.type.element_ty,
            [2, block_n // 128, scale_step], shared_bs)
    else:
        as_buf = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty, [2, block_m, scale_k], shared_as)
        bs_buf = gl.allocate_shared_memory(
            b_scale_ptr.type.element_ty, [2, block_n, scale_k], shared_bs)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * block_m * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(block_m, block_k),
        layout=shared_a)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * block_n * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(block_n, block_k),
        layout=shared_b)
    if preshuffle_scales:
        as_desc = tdm.make_tensor_descriptor(
            base=a_scale_ptr + (pid_m * block_m) // 128 * stride_as,
            shape=(M // 128, K // 32 * 128), strides=(stride_as, 1),
            block_shape=(block_m // 128, scale_step), layout=shared_as)
        bs_desc = tdm.make_tensor_descriptor(
            base=b_scale_ptr + (pid_n * block_n) // 128 * stride_bs,
            shape=(N // 128, K // 32 * 128), strides=(stride_bs, 1),
            block_shape=(block_n // 128, scale_step), layout=shared_bs)
    else:
        as_desc = tdm.make_tensor_descriptor(
            base=a_scale_ptr + pid_m * block_m * stride_as,
            shape=(M, K // 32), strides=(stride_as, 1),
            block_shape=(block_m, scale_k), layout=shared_as)
        bs_desc = tdm.make_tensor_descriptor(
            base=b_scale_ptr + pid_n * block_n * stride_bs,
            shape=(N, K // 32), strides=(stride_bs, 1),
            block_shape=(block_n, scale_k), layout=shared_bs)
    for slot in gl.static_range(2):
        a_desc, b_desc, as_desc, bs_desc = tiled_issue_refill(
            a_desc, b_desc, as_desc, bs_desc,
            a_buf, b_buf, as_buf, bs_buf, slot, block_k, scale_step)
    iter_max = gl.cdiv(K, block_k)
    gl.assume(iter_max >= 3)
    if split_n:
        acc0 = gl.zeros(
            (block_m, block_n // 2), dtype=gl.float32, layout=wmma)
        acc1 = gl.zeros(
            (block_m, block_n // 2), dtype=gl.float32, layout=wmma)
        for tile_idx in range(0, iter_max - 2):
            split_slot = tile_idx % 2
            a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
                tiled_bk256_stage8(
                    a_buf, b_buf, as_buf, bs_buf, split_slot,
                    a_desc, b_desc, as_desc, bs_desc, acc0, acc1,
                    dot_a, dot_b, dot_b_load, scale_a_layout,
                    scale_b_layout, scale_b_load, block_m, block_n,
                    cta_n_count, cta_tile_n))
        tdm.async_wait(3)
        penultimate_slot = (iter_max - 2) % 2
        acc0, acc1 = tiled_bk256_tail(
            a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc0, acc1,
            dot_a, dot_b, dot_b_load, scale_a_layout, scale_b_layout,
            scale_b_load, block_m, block_n, cta_n_count, cta_tile_n)
        tdm.async_wait(0)
        last_slot = (iter_max - 1) % 2
        acc0, acc1 = tiled_bk256_tail(
            a_buf, b_buf, as_buf, bs_buf, last_slot, acc0, acc1,
            dot_a, dot_b, dot_b_load, scale_a_layout, scale_b_layout,
            scale_b_load, block_m, block_n, cta_n_count, cta_tile_n)
        snapshot_cluster_wait()
        a_buf._keep_alive()
        b_buf._keep_alive()
        as_buf._keep_alive()
        bs_buf._keep_alive()
        tiled_store_split_n2(
            c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1,
            output_cga, cta_m_count, cta_n_count, cta_tile_m, cta_tile_n)
    else:
        acc = gl.zeros((block_m, block_n), dtype=gl.float32, layout=wmma)
        for tile_idx in range(0, iter_max - 2):
            full_slot = tile_idx % 2
            if block_k == 256:
                a_desc, b_desc, as_desc, bs_desc, acc = (
                    tiled_bk256_single_stage8(
                        a_buf, b_buf, as_buf, bs_buf, full_slot,
                        a_desc, b_desc, as_desc, bs_desc, acc,
                        dot_a, dot_b, scale_a_layout, scale_b_layout,
                        block_m, block_n))
            else:
                a_desc, b_desc, as_desc, bs_desc, acc = tiled_bk512_stage8(
                    a_buf, b_buf, as_buf, bs_buf, full_slot,
                    a_desc, b_desc, as_desc, bs_desc, acc,
                    dot_a, dot_b, scale_a_layout, scale_b_layout,
                    block_m, block_n)
        tdm.async_wait(3)
        penultimate_slot = (iter_max - 2) % 2
        if block_k == 256:
            acc = tiled_bk256_single_tail(
                a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc,
                dot_a, dot_b, scale_a_layout, scale_b_layout,
                block_m, block_n)
        else:
            acc = tiled_consume_tail(
                a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc,
                dot_a, dot_b, scale_a_layout, scale_b_layout,
                block_m, block_n, block_k)
        tdm.async_wait(0)
        last_slot = (iter_max - 1) % 2
        if block_k == 256:
            acc = tiled_bk256_single_tail(
                a_buf, b_buf, as_buf, bs_buf, last_slot, acc,
                dot_a, dot_b, scale_a_layout, scale_b_layout,
                block_m, block_n)
        else:
            acc = tiled_consume_tail(
                a_buf, b_buf, as_buf, bs_buf, last_slot, acc,
                dot_a, dot_b, scale_a_layout, scale_b_layout,
                block_m, block_n, block_k)
        snapshot_cluster_wait()
        a_buf._keep_alive()
        b_buf._keep_alive()
        as_buf._keep_alive()
        bs_buf._keep_alive()
        snapshot_tdm_store_full_tile(
            c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
            wmma, block_m, block_n, cta_n_count)


def build_tiled_layouts(config, cluster_width):
    cta_m, cta_n, block_k, split_n, preshuffle = TILED_CONFIGS[config]
    block_m = cta_m * cluster_width
    block_n = cta_n * cluster_width
    cga = make_cga_layout(
        [cluster_width, cluster_width], [cluster_width, cluster_width],
        [0, 1])
    output_cga = tuple(tuple(basis) + (0, 0) for basis in cga)
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[block_k, 16]], [block_m, block_k], [1, 0])
    compute_n = block_n // 2 if split_n else block_n
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[block_k, 16]], [compute_n, block_k], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, compute_n, local_a, local_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=cta_m,
        slice_n=cta_n // 2 if split_n else cta_n)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga)
    if split_n:
        n_basis = cta_n // 2 // 16
        load_wmma = gl.amd.AMDWMMALayout(
            3, local_wmma.transposed, local_wmma.warp_bases,
            tuple(local_wmma.reg_bases) + ((0, n_basis),),
            local_wmma.instr_shape, cga)
    else:
        load_wmma = wmma
    dot_a = gl.DotOperandLayout(0, load_wmma, 16)
    dot_b = gl.DotOperandLayout(1, load_wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[block_k, 16]], [block_m, block_k], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[block_k, 16]], [block_n, block_k], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        cta_m, cta_n, padded_a, padded_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=cta_m, slice_n=cta_n)
    scale_k = block_k // 32
    if preshuffle:
        scale_step = scale_k * 128
        shared_as = gl.PaddedSharedLayout.with_identity_for(
            [[256, 8]], [block_m // 128, scale_step], [1, 0], cga_a)
        shared_bs = gl.PaddedSharedLayout.with_identity_for(
            [[256, 8]], [block_n // 128, scale_step], [1, 0], cga_b)
    else:
        shared_as = gl.PaddedSharedLayout.with_identity_for(
            [[scale_k, 8]], [block_m, scale_k], [1, 0], cga_a)
        shared_bs = gl.PaddedSharedLayout.with_identity_for(
            [[scale_k, 8]], [block_n, scale_k], [1, 0], cga_b)
    return (shared_a, shared_b, shared_as, shared_bs, wmma, load_wmma,
            output_cga)


def make_tiled_case(args):
    cta_m, cta_n, block_k, split_n, preshuffle = (
        TILED_CONFIGS[args.tiled_config])
    cluster = args.tiled_cluster_width
    block_m = cta_m * cluster
    block_n = cta_n * cluster
    if args.M % block_m or args.N % block_n or args.K % block_k:
        raise ValueError("shape is not divisible by the selected tiled kernel")
    if args.K // block_k < 3:
        raise ValueError("tiled kernel requires at least three K tiles")
    torch.manual_seed(args.seed)
    if args.input_mode == "trig":
        a, b, a_scale_obj, b_scale_obj = (
            make_hipblaslt_mxfp8_trig_inputs(args.M, args.N, args.K))
    else:
        a = init_data("float8_e4m3", args.M, args.K)
        b = init_data("float8_e4m3", args.K, args.N)
        scale_k = args.K // 32
        a_scale_obj = MXScaleTensor(
            size=(args.M, scale_k)).random(low=1.0, high=32.0)
        b_scale_obj = MXScaleTensor(
            size=(args.N, scale_k)).random(low=1.0, high=32.0)
    reference = None
    if args.check:
        reference = torch_gemm_mxfp(
            a, b, a_scale_obj, b_scale_obj, 32,
            args.M, args.N, args.K).to(torch.bfloat16)
    a_d = a.contiguous().cuda()
    b_d = b.T.contiguous().cuda()
    if preshuffle:
        as_d = pack_scale(a_scale_obj.data, 4).cuda()
        bs_d = pack_scale(b_scale_obj.data, 4).cuda()
    else:
        as_d = a_scale_obj.data.contiguous().cuda()
        bs_d = b_scale_obj.data.contiguous().cuda()
    c_d = torch.empty(
        (args.M, args.N), dtype=torch.bfloat16, device="cuda")
    layouts = build_tiled_layouts(args.tiled_config, cluster)
    grid = (
        triton.cdiv(args.M, block_m) * triton.cdiv(args.N, block_n), 1)

    def launch():
        shared_a, shared_b, shared_as, shared_bs, wmma, load_wmma, cga = layouts
        return mxfp8_stage8_tiled_gfx1250[grid](
            a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
            a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1), as_d.stride(0), bs_d.stride(0),
            GRID_MN=grid[0], shared_a=shared_a, shared_b=shared_b,
            shared_as=shared_as, shared_bs=shared_bs, wmma=wmma,
            load_wmma=load_wmma, output_cga=cga,
            block_m=block_m, block_n=block_n, block_k=block_k,
            cta_m_count=cluster, cta_n_count=cluster,
            cta_tile_m=cta_m, cta_tile_n=cta_n, split_n=split_n,
            preshuffle_scales=preshuffle,
            num_warps=8, waves_per_eu=2, num_ctas=cluster * cluster,
            llvm_fn_attrs=AGPR_ATTRS)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            c_d.cpu(), reference, rtol=1e-2, atol=5e-1)
        print("result verified", flush=True)

    return launch, check


def make_best_bf16_case(args):
    if args.M % 1024 or args.N % 1024 or args.K % 128:
        raise ValueError("BF16 requires M/N divisible by 1024 and K by 128")
    if args.K // 128 < 2 or (args.K // 128) % 2:
        raise ValueError("BF16 requires an even number of at least two K tiles")
    torch.manual_seed(args.seed)
    if args.input_mode == "random":
        a = torch.randn(
            (args.M, args.K), dtype=torch.bfloat16, device="cuda")
        b = torch.randn(
            (args.N, args.K), dtype=torch.bfloat16, device="cuda")
    else:
        a = torch.empty(
            (args.M, args.K), dtype=torch.bfloat16, device="cuda")
        b = torch.empty(
            (args.N, args.K), dtype=torch.bfloat16, device="cuda")
        chunk = 8 * 1024 * 1024
        for output, cosine in ((a, False), (b, True)):
            flat = output.view(-1)
            for begin in range(0, flat.numel(), chunk):
                end = min(begin + chunk, flat.numel())
                angle = torch.arange(
                    begin, end, dtype=torch.float64, device="cuda")
                value = torch.cos(angle) if cosine else torch.sin(angle)
                flat[begin:end].copy_(value.float())
    c = torch.zeros(
        (args.M, args.N), dtype=torch.bfloat16, device="cuda")
    layouts = build_bf16_layouts(4, 4)
    grid = (
        triton.cdiv(args.M, 1024) * triton.cdiv(args.N, 1024), 1)

    def launch():
        shared_a, shared_b, wmma = layouts
        return bf16_kernelc_2pf_bk128_fused_cluster_4x4_gfx1250[grid](
            a, b, c, args.M, args.N, args.K,
            a.stride(0), a.stride(1), b.stride(1), b.stride(0),
            c.stride(0), c.stride(1), GRID_MN=grid[0],
            SHARED_LAYOUT_A=shared_a, SHARED_LAYOUT_B=shared_b,
            WMMA_LAYOUT=wmma, BLOCK_M=1024, BLOCK_N=1024,
            CTA_M=4, CTA_N=4, num_warps=8, waves_per_eu=2, num_ctas=16)

    def check():
        c.zero_()
        launch()
        torch.cuda.synchronize()
        reference = (a.cpu().float() @ b.cpu().T.float()).to(torch.bfloat16)
        torch.testing.assert_close(
            c.cpu(), reference, rtol=1e-2, atol=1e-2)
        print("result verified", flush=True)

    return launch, check


def make_best_mxfp8_case(args):
    block_m = block_n = 1024
    if args.M % block_m or args.N % block_n or args.K % 256:
        raise ValueError("MXFP8 requires M/N divisible by 1024 and K by 256")
    if args.K // 256 < 3:
        raise ValueError("MXFP8 requires at least three K tiles")
    m_split = getattr(args, "mxfp8_m_split_accumulators", False)
    if m_split and (args.K // 256 < 4 or (args.K // 256) % 2):
        raise ValueError(
            "M-split MXFP8 requires an even number of at least four K tiles")
    if (getattr(args, "mxfp8_unroll_two_const_slots", False)
            and (args.K // 256 < 4 or (args.K // 256) % 2)):
        raise ValueError(
            "two-iteration MXFP8 unroll requires an even number of at least "
            "four K tiles")
    if (getattr(args, "mxfp8_unroll_four_const_slots", False)
            and (args.K // 256 < 4 or (args.K // 256) % 4)):
        raise ValueError(
            "four-iteration MXFP8 unroll requires a multiple of four K tiles")
    if (getattr(args, "mxfp8_unroll_two_const_slots", False)
            and getattr(args, "mxfp8_unroll_four_const_slots", False)):
        raise ValueError("select only one MXFP8 unroll factor")
    if (getattr(args, "mxfp8_separate_slot_descriptors", False)
            and not getattr(args, "mxfp8_unroll_two_const_slots", False)):
        raise ValueError(
            "separate slot descriptors require two-iteration unrolling")
    paired_scales = getattr(args, "mxfp8_paired_scale_tdm", False)
    if paired_scales and not getattr(
            args, "mxfp8_unroll_two_const_slots", False):
        raise ValueError("paired scale TDM requires two-iteration unrolling")
    triple_scale_ring = getattr(args, "mxfp8_triple_scale_ring", False)
    if triple_scale_ring and not getattr(
            args, "mxfp8_unroll_two_const_slots", False):
        raise ValueError("triple scale ring requires two-iteration unrolling")
    four_scale_ring = getattr(args, "mxfp8_four_scale_ring", False)
    if four_scale_ring and not getattr(
            args, "mxfp8_unroll_two_const_slots", False):
        raise ValueError("four-slot scale ring requires two-iteration unrolling")
    if sum((paired_scales, triple_scale_ring, four_scale_ring)) > 1:
        raise ValueError("select only one experimental scale buffering scheme")
    torch.manual_seed(args.seed)
    if args.input_mode == "trig":
        a, b, a_scale_obj, b_scale_obj = (
            make_hipblaslt_mxfp8_trig_inputs(args.M, args.N, args.K))
    else:
        a = init_data("float8_e4m3", args.M, args.K)
        b = init_data("float8_e4m3", args.K, args.N)
        scale_k = args.K // 32
        a_scale_obj = MXScaleTensor(
            size=(args.M, scale_k)).random(low=1.0, high=32.0)
        b_scale_obj = MXScaleTensor(
            size=(args.N, scale_k)).random(low=1.0, high=32.0)
    reference = None
    if args.check:
        reference = torch_gemm_mxfp(
            a, b, a_scale_obj, b_scale_obj, 32,
            args.M, args.N, args.K).to(torch.bfloat16)
    a_d = a.contiguous().cuda()
    b_d = b.T.contiguous().cuda()
    as_d = pack_scale(a_scale_obj.data, 4).cuda()
    bs_d = pack_scale(b_scale_obj.data, 4).cuda()
    c_d = torch.empty(
        (args.M, args.N), dtype=torch.bfloat16, device="cuda")
    layouts = (
        build_mxfp8_bk256_msplit_layouts(4, 4)
        if m_split else (
            build_mxfp8_bk256_pair_scale_layouts(4, 4)
            if paired_scales else build_mxfp8_bk256_opt_layouts(4, 4)))
    grid = (
        triton.cdiv(args.M, block_m) * triton.cdiv(args.N, block_n), 1)

    def launch():
        shared_a, shared_b, shared_as, shared_bs, wmma, load_wmma, cga = layouts
        if m_split:
            return fp8_scaled_cluster_bk256_msplit_kernel_gfx1250[grid](
                a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
                a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
                c_d.stride(0), c_d.stride(1),
                as_d.stride(0), bs_d.stride(0),
                GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
                SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
                SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
                LOAD_WMMA_LAYOUT=load_wmma, OUTPUT_CGA_LAYOUT=cga,
                BLOCK_M=block_m, BLOCK_N=block_n, CTA_M=4, CTA_N=4,
                WAIT_REFILL_FIRST=getattr(
                    args, "mxfp8_stage6_wait_refill_first", False),
                num_warps=8, waves_per_eu=2, num_ctas=16,
                llvm_fn_attrs=AGPR_ATTRS)
        return fp8_scaled_cluster_bk256_opt_kernel_gfx1250[grid](
            a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
            a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1), as_d.stride(0), bs_d.stride(0),
            GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
            SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
            LOAD_WMMA_LAYOUT=load_wmma, OUTPUT_CGA_LAYOUT=cga,
            BLOCK_M=block_m, BLOCK_N=block_n, CTA_M=4, CTA_N=4,
            PID_REUSE_A=getattr(args, "mxfp8_pid_reuse", "b") == "a",
            PID_GROUP_SIZE=getattr(args, "mxfp8_pid_group_size", 4),
            STAGE8_REFILL=True, TWO_TDM=False,
            B_SCALE_CONTIGUOUS=False, A_SCALE_COMBINED=False,
            DESC_UPDATE_STAGE5=(
                getattr(
                    args, "mxfp8_desc_update_stage5_after_arrive", False)
                or getattr(
                    args, "mxfp8_stage5_advance_wait_refill_first", False)),
            SCALE_LOAD_STAGE1=False,
            B_LOAD_STAGE1=True, num_warps=8, waves_per_eu=2, num_ctas=16,
            STAGE6_WAIT_REFILL_FIRST=(
                getattr(args, "mxfp8_stage6_wait_refill_first", False)
                or getattr(args, "mxfp8_stage7_cluster_sync", False)),
            EARLY_CLUSTER_SYNC=getattr(
                args, "mxfp8_early_cluster_sync", False),
            STAGE7_CLUSTER_SYNC=getattr(
                args, "mxfp8_stage7_cluster_sync", False),
            UNROLL_TWO_CONST_SLOTS=getattr(
                args, "mxfp8_unroll_two_const_slots", False),
            UNROLL_FOUR_CONST_SLOTS=getattr(
                args, "mxfp8_unroll_four_const_slots", False),
            SEPARATE_SLOT_DESCRIPTORS=getattr(
                args, "mxfp8_separate_slot_descriptors", False),
            STAGE5_ADVANCE_WAIT_REFILL_FIRST=getattr(
                args, "mxfp8_stage5_advance_wait_refill_first", False),
            ARRIVE_STAGE1_WAIT_STAGE7=getattr(
                args, "mxfp8_arrive_stage1_wait_stage7", False),
            PAIRED_SCALE_TDM=paired_scales,
            TRIPLE_SCALE_RING=triple_scale_ring,
            FOUR_SCALE_RING=four_scale_ring,
            TRIPLE_REFILL_DATA_BEFORE_WMMA=getattr(
                args, "mxfp8_triple_refill_data_before_wmma", False),
            TRIPLE_STAGE5_ADVANCE_REFILL_ALL_BEFORE_WMMA=getattr(
                args, "mxfp8_triple_stage5_advance_refill_all_before_wmma",
                False),
            TRIPLE_STAGE5_ADVANCE_ORIGINAL_STAGE7_ORDER=getattr(
                args, "mxfp8_triple_stage5_advance_original_stage7_order",
                False),
            PIPELINED_OUTPUT_STORE=getattr(
                args, "mxfp8_pipelined_output_store", False),
            llvm_fn_attrs=AGPR_ATTRS)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            c_d.cpu(), reference, rtol=1e-2, atol=5e-1)
        print("result verified", flush=True)

    return launch, check


def make_best_fp8_mxfp4_case(args):
    block_m = block_n = 1024
    if args.M % block_m or args.N % block_n or args.K % 256:
        raise ValueError(
            "FP8xMXFP4 requires M/N divisible by 1024 and K by 256")
    if args.K // 256 < 4:
        raise ValueError("FP8xMXFP4 requires at least four K tiles")
    torch.manual_seed(args.seed)
    if args.input_mode == "trig":
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
    reference = None
    if args.check:
        a_scale_obj = MXScaleTensor(data=torch.ones(
            (args.M, scale_k), dtype=torch.float32))
        reference = torch_gemm_mxfp(
            a, b, a_scale_obj, b_scale_obj, 32,
            args.M, args.N, args.K).to(torch.bfloat16)
    a_d = a.contiguous().cuda()
    b_d = b.to_packed_tensor(dim=0).data.T.contiguous().cuda()
    bs_d = pack_scale(b_scale_obj.data, 4).cuda()
    c_d = torch.empty(
        (args.M, args.N), dtype=torch.bfloat16, device="cuda")
    layouts = build_fp8_mxfp4_stage8_layouts(4)
    grid = (
        triton.cdiv(args.M, block_m) * triton.cdiv(args.N, block_n), 1)
    group_size = 8 if args.K <= 8192 else 4

    def launch():
        shared_a, shared_b, shared_bs, wmma, load_wmma, cga = layouts
        return fp8_mxfp4_stage8_gfx1250[grid](
            a_d, b_d, c_d, bs_d, args.M, args.N, args.K,
            a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1), bs_d.stride(0),
            GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, SHARED_SCALE_B=shared_bs,
            WMMA_LAYOUT=wmma, LOAD_WMMA_LAYOUT=load_wmma,
            OUTPUT_CGA_LAYOUT=cga, BLOCK_M=block_m, BLOCK_N=block_n,
            CTA_M=4, CTA_N=4, REUSE_A=True, GROUP_SIZE=group_size,
            WAITCNT=4, B_LOAD_STAGE1=True, num_warps=8, waves_per_eu=2,
            num_ctas=16, llvm_fn_attrs=AGPR_ATTRS)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            c_d.cpu(), reference, rtol=1e-2, atol=5e-1)
        print("result verified", flush=True)

    return launch, check


def make_best_mxfp4_case(args):
    if args.M % 1024 or args.N % 1024 or args.K % 512:
        raise ValueError("MXFP4 requires M/N divisible by 1024 and K by 512")
    if args.K // 512 < 4 or (args.K // 512) % 2:
        raise ValueError(
            "MXFP4 requires an even number of at least four K tiles")
    torch.manual_seed(args.seed)

    def make_mxfp4_input(shape, cosine):
        if args.input_mode != "trig":
            return init_data("float4", shape[0], shape[1])
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

    a = make_mxfp4_input((args.M, args.K), False)
    b = make_mxfp4_input((args.K, args.N), True)
    scale_k = args.K // 32
    a_scale_obj = MXScaleTensor(
        size=(args.M, scale_k)).random(low=1.0, high=32.0)
    b_scale_obj = MXScaleTensor(
        size=(args.N, scale_k)).random(low=1.0, high=32.0)
    reference = None
    if args.check:
        reference = torch_gemm_mxfp(
            a, b, a_scale_obj, b_scale_obj, 32,
            args.M, args.N, args.K).to(torch.bfloat16)
    as_d = pack_scale(a_scale_obj.data, 4).cuda()
    bs_d = pack_scale(b_scale_obj.data, 4).cuda()
    a_d = a.to_packed_tensor(dim=1).data.contiguous().cuda()
    b_d = b.to_packed_tensor(dim=0).data.T.contiguous().cuda()
    c_d = torch.empty(
        (args.M, args.N), dtype=torch.bfloat16, device="cuda")
    layouts = build_mxfp4_layouts()
    grid = (
        triton.cdiv(args.M, 1024) * triton.cdiv(args.N, 1024), 1)

    def launch():
        shared_a, shared_b, shared_as, shared_bs, wmma = layouts
        use_unroll2 = getattr(args, "kernel", "mxfp4") == "mxfp4_u2"
        kernel = (
            mxfp4_bk512_unroll2_triple_scale_gfx1250
            if use_unroll2
            else mxfp4_bk512_stage8_gfx1250)
        common_kwargs = dict(
            GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
            SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
            num_warps=8, waves_per_eu=2, num_ctas=16,
            llvm_fn_attrs=AGPR_ATTRS)
        if use_unroll2:
            return kernel[grid](
                a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
                a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
                c_d.stride(0), c_d.stride(1), bs_d.stride(0),
                **common_kwargs)
        return kernel[grid](
            a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
            a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1), bs_d.stride(0),
            WAITCNT=2, B_LOAD_STAGE1=True,
            **common_kwargs)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            c_d.cpu(), reference, rtol=1e-2, atol=5e-1)
        print("result verified", flush=True)

    return launch, check


BEST_CASE_BUILDERS = {
    "bf16": make_best_bf16_case,
    "mxfp8": make_best_mxfp8_case,
    "fp8_mxfp4": make_best_fp8_mxfp4_case,
    "mxfp4": make_best_mxfp4_case,
    "mxfp4_u2": make_best_mxfp4_case,
}


def populate_best_defaults(args):
    args.bf16_output = True
    args.bf16_cluster_shape = (4, 4)
    args.mxfp8_cluster_width = 4
    args.mxfp8_cluster_shape = None
    args.mxfp8_bk128_schedule = "refill-load"
    args.fp8_mxfp4_cluster_width = 4
    args.fp8_mxfp4_num_buffers = 3
    args.fp8_mxfp4_reuse_order = "a"
    args.fp8_mxfp4_group_size = None
    args.fp8_mxfp4_stage8_waitcnt = 4
    args.mxfp4_stage8_waitcnt = 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel",
        choices=(
            "bf16", "mxfp8", "fp8_mxfp4", "mxfp4", "mxfp4_u2",
            "mxfp8_tiled"),
        required=True)
    parser.add_argument(
        "--tiled-config", choices=tuple(TILED_CONFIGS),
        default="128x128x256")
    parser.add_argument(
        "--tiled-cluster-width", type=int, choices=(2, 4), default=4)
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-K", type=int, default=65536)
    parser.add_argument(
        "--input-mode", choices=("random", "trig"), default="trig")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mxfp8-stage6-wait-refill-first", action="store_true",
        help="wait after stage-6 WMMA, then refill before stage-7 WMMA")
    parser.add_argument(
        "--mxfp8-early-cluster-sync", action="store_true",
        help="move cluster arrive/wait to stages 1/5")
    parser.add_argument(
        "--mxfp8-stage7-cluster-sync", action="store_true",
        help="put cluster arrive/wait before refill at stage 7")
    parser.add_argument(
        "--mxfp8-desc-update-stage5-after-arrive", action="store_true",
        help="advance descriptors in stage 5 after cluster arrival")
    parser.add_argument(
        "--mxfp8-stage5-advance-wait-refill-first", action="store_true",
        help="stage 5 arrive/advance/wait, then stage 7 refill/WMMA")
    parser.add_argument(
        "--mxfp8-arrive-stage1-wait-stage7", action="store_true",
        help="move cluster arrival to stage 1 and retain stage-7 wait")
    parser.add_argument(
        "--mxfp8-pid-reuse", choices=("a", "b"), default="b",
        help="group program IDs to reuse matrix A or B within each XCD")
    parser.add_argument(
        "--mxfp8-pid-group-size", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument(
        "--mxfp8-unroll-two-const-slots", action="store_true",
        help="unroll two K tiles with literal LDS slots 0 and 1")
    parser.add_argument(
        "--mxfp8-unroll-four-const-slots", action="store_true",
        help="unroll four K tiles with literal LDS slots 0, 1, 0, 1")
    parser.add_argument(
        "--mxfp8-separate-slot-descriptors", action="store_true",
        help="carry BK512 descriptor streams for slots 0 and 1")
    parser.add_argument(
        "--mxfp8-paired-scale-tdm", action="store_true",
        help="load both unroll2 scale tiles during the slot-0 body")
    parser.add_argument(
        "--mxfp8-triple-scale-ring", action="store_true",
        help="prefetch scales three tiles ahead with two data slots")
    parser.add_argument(
        "--mxfp8-four-scale-ring", action="store_true",
        help="use four scale slots while retaining distance-three prefetch")
    parser.add_argument(
        "--mxfp8-triple-refill-data-before-wmma", action="store_true",
        help="in triple-ring stage 7, refill A/B before the second WMMA")
    parser.add_argument(
        "--mxfp8-triple-stage5-advance-refill-all-before-wmma",
        action="store_true",
        help="advance in stage 5, then refill A/B/scales before stage-7 WMMA")
    parser.add_argument(
        "--mxfp8-triple-stage5-advance-original-stage7-order",
        action="store_true",
        help="advance in stage 5 while retaining WMMA-before-refill stage 7")
    parser.add_argument(
        "--mxfp8-pipelined-output-store", action="store_true",
        help="overlap two output-half TDM stores using separate LDS buffers")
    parser.add_argument(
        "--mxfp8-m-split-accumulators", action="store_true",
        help="split each CTA accumulator into two M128 by N256 halves")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--probe-iters", type=int, default=20)
    parser.add_argument("--graph-ms", type=float, default=100.0)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--iters-per-graph", type=int, default=50)
    args = parser.parse_args()
    if not args.check and not args.benchmark:
        parser.error("select --check and/or --benchmark")
    populate_best_defaults(args)
    builders = {**BEST_CASE_BUILDERS, "mxfp8_tiled": make_tiled_case}
    launch, check = builders[args.kernel](args)
    if args.check:
        check()
    if args.benchmark:
        run_benchmark(launch, args.M, args.N, args.K, args)
    del launch, check
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
