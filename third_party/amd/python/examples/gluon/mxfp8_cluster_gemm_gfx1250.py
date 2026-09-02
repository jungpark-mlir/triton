# Usage (the cluster shape is selected automatically):
#   # 4x4 cluster: M and N are divisible by 1024.
#   python3 third_party/amd/python/examples/gluon/mxfp8_cluster_gemm_gfx1250.py \
#       -M 4096 -N 4096 -K 65536 --fp8-bf16-output --benchmark
#
#   # 2x2 cluster: M and N are divisible by 512, but not both by 1024.
#   python3 third_party/amd/python/examples/gluon/mxfp8_cluster_gemm_gfx1250.py \
#       -M 512 -N 65536 -K 1536 --fp8-bf16-output --benchmark
#
#   # 1x1 cluster: M and N are divisible by 256, but not both by 512.
#   python3 third_party/amd/python/examples/gluon/mxfp8_cluster_gemm_gfx1250.py \
#       -M 256 -N 768 -K 1024 --check

"""Leading4 MXFP8 GEMM for gfx1250 with automatic cluster selection.

The host selects the largest supported square cluster whose aggregate tile
divides M and N. Each CTA computes 256x256 output elements using two
scaled-WMMA K=128 steps per BK=256 iteration.
"""

import argparse
import time

import torch
import triton
from triton._C.libtriton.gluon_ir import make_cga_layout
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm

try:
    from .mxfp_gemm_cdna5 import MXScaleTensor, init_data, torch_gemm_mxfp
except ImportError:
    from mxfp_gemm_cdna5 import MXScaleTensor, init_data, torch_gemm_mxfp


FP8_DTYPE_TO_KERNEL = {
    "float8_e5m2": "e5m2",
    "float8_e4m3": "e4m3",
}
FP8_DTYPE_TO_TORCH = {
    "float8_e5m2": torch.float8_e5m2,
    "float8_e4m3": torch.float8_e4m3fn,
}

CTA_TILE = 256
BLOCK_K = 256
NUM_WARPS = 8
NUM_BUFFERS = 2
SUPPORTED_CLUSTER_WIDTHS = (4, 2, 1)


@gluon.jit
def _get_tile_ids(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, GRID_MN: gl.constexpr,
                  NUM_XCDS: gl.constexpr, GROUP_SIZE_M: gl.constexpr):
    """Map a linear program ID to an XCD- and M-group-swizzled output tile."""
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
            pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        pids_per_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // pids_per_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % pids_per_group) % group_size_m
        pid_n = (pid % pids_per_group) // group_size_m
    return pid_m, pid_n


@gluon.jit
def _tdm_store_output(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
                      STORE_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                      CTA_N: gl.constexpr):
    # Eight 16-bit elements of row padding avoid the bank conflicts of the
    # identity layout and outperform the swizzled TDM-store layout.
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N // CTA_N, 8]], [BLOCK_M, BLOCK_N], [1, 0], STORE_LAYOUT.cga_layout)
    shared = gl.allocate_shared_memory(c_ptr.type.element_ty, [BLOCK_M, BLOCK_N], shared_layout)
    shared.store(acc.to(c_ptr.type.element_ty))
    desc = tdm.make_tensor_descriptor(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        block_shape=(BLOCK_M, BLOCK_N), layout=shared_layout)
    tdm.async_store(desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], shared)
    tdm.async_wait(0)


@gluon.jit
def _tdm_store_output_split_n2(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_M: gl.constexpr, CTA_N: gl.constexpr):
    """Overlap the first half-tile TDM store with staging the second half."""
    gl.static_assert(CTA_M == 2 and CTA_N == 2)
    cta_m: gl.constexpr = BLOCK_M // CTA_M
    cta_n: gl.constexpr = BLOCK_N // CTA_N
    half_n: gl.constexpr = cta_n // 2
    cga_layout: gl.constexpr = ((1, 0, 0, 0), (0, 1, 0, 0))
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[half_n, 8]], [CTA_M, CTA_N, cta_m, half_n], [3, 2, 1, 0],
        cga_layout)
    shared0 = gl.allocate_shared_memory(
        c_ptr.type.element_ty, [CTA_M, CTA_N, cta_m, half_n], shared_layout)
    shared1 = gl.allocate_shared_memory(
        c_ptr.type.element_ty, [CTA_M, CTA_N, cta_m, half_n], shared_layout)

    acc4 = acc.reshape((CTA_M, cta_m, CTA_N, cta_n)).permute((0, 2, 1, 3))
    output0 = gl.amd.slice(
        acc4, [CTA_M, CTA_N, cta_m, half_n], [0, 0, 0, 0])
    output1 = gl.amd.slice(
        acc4, [CTA_M, CTA_N, cta_m, half_n], [0, 0, 0, half_n])
    shared0.store(output0.to(c_ptr.type.element_ty))

    desc = tdm.make_tensor_descriptor(
        base=c_ptr,
        shape=(M // cta_m, N // cta_n, cta_m, cta_n),
        strides=(cta_m * stride_cm, cta_n * stride_cn, stride_cm, stride_cn),
        block_shape=(CTA_M, CTA_N, cta_m, half_n),
        layout=shared_layout)
    tdm.async_store(
        desc, [pid_m * CTA_M, pid_n * CTA_N, 0, 0], shared0)

    shared1.store(output1.to(c_ptr.type.element_ty))
    tdm.async_store(
        desc, [pid_m * CTA_M, pid_n * CTA_N, 0, half_n], shared1)
    tdm.async_wait(0)


@gluon.jit
def _tdm_store_output_quarters(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_M: gl.constexpr, CTA_N: gl.constexpr):
    """Pipeline four quarter-tile TDM stores behind subsequent LDS staging."""
    gl.static_assert(CTA_M == 2 and CTA_N == 2)
    cta_m: gl.constexpr = BLOCK_M // CTA_M
    cta_n: gl.constexpr = BLOCK_N // CTA_N
    quarter_m: gl.constexpr = cta_m // 2
    quarter_n: gl.constexpr = cta_n // 2
    cga_layout: gl.constexpr = ((1, 0, 0, 0), (0, 1, 0, 0))
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[quarter_n, 8]], [CTA_M, CTA_N, quarter_m, quarter_n],
        [3, 2, 1, 0], cga_layout)
    shared0 = gl.allocate_shared_memory(
        c_ptr.type.element_ty,
        [CTA_M, CTA_N, quarter_m, quarter_n], shared_layout)
    shared1 = gl.allocate_shared_memory(
        c_ptr.type.element_ty,
        [CTA_M, CTA_N, quarter_m, quarter_n], shared_layout)
    shared2 = gl.allocate_shared_memory(
        c_ptr.type.element_ty,
        [CTA_M, CTA_N, quarter_m, quarter_n], shared_layout)
    shared3 = gl.allocate_shared_memory(
        c_ptr.type.element_ty,
        [CTA_M, CTA_N, quarter_m, quarter_n], shared_layout)

    acc4 = acc.reshape((CTA_M, cta_m, CTA_N, cta_n)).permute((0, 2, 1, 3))
    output0 = gl.amd.slice(
        acc4, [CTA_M, CTA_N, quarter_m, quarter_n], [0, 0, 0, 0])
    output1 = gl.amd.slice(
        acc4, [CTA_M, CTA_N, quarter_m, quarter_n],
        [0, 0, 0, quarter_n])
    output2 = gl.amd.slice(
        acc4, [CTA_M, CTA_N, quarter_m, quarter_n],
        [0, 0, quarter_m, 0])
    output3 = gl.amd.slice(
        acc4, [CTA_M, CTA_N, quarter_m, quarter_n],
        [0, 0, quarter_m, quarter_n])

    desc = tdm.make_tensor_descriptor(
        base=c_ptr,
        shape=(M // cta_m, N // cta_n, cta_m, cta_n),
        strides=(cta_m * stride_cm, cta_n * stride_cn, stride_cm, stride_cn),
        block_shape=(CTA_M, CTA_N, quarter_m, quarter_n),
        layout=shared_layout)
    base_m = pid_m * CTA_M
    base_n = pid_n * CTA_N

    shared0.store(output0.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, 0], shared0)
    shared1.store(output1.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, quarter_n], shared1)
    shared2.store(output2.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, quarter_m, 0], shared2)
    shared3.store(output3.to(c_ptr.type.element_ty))
    tdm.async_store(
        desc, [base_m, base_n, quarter_m, quarter_n], shared3)
    tdm.async_wait(0)


@gluon.jit
def _cluster_sync():
    gl.amd.gfx1250.cluster.arrive()
    gl.amd.gfx1250.cluster.wait()


@gluon.jit
def _load_scale(shared, slot, start_nonk: gl.constexpr, start_k: gl.constexpr,
                LAYOUT: gl.constexpr, BLOCK_NONK: gl.constexpr, BK_SCALE: gl.constexpr,
                SUBTILE_NONK: gl.constexpr, SUBTILE_SCALE_K: gl.constexpr,
                PRESHUFFLE_FACTOR: gl.constexpr, SCALE_KWIDTH: gl.constexpr):
    scale = shared.index(slot).reshape(
        (BLOCK_NONK // PRESHUFFLE_FACTOR, BK_SCALE // SCALE_KWIDTH,
         PRESHUFFLE_FACTOR // 4, 4, SCALE_KWIDTH))
    scale = scale.permute((0, 3, 2, 1, 4)).reshape((BLOCK_NONK, BK_SCALE))
    return scale.slice(start_nonk, SUBTILE_NONK, 0).slice(
        start_k, SUBTILE_SCALE_K, 1).load(layout=LAYOUT)


@gluon.jit
def _issue_leading4_scale_load(as_desc, bs_desc, as_buf, bs_buf, tile_idx, slot,
                               BK_SCALE_PRESHUFFLED: gl.constexpr):
    scale_k = tile_idx * BK_SCALE_PRESHUFFLED
    as_load = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, scale_k])
    bs_load = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, scale_k])
    tdm.async_load_fused([
        (as_load, as_buf.index(slot), 0b00000011),
        (bs_load, bs_buf.index(slot), 0b00001100),
    ])


@gluon.jit
def _issue_leading4_data_load(a_desc, b_desc, a_buf, b_buf, tile_idx, slot,
                              BLOCK_K: gl.constexpr):
    tile_k = tile_idx * BLOCK_K
    a_load = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, tile_k])
    b_load = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, tile_k])
    tdm.async_load_fused([
        (a_load, a_buf.index(slot), 0b00000011),
        (b_load, b_buf.index(slot), 0b00001100),
    ])


def _build_cluster_layouts(
        block_m, block_n, scale_block, cga_layout, split_output_store=False,
        quarter_output_store=False):
    """Build CGA-distributed operand, scale, and accumulator layouts."""
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 16]], [block_m, BLOCK_K], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 16]], [block_n, BLOCK_K], [1, 0])
    slice_m = CTA_TILE // 2 if quarter_output_store else CTA_TILE
    slice_n = CTA_TILE // 2 if split_output_store or quarter_output_store else CTA_TILE
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, block_n, local_a, local_b, NUM_WARPS, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=slice_m, slice_n=slice_n)

    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases, local_wmma.reg_bases,
        local_wmma.instr_shape, cga_layout)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]]) for basis in dot_b.cga_layout)

    shared_a = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 16]], [block_m, BLOCK_K], [1, 0], cga_a)
    shared_b = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 16]], [block_n, BLOCK_K], [1, 0], cga_b)

    bk_scale = BLOCK_K // scale_block
    preshuffle_factor = 128
    shared_scale_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_m // preshuffle_factor, bk_scale * preshuffle_factor],
        [1, 0], cga_a)
    shared_scale_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_n // preshuffle_factor, bk_scale * preshuffle_factor],
        [1, 0], cga_b)

    return shared_a, shared_b, shared_scale_a, shared_scale_b, wmma


@gluon.jit
def _issue_refill(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
                  tile_idx, slot, BLOCK_K: gl.constexpr,
                  BK_SCALE_PRESHUFFLED: gl.constexpr):
    """Issue one complete refill from the leading four warps."""
    _issue_leading4_data_load(a_desc, b_desc, a_buf, b_buf, tile_idx, slot, BLOCK_K)
    _issue_leading4_scale_load(
        as_desc, bs_desc, as_buf, bs_buf, tile_idx, slot, BK_SCALE_PRESHUFFLED)


@gluon.jit
def _consume_drain_tile(a_buf, b_buf, as_buf, bs_buf, slot, acc,
                        DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr,
                        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
                        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
                        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                        BLOCK_K: gl.constexpr, BK_SCALE: gl.constexpr,
                        SUBTILE_K: gl.constexpr, SUBTILE_SCALE_K: gl.constexpr,
                        PRESHUFFLE_FACTOR: gl.constexpr, SCALE_KWIDTH: gl.constexpr):
    """Consume a BK=256 slot as two scaled-WMMA K=128 operations."""
    a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(0, SUBTILE_K, 1).load(layout=DOT_A)
    as0 = _load_scale(
        as_buf, slot, 0, 0, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE,
        BLOCK_M, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
        0, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
    bs0 = _load_scale(
        bs_buf, slot, 0, 0, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE,
        BLOCK_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    acc = gl.amd.gfx1250.wmma_scaled(a0, as0, DTYPE_A, b0, bs0, DTYPE_B, acc)

    a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
        SUBTILE_K, SUBTILE_K, 1).load(layout=DOT_A)
    as1 = _load_scale(
        as_buf, slot, 0, SUBTILE_SCALE_K, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE,
        BLOCK_M, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
        SUBTILE_K, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
    bs1 = _load_scale(
        bs_buf, slot, 0, SUBTILE_SCALE_K, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE,
        BLOCK_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    return gl.amd.gfx1250.wmma_scaled(a1, as1, DTYPE_A, b1, bs1, DTYPE_B, acc)


@gluon.jit
def _consume_pipelined_drain_tile(
        a_buf, b_buf, as_buf, bs_buf, slot, acc,
        DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        BLOCK_K: gl.constexpr, BK_SCALE: gl.constexpr,
        SUBTILE_K: gl.constexpr, SUBTILE_SCALE_K: gl.constexpr,
        PRESHUFFLE_FACTOR: gl.constexpr, SCALE_KWIDTH: gl.constexpr):
    """Consume a drain slot while preserving the load/WMMA warp pipeline."""
    with gl.amd.warp_pipeline_stage("stage0_tail", priority=0):
        a0 = a_buf.index(slot).slice(
            0, BLOCK_M, 0).slice(0, SUBTILE_K, 1).load(layout=DOT_A)
        as0 = _load_scale(
            as_buf, slot, 0, 0, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE,
            BLOCK_M, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
        b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            0, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
        bs0 = _load_scale(
            bs_buf, slot, 0, 0, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE,
            BLOCK_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("stage1_tail", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, DTYPE_A, b0, bs0, DTYPE_B, acc)

    with gl.amd.warp_pipeline_stage("stage0_tail", priority=0):
        a1 = a_buf.index(slot).slice(
            0, BLOCK_M, 0).slice(
                SUBTILE_K, SUBTILE_K, 1).load(layout=DOT_A)
        as1 = _load_scale(
            as_buf, slot, 0, SUBTILE_SCALE_K, SCALE_A_LAYOUT, BLOCK_M,
            BK_SCALE, BLOCK_M, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR,
            SCALE_KWIDTH)
        b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            SUBTILE_K, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
        bs1 = _load_scale(
            bs_buf, slot, 0, SUBTILE_SCALE_K, SCALE_B_LAYOUT, BLOCK_N,
            BK_SCALE, BLOCK_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR,
            SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("stage1_tail", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, DTYPE_A, b1, bs1, DTYPE_B, acc)
    return acc


@gluon.jit
def _consume_and_refill(a_buf, b_buf, as_buf, bs_buf, slot,
                        a_desc, b_desc, as_desc, bs_desc, refill_idx, acc,
                        DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr,
                        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
                        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
                        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                        BLOCK_K: gl.constexpr, BK_SCALE: gl.constexpr,
                        SUBTILE_K: gl.constexpr, SUBTILE_SCALE_K: gl.constexpr,
                        PRESHUFFLE_FACTOR: gl.constexpr,
                        BK_SCALE_PRESHUFFLED: gl.constexpr,
                        SCALE_KWIDTH: gl.constexpr,
                        CTA_M: gl.constexpr, CTA_N: gl.constexpr):
    """Consume a slot, synchronize its cluster users, then refill that slot."""
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            0, SUBTILE_K, 1).load(layout=DOT_A)
        as0 = _load_scale(
            as_buf, slot, 0, 0, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE,
            BLOCK_M, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
        b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            0, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
        bs0 = _load_scale(
            bs_buf, slot, 0, 0, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE,
            BLOCK_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(a0, as0, DTYPE_A, b0, bs0, DTYPE_B, acc)

    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            SUBTILE_K, SUBTILE_K, 1).load(layout=DOT_A)
        as1 = _load_scale(
            as_buf, slot, 0, SUBTILE_SCALE_K, SCALE_A_LAYOUT, BLOCK_M,
            BK_SCALE, BLOCK_M, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR,
            SCALE_KWIDTH)
        b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            SUBTILE_K, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
        bs1 = _load_scale(
            bs_buf, slot, 0, SUBTILE_SCALE_K, SCALE_B_LAYOUT, BLOCK_N,
            BK_SCALE, BLOCK_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR,
            SCALE_KWIDTH)
        if CTA_M * CTA_N > 1:
            gl.amd.gfx1250.cluster.arrive()

    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(a1, as1, DTYPE_A, b1, bs1, DTYPE_B, acc)
        if CTA_M * CTA_N > 1:
            gl.amd.gfx1250.cluster.wait()
        _issue_refill(
            a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
            refill_idx, slot, BLOCK_K, BK_SCALE_PRESHUFFLED)
    return acc


@gluon.jit
def mxfp8_cluster_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale,
        DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr,
        SCALE_BLOCK: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
        GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr,
        SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr,
        SHARED_SCALE_A: gl.constexpr, SHARED_SCALE_B: gl.constexpr,
        WMMA_LAYOUT: gl.constexpr, CTA_M: gl.constexpr, CTA_N: gl.constexpr,
        PIPELINE_TAIL: gl.constexpr, SPLIT_OUTPUT_STORE: gl.constexpr,
        QUARTER_OUTPUT_STORE: gl.constexpr,
        SINGLE_SLOT_PROLOGUE: gl.constexpr):
    gl.static_assert(CTA_M == CTA_N and (CTA_M == 1 or CTA_M == 2 or CTA_M == 4))
    gl.static_assert(BLOCK_M == CTA_M * 256 and BLOCK_N == CTA_N * 256)
    gl.static_assert(BLOCK_K == 256)
    gl.static_assert(SCALE_BLOCK == 16 or SCALE_BLOCK == 32)
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)

    pid_m, pid_n = _get_tile_ids(
        M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    subtile_k: gl.constexpr = BLOCK_K // 2
    bk_scale: gl.constexpr = BLOCK_K // SCALE_BLOCK
    subtile_scale_k: gl.constexpr = bk_scale // 2
    preshuffle_factor: gl.constexpr = 128
    bk_scale_preshuffled: gl.constexpr = bk_scale * preshuffle_factor
    scale_kwidth: gl.constexpr = min(4, bk_scale)
    nbuf: gl.constexpr = 2

    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [BLOCK_M, subtile_scale_k])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [BLOCK_N, subtile_scale_k])

    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [nbuf, BLOCK_M, BLOCK_K], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [nbuf, BLOCK_N, BLOCK_K], SHARED_LAYOUT_B)
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty,
        [nbuf, BLOCK_M // preshuffle_factor, bk_scale_preshuffled],
        SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty,
        [nbuf, BLOCK_N // preshuffle_factor, bk_scale_preshuffled],
        SHARED_SCALE_B)

    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am,
        shape=(M, K), strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, BLOCK_K), layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn,
        shape=(N, K), strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_N, BLOCK_K), layout=SHARED_LAYOUT_B)

    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + (pid_m * BLOCK_M) // preshuffle_factor * stride_scale,
        shape=(M // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1),
        block_shape=(BLOCK_M // preshuffle_factor, bk_scale_preshuffled),
        layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * BLOCK_N) // preshuffle_factor * stride_scale,
        shape=(N // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1),
        block_shape=(BLOCK_N // preshuffle_factor, bk_scale_preshuffled),
        layout=SHARED_SCALE_B)

    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max >= nbuf)
    wait_count: gl.constexpr = 2

    if SINGLE_SLOT_PROLOGUE:
        _issue_refill(
            a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
            0, 0, BLOCK_K, bk_scale_preshuffled)
        tdm.async_wait(0)
        if CTA_M * CTA_N > 1:
            _cluster_sync()

        _issue_refill(
            a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
            1, 1, BLOCK_K, bk_scale_preshuffled)
    else:
        for prefetch_idx in gl.static_range(nbuf):
            _issue_refill(
                a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
                prefetch_idx, prefetch_idx, BLOCK_K, bk_scale_preshuffled)

        tdm.async_wait(wait_count)
        if CTA_M * CTA_N > 1:
            _cluster_sync()

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    for tile_idx in range(0, iter_max - nbuf):
        slot = tile_idx % nbuf
        acc = _consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, slot, a_desc, b_desc, as_desc, bs_desc,
            tile_idx + nbuf, acc, DTYPE_A, DTYPE_B, dot_a, dot_b,
            scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N, BLOCK_K, bk_scale,
            subtile_k, subtile_scale_k, preshuffle_factor, bk_scale_preshuffled,
            scale_kwidth, CTA_M, CTA_N)
        tdm.async_wait(wait_count)

    penultimate_slot = (iter_max - 2) % nbuf
    if PIPELINE_TAIL:
        acc = _consume_pipelined_drain_tile(
            a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc,
            DTYPE_A, DTYPE_B, dot_a, dot_b, scale_a_layout, scale_b_layout,
            BLOCK_M, BLOCK_N, BLOCK_K, bk_scale, subtile_k,
            subtile_scale_k, preshuffle_factor, scale_kwidth)
    else:
        acc = _consume_drain_tile(
            a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc,
            DTYPE_A, DTYPE_B, dot_a, dot_b, scale_a_layout, scale_b_layout,
            BLOCK_M, BLOCK_N, BLOCK_K, bk_scale, subtile_k,
            subtile_scale_k, preshuffle_factor, scale_kwidth)

    tdm.async_wait(0)
    if CTA_M * CTA_N > 1:
        _cluster_sync()
    last_slot = (iter_max - 1) % nbuf
    if PIPELINE_TAIL:
        acc = _consume_pipelined_drain_tile(
            a_buf, b_buf, as_buf, bs_buf, last_slot, acc,
            DTYPE_A, DTYPE_B, dot_a, dot_b, scale_a_layout, scale_b_layout,
            BLOCK_M, BLOCK_N, BLOCK_K, bk_scale, subtile_k,
            subtile_scale_k, preshuffle_factor, scale_kwidth)
    else:
        acc = _consume_drain_tile(
            a_buf, b_buf, as_buf, bs_buf, last_slot, acc,
            DTYPE_A, DTYPE_B, dot_a, dot_b, scale_a_layout, scale_b_layout,
            BLOCK_M, BLOCK_N, BLOCK_K, bk_scale, subtile_k,
            subtile_scale_k, preshuffle_factor, scale_kwidth)

    if QUARTER_OUTPUT_STORE:
        _tdm_store_output_quarters(
            c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
            BLOCK_M, BLOCK_N, CTA_M, CTA_N)
    elif SPLIT_OUTPUT_STORE:
        _tdm_store_output_split_n2(
            c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
            BLOCK_M, BLOCK_N, CTA_M, CTA_N)
    else:
        _tdm_store_output(
            c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc, WMMA_LAYOUT,
            BLOCK_M, BLOCK_N, CTA_N)


def select_cluster_width(m, n):
    """Return the largest supported square cluster tile that divides M and N."""
    for width in SUPPORTED_CLUSTER_WIDTHS:
        tile = width * CTA_TILE
        if m % tile == 0 and n % tile == 0:
            return width
    raise ValueError("M and N must both be divisible by the 256x256 per-CTA tile")


def _pack_scale(scale, scale_kwidth):
    """Pack [nonK, Kscale] into Triton's blocked scale layout."""
    non_k, k_scale = scale.shape
    factor = 128
    scale = scale.view(
        non_k // factor, 4, factor // 4, k_scale // scale_kwidth, scale_kwidth)
    scale = scale.permute(0, 3, 2, 1, 4).contiguous()
    return scale.view(non_k // factor, k_scale * factor)


def _fill_trigonometric(tensor, cosine):
    flat = tensor.view(-1)
    chunk = 8 * 1024 * 1024
    for begin in range(0, flat.numel(), chunk):
        end = min(begin + chunk, flat.numel())
        angle = torch.arange(begin, end, dtype=torch.float64)
        values = torch.cos(angle) if cosine else torch.sin(angle)
        flat[begin:end].copy_(values.float())


def _make_inputs(args):
    torch.manual_seed(args.seed)
    if args.input_mode == "random":
        a = init_data(args.dtype_a, args.M, args.K)
        b = init_data(args.dtype_b, args.K, args.N)
    else:
        a = torch.empty((args.M, args.K), dtype=FP8_DTYPE_TO_TORCH[args.dtype_a])
        b = torch.empty((args.K, args.N), dtype=FP8_DTYPE_TO_TORCH[args.dtype_b])
        for output, cosine in ((a, False), (b, True)):
            _fill_trigonometric(output, cosine)

    scale_k = args.K // args.scale_block
    a_scale = MXScaleTensor(size=(args.M, scale_k)).random(low=1.0, high=32.0)
    b_scale = MXScaleTensor(size=(args.N, scale_k)).random(low=1.0, high=32.0)
    return a, b, a_scale, b_scale


def _validate_args(args):
    if args.M <= 0 or args.N <= 0 or args.K <= 0:
        raise ValueError("M, N, and K must be positive")
    cluster_width = select_cluster_width(args.M, args.N)
    if args.split_output_store and args.quarter_output_store:
        raise ValueError("select only one split-output mode")
    if (args.split_output_store or args.quarter_output_store) and cluster_width != 2:
        raise ValueError("split-output modes currently require a 2x2 cluster")
    if args.K % BLOCK_K:
        raise ValueError(f"K must be divisible by the BK={BLOCK_K} kernel tile")
    if args.K < NUM_BUFFERS * BLOCK_K:
        raise ValueError(f"K must contain at least {NUM_BUFFERS} BK={BLOCK_K} tiles")
    if args.single_slot_prologue and args.K < (NUM_BUFFERS + 1) * BLOCK_K:
        raise ValueError("--single-slot-prologue requires at least three K tiles")
    if args.scale_block not in (16, 32):
        raise ValueError("--scale-block must be 16 or 32")
    if args.K % args.scale_block:
        raise ValueError("K must be divisible by --scale-block")

    scale_k = args.K // args.scale_block
    scale_kwidth = min(4, BLOCK_K // args.scale_block)
    if scale_k % scale_kwidth:
        raise ValueError("K/scale_block must be divisible by the scale K-width")
    return cluster_width


def _make_case(args):
    cluster_width = _validate_args(args)
    block_m = block_n = cluster_width * CTA_TILE
    grid = (
        triton.cdiv(args.M, block_m) * triton.cdiv(args.N, block_n),
        1,
    )

    a, b, a_scale_obj, b_scale_obj = _make_inputs(args)
    reference = None
    output_dtype = torch.bfloat16 if args.fp8_bf16_output else torch.float16
    if args.check:
        reference = torch_gemm_mxfp(
            a, b, a_scale_obj, b_scale_obj, args.scale_block,
            args.M, args.N, args.K).to(output_dtype)

    scale_kwidth = min(4, BLOCK_K // args.scale_block)
    a_scale = _pack_scale(a_scale_obj.data, scale_kwidth)
    b_scale = _pack_scale(b_scale_obj.data, scale_kwidth)

    a_d = a.contiguous().cuda()
    b_d = b.T.contiguous().cuda()
    a_scale_d = a_scale.cuda()
    b_scale_d = b_scale.cuda()
    c_d = torch.empty((args.M, args.N), dtype=output_dtype, device="cuda")

    cga_layout = make_cga_layout(
        [cluster_width, cluster_width],
        [cluster_width, cluster_width],
        [0, 1],
    )
    shared_a, shared_b, shared_scale_a, shared_scale_b, wmma = _build_cluster_layouts(
        block_m, block_n, args.scale_block, cga_layout, args.split_output_store,
        args.quarter_output_store)

    def launch():
        return mxfp8_cluster_kernel_gfx1250[grid](
            a_d, b_d, c_d, a_scale_d, b_scale_d,
            args.M, args.N, args.K,
            a_d.stride(0), a_d.stride(1),
            b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1),
            b_scale_d.stride(0),
            FP8_DTYPE_TO_KERNEL[args.dtype_a],
            FP8_DTYPE_TO_KERNEL[args.dtype_b],
            args.scale_block, block_m, block_n, BLOCK_K, args.group_size_m,
            GRID_MN=grid[0], NUM_XCDS=args.num_xcds,
            SHARED_LAYOUT_A=shared_a, SHARED_LAYOUT_B=shared_b,
            SHARED_SCALE_A=shared_scale_a, SHARED_SCALE_B=shared_scale_b,
            WMMA_LAYOUT=wmma,
            CTA_M=cluster_width, CTA_N=cluster_width,
            PIPELINE_TAIL=args.pipeline_tail,
            SPLIT_OUTPUT_STORE=args.split_output_store,
            QUARTER_OUTPUT_STORE=args.quarter_output_store,
            SINGLE_SLOT_PROLOGUE=args.single_slot_prologue,
            num_warps=NUM_WARPS, num_ctas=cluster_width * cluster_width,
            waves_per_eu=NUM_WARPS // 4,
            llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ),
        )

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        if args.fp8_bf16_output:
            torch.testing.assert_close(c_d.cpu(), reference, rtol=1e-2, atol=5e-1)
        else:
            torch.testing.assert_close(c_d.cpu(), reference, rtol=1e-3, atol=1e-3)
        print("result verified", flush=True)

    return launch, check, cluster_width, block_m


def _event_probe(fn, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _print_benchmark_result(elapsed, total_iters, args):
    per_iter = elapsed / total_iters
    tflops = 2 * args.M * args.N * args.K / per_iter / 1e12
    print(f"total elapsed    : {elapsed:.6f} s")
    print(f"per-iter         : {per_iter * 1e6:.2f} us")
    print(f"TFLOPS           : {tflops:.3f}")


def _run_direct_benchmark(launch, args):
    if args.direct_iters <= 0:
        raise ValueError("--direct-iters must be positive")
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.direct_iters):
        launch()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print("benchmark mode   : direct synchronized CPU wall")
    print(f"warmup iters     : {args.warmup}")
    print(f"total iters      : {args.direct_iters}")
    _print_benchmark_result(elapsed, args.direct_iters, args)


class _CapturedGraph:

    def __init__(self, graph, keepalive):
        self._graph = graph
        self._keepalive = keepalive

    def replay(self):
        self._graph.replay()


def _capture_graph(fn, iterations):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(iterations):
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    # A CUDAGraph retains raw device addresses, but does not own the tensors
    # captured by fn. Keep fn (and its tensor closure) alive with the graph.
    return _CapturedGraph(graph, fn)


def _run_graph_benchmark(launch, args):
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()

    probe_ms = _event_probe(launch, args.probe_iters)
    iterations = args.iters_per_graph
    if iterations is None:
        iterations = max(1, int(args.graph_ms / max(probe_ms, 1e-6)))
    if iterations <= 0:
        raise ValueError("--iters-per-graph must be positive")

    graph = _capture_graph(launch, iterations)
    total_iters = args.n_replays * iterations
    start = time.perf_counter()
    for _ in range(args.n_replays):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"probe per-iter   : {probe_ms * 1e3:.2f} us")
    print(f"iters per graph  : {iterations}")
    print(f"replays          : {args.n_replays}")
    print(f"total iters      : {total_iters}")
    _print_benchmark_result(elapsed, total_iters, args)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="gfx1250 MXFP8 GEMM with automatic 1x1/2x2/4x4 cluster selection")
    parser.add_argument("-M", type=int, default=8192)
    parser.add_argument("-N", type=int, default=8192)
    parser.add_argument("-K", type=int, default=8192)
    parser.add_argument("--dtype-a", choices=FP8_DTYPE_TO_KERNEL, default="float8_e4m3")
    parser.add_argument("--dtype-b", choices=FP8_DTYPE_TO_KERNEL, default="float8_e4m3")
    parser.add_argument("--scale-block", type=int, choices=[16, 32], default=32)
    parser.add_argument(
        "--fp8-bf16-output", action="store_true",
        help="Store BF16 instead of FP16")
    parser.add_argument("--input-mode", choices=["random", "trig"], default="random")
    parser.add_argument("--group-size-m", type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument("--num-xcds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-mode", choices=["graph", "direct"], default="graph")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--direct-iters", type=int, default=1000)
    parser.add_argument("--probe-iters", type=int, default=20)
    parser.add_argument("--graph-ms", type=float, default=100.0)
    parser.add_argument("--n-replays", type=int, default=20)
    parser.add_argument("--iters-per-graph", type=int)
    parser.add_argument(
        "--pipeline-tail", action="store_true",
        help="warp-pipeline the final two prefetched K tiles")
    parser.add_argument(
        "--split-output-store", action="store_true",
        help="stage and TDM-store the 2x2 output tile in two N halves")
    parser.add_argument(
        "--quarter-output-store", action="store_true",
        help="stage and TDM-store the 2x2 output tile in four quarters")
    parser.add_argument(
        "--single-slot-prologue", action="store_true",
        help="load one K tile initially and overlap the second with first-tile compute")
    return parser


def main():
    args = _build_parser().parse_args()
    if not args.check and not args.benchmark:
        args.check = True

    launch, check, cluster_width, block_m = _make_case(args)
    print(
        f"shape             : {args.M}x{args.N}x{args.K}\n"
        f"cluster           : {cluster_width}x{cluster_width}\n"
        f"aggregate tile    : {block_m}x{block_m}x{BLOCK_K}\n"
        f"output dtype      : {'bfloat16' if args.fp8_bf16_output else 'float16'}\n"
        f"prologue          : {'single-slot' if args.single_slot_prologue else 'two-slot'}\n"
        f"output store      : "
        f"{'quarter TDM' if args.quarter_output_store else 'split-n2 TDM' if args.split_output_store else 'full-tile TDM'}"
    )

    if args.check:
        check()
    if args.benchmark:
        if args.benchmark_mode == "direct":
            _run_direct_benchmark(launch, args)
        else:
            _run_graph_benchmark(launch, args)


if __name__ == "__main__":
    main()
