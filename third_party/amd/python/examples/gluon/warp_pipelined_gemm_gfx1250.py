import argparse
import time

import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import async_copy as cp
from triton.experimental.gluon.language.amd.gfx1250 import tdm

try:
    from .mxfp_gemm_gfx1250 import (
        get_scale_blocked_layout,
        get_wmma_layout,
        MXFPGEMMConfig,
        MXScaleTensor,
        init_data,
        pack_scale,
        torch_gemm_mxfp,
    )
except ImportError:
    from mxfp_gemm_gfx1250 import (
        get_scale_blocked_layout,
        get_wmma_layout,
        MXFPGEMMConfig,
        MXScaleTensor,
        init_data,
        pack_scale,
        torch_gemm_mxfp,
    )


MXFP_DTYPE_TO_KERNEL = {
    "float8_e5m2": "e5m2",
    "float8_e4m3": "e4m3",
    "float4": "e2m1",
}


@gluon.jit
def _remap_xcd_chunked(pid, grid_mn, NUM_XCDS: gl.constexpr, CHUNK_SIZE: gl.constexpr):
    if pid >= (grid_mn // (NUM_XCDS * CHUNK_SIZE)) * (NUM_XCDS * CHUNK_SIZE):
        return pid
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    return (local_pid // CHUNK_SIZE) * NUM_XCDS * CHUNK_SIZE + xcd * CHUNK_SIZE + (local_pid % CHUNK_SIZE)


@gluon.jit
def get_xcd_swizzled_pids(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, GRID_MN: gl.constexpr,
                          NUM_XCDS: gl.constexpr, GROUP_SIZE_M: gl.constexpr):
    pid = _remap_xcd_chunked(gl.program_id(axis=0), GRID_MN, NUM_XCDS, 2)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = gl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@gluon.jit
def _store_quadrants(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc_tl, acc_bl, acc_tr, acc_br,
                     STORE_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    offs_m = gl.arange(0, BLOCK_M // 2, layout=gl.SliceLayout(1, STORE_LAYOUT))
    offs_n = gl.arange(0, BLOCK_N // 2, layout=gl.SliceLayout(0, STORE_LAYOUT))
    quad_offsets = stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    mask_tl = (pid_m * BLOCK_M + offs_m[:, None] < M) & (pid_n * BLOCK_N + offs_n[None, :] < N)
    mask_bl = (pid_m * BLOCK_M + BLOCK_M // 2 + offs_m[:, None] < M) & (pid_n * BLOCK_N + offs_n[None, :] < N)
    mask_tr = (pid_m * BLOCK_M + offs_m[:, None] < M) & (pid_n * BLOCK_N + BLOCK_N // 2 + offs_n[None, :] < N)
    mask_br = (pid_m * BLOCK_M + BLOCK_M // 2 + offs_m[:, None] < M) & (
        pid_n * BLOCK_N + BLOCK_N // 2 + offs_n[None, :] < N)

    gl.amd.gfx1250.buffer_store(gl.convert_layout(acc_tl.to(c_ptr.type.element_ty), STORE_LAYOUT), base, quad_offsets,
                                mask=mask_tl)
    gl.amd.gfx1250.buffer_store(gl.convert_layout(acc_bl.to(c_ptr.type.element_ty), STORE_LAYOUT),
                                base + (BLOCK_M // 2) * stride_cm, quad_offsets, mask=mask_bl)
    gl.amd.gfx1250.buffer_store(gl.convert_layout(acc_tr.to(c_ptr.type.element_ty), STORE_LAYOUT),
                                base + (BLOCK_N // 2) * stride_cn, quad_offsets, mask=mask_tr)
    gl.amd.gfx1250.buffer_store(gl.convert_layout(acc_br.to(c_ptr.type.element_ty), STORE_LAYOUT),
                                base + (BLOCK_M // 2) * stride_cm + (BLOCK_N // 2) * stride_cn, quad_offsets,
                                mask=mask_br)


@gluon.jit
def _load_mxfp_scale(scale_buffer, idx, layout: gl.constexpr, BLOCK_NONK: gl.constexpr, BK_SCALE: gl.constexpr,
                     SCALE_PRESHUFFLE: gl.constexpr, PRESHUFFLE_FACTOR: gl.constexpr,
                     SCALE_KWIDTH: gl.constexpr):
    scale_slice = scale_buffer.index(idx)
    if SCALE_PRESHUFFLE:
        scale_slice = scale_slice.reshape((BLOCK_NONK // PRESHUFFLE_FACTOR, BK_SCALE // SCALE_KWIDTH,
                                           PRESHUFFLE_FACTOR // 4, 4, SCALE_KWIDTH)).permute(
                                               (0, 3, 2, 1, 4)).reshape((BLOCK_NONK, BK_SCALE))
    return scale_slice.load(layout=layout)


@gluon.jit
def _wait_mxfp_scale_pipeline(waitcnt: int, ASYNC_COPY_SCALE: gl.constexpr):
    if ASYNC_COPY_SCALE:
        tdm.async_wait(0)
        cp.wait_group(0)
    else:
        tdm.async_wait(waitcnt)


@gluon.jit
def _issue_mxfp_scale_load(scale_desc, scale_ptrs, load_k, scale_buffer, slot, BK_SCALE_PRESHUFFLED: gl.constexpr,
                           ASYNC_COPY_SCALE: gl.constexpr):
    if ASYNC_COPY_SCALE:
        cp.global_to_shared(scale_buffer.index(slot), scale_ptrs + load_k * BK_SCALE_PRESHUFFLED)
        cp.commit_group()
    else:
        tdm.async_load(scale_desc, [0, load_k * BK_SCALE_PRESHUFFLED], scale_buffer.index(slot))


@gluon.jit
def f16_slice_mn_warp_pipeline_kernel_gfx1250(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk,
                                             stride_bn, stride_cm, stride_cn, BLOCK_M: gl.constexpr,
                                             BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GRID_MN: gl.constexpr,
                                             NUM_XCDS: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
                                             WARP_BASES: gl.constexpr, NUM_BUFFERS: gl.constexpr,
                                             RESOLVE_PARTITION_CONFLICTS: gl.constexpr, NUM_WARPS: gl.constexpr):
    gl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 64)
    gl.static_assert(NUM_BUFFERS >= 2, "sliceMN requires at least two LDS buffers")
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    padded_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M // 2, BLOCK_K], [1, 0])
    padded_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_N // 2, BLOCK_K], [1, 0])
    if RESOLVE_PARTITION_CONFLICTS:
        layouts: gl.constexpr = gl.amd.gfx1250.make_partitioned_dot_layouts(
            BLOCK_M // 2, BLOCK_N // 2, padded_a, padded_b, NUM_WARPS, [16, 16, 32], a_transposed=False,
            b_transposed=True)
        shared_a: gl.constexpr = layouts[0]
        shared_b: gl.constexpr = layouts[1]
        wmma_layout: gl.constexpr = layouts[2]
    else:
        shared_a: gl.constexpr = padded_a
        shared_b: gl.constexpr = padded_b
        wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma_layout, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma_layout, 8)
    store_layout: gl.constexpr = wmma_layout

    nbuf: gl.constexpr = NUM_BUFFERS
    a_top_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, BLOCK_M // 2, BLOCK_K], shared_a)
    a_bot_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, BLOCK_M // 2, BLOCK_K], shared_a)
    b_left_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, BLOCK_N // 2, BLOCK_K], shared_b)
    b_right_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, BLOCK_N // 2, BLOCK_K], shared_b)

    a_base = pid_m * BLOCK_M * stride_am
    b_base = pid_n * BLOCK_N * stride_bn
    a_top_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base, shape=(M, K), strides=(stride_am, stride_ak),
                                            block_shape=(BLOCK_M // 2, BLOCK_K), layout=shared_a)
    a_bot_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base + (BLOCK_M // 2) * stride_am, shape=(M, K),
                                            strides=(stride_am, stride_ak), block_shape=(BLOCK_M // 2, BLOCK_K),
                                            layout=shared_a)
    b_left_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base, shape=(N, K), strides=(stride_bn, stride_bk),
                                             block_shape=(BLOCK_N // 2, BLOCK_K), layout=shared_b)
    b_right_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base + (BLOCK_N // 2) * stride_bn, shape=(N, K),
                                              strides=(stride_bn, stride_bk), block_shape=(BLOCK_N // 2, BLOCK_K),
                                              layout=shared_b)

    for i in gl.static_range(NUM_BUFFERS - 1):
        tdm.async_load(b_left_desc, [0, i * BLOCK_K], b_left_buf.index(i))
        tdm.async_load(a_top_desc, [0, i * BLOCK_K], a_top_buf.index(i))
        tdm.async_load(a_bot_desc, [0, i * BLOCK_K], a_bot_buf.index(i))
        tdm.async_load(b_right_desc, [0, i * BLOCK_K], b_right_buf.index(i))

    prefetch_wait: gl.constexpr = 4 * (NUM_BUFFERS - 1) - 2
    steady_wait: gl.constexpr = 4 * (NUM_BUFFERS - 1) - 3
    tdm.async_wait(prefetch_wait)
    a_top = a_top_buf.index(0).load(layout=dot_a)
    b_left = b_left_buf.index(0).permute([1, 0]).load(layout=dot_b)

    acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), dtype=gl.float32, layout=wmma_layout)
    acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), dtype=gl.float32, layout=wmma_layout)
    acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), dtype=gl.float32, layout=wmma_layout)
    acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), dtype=gl.float32, layout=wmma_layout)

    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max > 3)
    consume_k = 0
    load_k = NUM_BUFFERS - 1
    for _ in range(0, iter_max - (NUM_BUFFERS - 1)):
        read_slot = consume_k % nbuf
        next_slot = (consume_k + 1) % nbuf
        write_slot = load_k % nbuf
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.gfx1250.wmma(a_top, b_left, acc_tl)
        tdm.async_wait(steady_wait)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = a_bot_buf.index(read_slot).load(layout=dot_a)
            tdm.async_load(b_left_desc, [0, load_k * BLOCK_K], b_left_buf.index(write_slot))

        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.gfx1250.wmma(a_bot, b_left, acc_bl)
        tdm.async_wait(steady_wait)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = b_right_buf.index(read_slot).permute([1, 0]).load(layout=dot_b)
            tdm.async_load(a_top_desc, [0, load_k * BLOCK_K], a_top_buf.index(write_slot))

        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.gfx1250.wmma(a_top, b_right, acc_tr)
        tdm.async_wait(steady_wait)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = b_left_buf.index(next_slot).permute([1, 0]).load(layout=dot_b)
            tdm.async_load(a_bot_desc, [0, load_k * BLOCK_K], a_bot_buf.index(write_slot))

        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.gfx1250.wmma(a_bot, b_right, acc_br)
        tdm.async_wait(steady_wait)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = a_top_buf.index(next_slot).load(layout=dot_a)
            tdm.async_load(b_right_desc, [0, load_k * BLOCK_K], b_right_buf.index(write_slot))
        consume_k += 1
        load_k += 1

    for i in gl.static_range(NUM_BUFFERS - 1):
        read_slot = (iter_max - (NUM_BUFFERS - 1 - i)) % nbuf
        acc_tl = gl.amd.gfx1250.wmma(a_top, b_left, acc_tl)
        tdm.async_wait(4 * (NUM_BUFFERS - 1 - i) - 3)
        a_bot = a_bot_buf.index(read_slot).load(layout=dot_a)
        acc_bl = gl.amd.gfx1250.wmma(a_bot, b_left, acc_bl)
        tdm.async_wait(4 * (NUM_BUFFERS - 1 - i) - 4)
        b_right = b_right_buf.index(read_slot).permute([1, 0]).load(layout=dot_b)
        acc_tr = gl.amd.gfx1250.wmma(a_top, b_right, acc_tr)
        if i < NUM_BUFFERS - 2:
            next_slot = (iter_max - (NUM_BUFFERS - 1 - i) + 1) % nbuf
            tdm.async_wait(4 * (NUM_BUFFERS - 2 - i) - 1)
            b_left = b_left_buf.index(next_slot).permute([1, 0]).load(layout=dot_b)
            acc_br = gl.amd.gfx1250.wmma(a_bot, b_right, acc_br)
            tdm.async_wait(4 * (NUM_BUFFERS - 2 - i) - 2)
            a_top = a_top_buf.index(next_slot).load(layout=dot_a)
        else:
            acc_br = gl.amd.gfx1250.wmma(a_bot, b_right, acc_br)

    _store_quadrants(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc_tl, acc_bl, acc_tr, acc_br, store_layout,
                     BLOCK_M, BLOCK_N)


@gluon.jit
def mxfp_slice_mn_warp_pipeline_kernel_gfx1250(a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am,
                                              stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scale,
                                              DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr,
                                              SCALE_BLOCK: gl.constexpr, BLOCK_M: gl.constexpr,
                                              BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                                              GROUP_SIZE_M: gl.constexpr, GRID_MN: gl.constexpr,
                                              NUM_XCDS: gl.constexpr, TRANSPOSE_B: gl.constexpr,
                                              WITH_A_SCALE: gl.constexpr, NUM_WARPS: gl.constexpr,
                                              NUM_BUFFERS: gl.constexpr,
                                              RESOLVE_PARTITION_CONFLICTS: gl.constexpr,
                                              SCALE_PRESHUFFLE: gl.constexpr, ASYNC_COPY_SCALE: gl.constexpr):
    gl.static_assert(TRANSPOSE_B)
    gl.static_assert(NUM_BUFFERS >= 2, "sliceMN requires at least two LDS buffers")
    cfg: gl.constexpr = MXFPGEMMConfig(BLOCK_M, BLOCK_N, BLOCK_K, DTYPE_A, DTYPE_B, SCALE_BLOCK, NUM_BUFFERS,
                                       TRANSPOSE_B, WITH_A_SCALE, SCALE_PRESHUFFLE, NUM_WARPS, ASYNC_COPY_SCALE,
                                       (2, 2, 1), -1, "", RESOLVE_PARTITION_CONFLICTS)
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    HALF_M: gl.constexpr = BLOCK_M // 2
    HALF_N: gl.constexpr = BLOCK_N // 2
    BK_A: gl.constexpr = BLOCK_K // cfg.DIV_FACTOR_A
    BK_B: gl.constexpr = BLOCK_K // cfg.DIV_FACTOR_B
    BK_SCALE: gl.constexpr = BLOCK_K // SCALE_BLOCK
    PRESHUFFLE_FACTOR: gl.constexpr = 128 if SCALE_PRESHUFFLE else 1
    BK_SCALE_PRESHUFFLED: gl.constexpr = BK_SCALE * PRESHUFFLE_FACTOR
    HALF_M_PRESHUFFLED: gl.constexpr = HALF_M // PRESHUFFLE_FACTOR
    HALF_N_PRESHUFFLED: gl.constexpr = HALF_N // PRESHUFFLE_FACTOR
    SCALE_KWIDTH: gl.constexpr = 4 if BK_SCALE >= 4 else BK_SCALE

    padded_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BK_A if BK_A >= 256 else 256, 16]],
                                                                     [HALF_M, BK_A], [1, 0])
    padded_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BK_B if BK_B >= 256 else 256, 16]],
                                                                     [HALF_N, BK_B], [1, 0])
    shared_scale: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[256, 8]],
                                                                        [HALF_M_PRESHUFFLED,
                                                                         BK_SCALE_PRESHUFFLED], [1, 0])
    INSTR_M: gl.constexpr = 32 if (DTYPE_A == "e2m1" and DTYPE_B == "e2m1") else 16
    if RESOLVE_PARTITION_CONFLICTS:
        layouts: gl.constexpr = gl.amd.gfx1250.make_partitioned_dot_layouts(
            HALF_M, HALF_N, padded_a, padded_b, NUM_WARPS, [INSTR_M, 16, 128], a_transposed=False, b_transposed=True)
        shared_a: gl.constexpr = layouts[0]
        shared_b: gl.constexpr = layouts[1]
        wmma: gl.constexpr = layouts[2]
        wmma_packed: gl.constexpr = gl.amd.AMDWMMALayout(3, True, wmma.warp_bases, wmma.reg_bases, [INSTR_M, 16, 64])
    else:
        shared_a: gl.constexpr = padded_a
        shared_b: gl.constexpr = padded_b
        wmma: gl.constexpr = get_wmma_layout(NUM_WARPS, False, SCALE_PRESHUFFLE, INSTR_M)
        wmma_packed: gl.constexpr = get_wmma_layout(NUM_WARPS, True, SCALE_PRESHUFFLE, INSTR_M)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma_packed if DTYPE_A == "e2m1" else wmma, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma_packed if DTYPE_B == "e2m1" else wmma, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_a, [HALF_M, BK_SCALE])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_b, [HALF_N, BK_SCALE])
    store_layout: gl.constexpr = wmma

    nbuf: gl.constexpr = NUM_BUFFERS
    a_top_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, HALF_M, BK_A], shared_a)
    a_bot_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, HALF_M, BK_A], shared_a)
    b_left_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, HALF_N, BK_B], shared_b)
    b_right_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, HALF_N, BK_B], shared_b)
    if WITH_A_SCALE:
        as_top_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty,
                                               [nbuf, HALF_M_PRESHUFFLED, BK_SCALE_PRESHUFFLED], shared_scale)
        as_bot_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty,
                                               [nbuf, HALF_M_PRESHUFFLED, BK_SCALE_PRESHUFFLED], shared_scale)
    else:
        as_top_buf = gl.constexpr(0)
        as_bot_buf = gl.constexpr(0)
    bs_left_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty,
                                            [nbuf, HALF_N_PRESHUFFLED, BK_SCALE_PRESHUFFLED], shared_scale)
    bs_right_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty,
                                             [nbuf, HALF_N_PRESHUFFLED, BK_SCALE_PRESHUFFLED], shared_scale)

    a_base = pid_m * BLOCK_M * stride_am
    b_base = pid_n * BLOCK_N * stride_bn
    a_top_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base, shape=(M, K // cfg.DIV_FACTOR_A),
                                            strides=(stride_am, stride_ak), block_shape=(HALF_M, BK_A),
                                            layout=shared_a)
    a_bot_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base + HALF_M * stride_am,
                                            shape=(M, K // cfg.DIV_FACTOR_A), strides=(stride_am, stride_ak),
                                            block_shape=(HALF_M, BK_A), layout=shared_a)
    b_left_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base, shape=(N, K // cfg.DIV_FACTOR_B),
                                             strides=(stride_bn, stride_bk), block_shape=(HALF_N, BK_B),
                                             layout=shared_b)
    b_right_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base + HALF_N * stride_bn,
                                              shape=(N, K // cfg.DIV_FACTOR_B), strides=(stride_bn, stride_bk),
                                              block_shape=(HALF_N, BK_B), layout=shared_b)

    if WITH_A_SCALE:
        as_top_base = (pid_m * BLOCK_M) // PRESHUFFLE_FACTOR * stride_scale
        as_bot_base = (pid_m * BLOCK_M + HALF_M) // PRESHUFFLE_FACTOR * stride_scale
        as_top_desc = tdm.make_tensor_descriptor(base=a_scale_ptr + as_top_base,
                                                 shape=(M // PRESHUFFLE_FACTOR,
                                                        K // SCALE_BLOCK * PRESHUFFLE_FACTOR),
                                                 strides=(stride_scale, 1),
                                                 block_shape=(HALF_M_PRESHUFFLED, BK_SCALE_PRESHUFFLED),
                                                 layout=shared_scale)
        as_bot_desc = tdm.make_tensor_descriptor(base=a_scale_ptr + as_bot_base,
                                                 shape=(M // PRESHUFFLE_FACTOR,
                                                        K // SCALE_BLOCK * PRESHUFFLE_FACTOR),
                                                 strides=(stride_scale, 1),
                                                 block_shape=(HALF_M_PRESHUFFLED, BK_SCALE_PRESHUFFLED),
                                                 layout=shared_scale)
    else:
        as_top_desc = gl.constexpr(0)
        as_bot_desc = gl.constexpr(0)
    bs_left_base = (pid_n * BLOCK_N) // PRESHUFFLE_FACTOR * stride_scale
    bs_right_base = (pid_n * BLOCK_N + HALF_N) // PRESHUFFLE_FACTOR * stride_scale
    bs_left_desc = tdm.make_tensor_descriptor(base=b_scale_ptr + bs_left_base,
                                              shape=(N // PRESHUFFLE_FACTOR,
                                                     K // SCALE_BLOCK * PRESHUFFLE_FACTOR),
                                              strides=(stride_scale, 1),
                                              block_shape=(HALF_N_PRESHUFFLED, BK_SCALE_PRESHUFFLED),
                                              layout=shared_scale)
    bs_right_desc = tdm.make_tensor_descriptor(base=b_scale_ptr + bs_right_base,
                                               shape=(N // PRESHUFFLE_FACTOR,
                                                      K // SCALE_BLOCK * PRESHUFFLE_FACTOR),
                                               strides=(stride_scale, 1),
                                               block_shape=(HALF_N_PRESHUFFLED, BK_SCALE_PRESHUFFLED),
                                               layout=shared_scale)

    scale_copy_layout: gl.constexpr = get_scale_blocked_layout(NUM_WARPS)
    scale_m = gl.arange(0, HALF_M_PRESHUFFLED, layout=gl.SliceLayout(1, scale_copy_layout))
    scale_n = gl.arange(0, HALF_N_PRESHUFFLED, layout=gl.SliceLayout(1, scale_copy_layout))
    scale_k = gl.arange(0, BK_SCALE_PRESHUFFLED, layout=gl.SliceLayout(0, scale_copy_layout))
    if WITH_A_SCALE:
        as_top_ptrs = a_scale_ptr + as_top_base + scale_m[:, None] * stride_scale + scale_k[None, :]
        as_bot_ptrs = a_scale_ptr + as_bot_base + scale_m[:, None] * stride_scale + scale_k[None, :]
    else:
        as_top_ptrs = gl.constexpr(0)
        as_bot_ptrs = gl.constexpr(0)
    bs_left_ptrs = b_scale_ptr + bs_left_base + scale_n[:, None] * stride_scale + scale_k[None, :]
    bs_right_ptrs = b_scale_ptr + bs_right_base + scale_n[:, None] * stride_scale + scale_k[None, :]

    for i in gl.static_range(NUM_BUFFERS - 1):
        tdm.async_load(b_left_desc, [0, i * BK_B], b_left_buf.index(i))
        if ASYNC_COPY_SCALE:
            cp.global_to_shared(bs_left_buf.index(i), bs_left_ptrs + i * BK_SCALE_PRESHUFFLED)
            cp.commit_group()
        else:
            tdm.async_load(bs_left_desc, [0, i * BK_SCALE_PRESHUFFLED], bs_left_buf.index(i))
        tdm.async_load(a_top_desc, [0, i * BK_A], a_top_buf.index(i))
        if WITH_A_SCALE:
            if ASYNC_COPY_SCALE:
                cp.global_to_shared(as_top_buf.index(i), as_top_ptrs + i * BK_SCALE_PRESHUFFLED)
                cp.commit_group()
            else:
                tdm.async_load(as_top_desc, [0, i * BK_SCALE_PRESHUFFLED], as_top_buf.index(i))
        tdm.async_load(a_bot_desc, [0, i * BK_A], a_bot_buf.index(i))
        if WITH_A_SCALE:
            if ASYNC_COPY_SCALE:
                cp.global_to_shared(as_bot_buf.index(i), as_bot_ptrs + i * BK_SCALE_PRESHUFFLED)
                cp.commit_group()
            else:
                tdm.async_load(as_bot_desc, [0, i * BK_SCALE_PRESHUFFLED], as_bot_buf.index(i))
        tdm.async_load(b_right_desc, [0, i * BK_B], b_right_buf.index(i))
        if ASYNC_COPY_SCALE:
            cp.global_to_shared(bs_right_buf.index(i), bs_right_ptrs + i * BK_SCALE_PRESHUFFLED)
            cp.commit_group()
        else:
            tdm.async_load(bs_right_desc, [0, i * BK_SCALE_PRESHUFFLED], bs_right_buf.index(i))

    wait_unit: gl.constexpr = 8 if WITH_A_SCALE else 6
    initial_needed: gl.constexpr = 4 if WITH_A_SCALE else 3
    prefetch_wait: gl.constexpr = wait_unit - initial_needed if NUM_BUFFERS == 2 else (NUM_BUFFERS - 2) * wait_unit - 2
    steady_wait: gl.constexpr = wait_unit - 3 if NUM_BUFFERS == 2 else (NUM_BUFFERS - 2) * wait_unit - 3
    _wait_mxfp_scale_pipeline(prefetch_wait, ASYNC_COPY_SCALE)
    a_top = a_top_buf.index(0).load(layout=dot_a)
    if WITH_A_SCALE:
        as_top = _load_mxfp_scale(as_top_buf, 0, scale_a_layout, HALF_M, BK_SCALE, SCALE_PRESHUFFLE,
                                  PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    else:
        as_top = 0
        as_top = as_top.to(gl.uint8)
    b_left = b_left_buf.index(0).permute([1, 0]).load(layout=dot_b)
    bs_left = _load_mxfp_scale(bs_left_buf, 0, scale_b_layout, HALF_N, BK_SCALE, SCALE_PRESHUFFLE,
                               PRESHUFFLE_FACTOR, SCALE_KWIDTH)

    acc_tl = gl.zeros((HALF_M, HALF_N), dtype=gl.float32, layout=wmma)
    acc_bl = gl.zeros((HALF_M, HALF_N), dtype=gl.float32, layout=wmma)
    acc_tr = gl.zeros((HALF_M, HALF_N), dtype=gl.float32, layout=wmma)
    acc_br = gl.zeros((HALF_M, HALF_N), dtype=gl.float32, layout=wmma)

    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max > 3)
    consume_k = 0
    load_k = NUM_BUFFERS - 1
    for _ in range(0, iter_max - (NUM_BUFFERS - 1)):
        read_slot = consume_k % nbuf
        next_slot = (consume_k + 1) % nbuf
        write_slot = load_k % nbuf
        acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_left, bs_left, DTYPE_B, acc_tl)
        _wait_mxfp_scale_pipeline(steady_wait, ASYNC_COPY_SCALE)
        a_bot = a_bot_buf.index(read_slot).load(layout=dot_a)
        if WITH_A_SCALE:
            as_bot = _load_mxfp_scale(as_bot_buf, read_slot, scale_a_layout, HALF_M, BK_SCALE, SCALE_PRESHUFFLE,
                                      PRESHUFFLE_FACTOR, SCALE_KWIDTH)
        else:
            as_bot = as_top
        tdm.async_load(b_left_desc, [0, load_k * BK_B], b_left_buf.index(write_slot))
        _issue_mxfp_scale_load(bs_left_desc, bs_left_ptrs, load_k, bs_left_buf, write_slot, BK_SCALE_PRESHUFFLED,
                               ASYNC_COPY_SCALE)

        acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_left, bs_left, DTYPE_B, acc_bl)
        _wait_mxfp_scale_pipeline(steady_wait, ASYNC_COPY_SCALE)
        b_right = b_right_buf.index(read_slot).permute([1, 0]).load(layout=dot_b)
        bs_right = _load_mxfp_scale(bs_right_buf, read_slot, scale_b_layout, HALF_N, BK_SCALE, SCALE_PRESHUFFLE,
                                    PRESHUFFLE_FACTOR, SCALE_KWIDTH)
        tdm.async_load(a_top_desc, [0, load_k * BK_A], a_top_buf.index(write_slot))
        if WITH_A_SCALE:
            _issue_mxfp_scale_load(as_top_desc, as_top_ptrs, load_k, as_top_buf, write_slot, BK_SCALE_PRESHUFFLED,
                                   ASYNC_COPY_SCALE)

        acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_right, bs_right, DTYPE_B, acc_tr)
        _wait_mxfp_scale_pipeline(steady_wait, ASYNC_COPY_SCALE)
        b_left = b_left_buf.index(next_slot).permute([1, 0]).load(layout=dot_b)
        bs_left = _load_mxfp_scale(bs_left_buf, next_slot, scale_b_layout, HALF_N, BK_SCALE, SCALE_PRESHUFFLE,
                                   PRESHUFFLE_FACTOR, SCALE_KWIDTH)
        tdm.async_load(a_bot_desc, [0, load_k * BK_A], a_bot_buf.index(write_slot))
        if WITH_A_SCALE:
            _issue_mxfp_scale_load(as_bot_desc, as_bot_ptrs, load_k, as_bot_buf, write_slot, BK_SCALE_PRESHUFFLED,
                                   ASYNC_COPY_SCALE)

        acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_right, bs_right, DTYPE_B, acc_br)
        _wait_mxfp_scale_pipeline(steady_wait, ASYNC_COPY_SCALE)
        a_top = a_top_buf.index(next_slot).load(layout=dot_a)
        if WITH_A_SCALE:
            as_top = _load_mxfp_scale(as_top_buf, next_slot, scale_a_layout, HALF_M, BK_SCALE, SCALE_PRESHUFFLE,
                                      PRESHUFFLE_FACTOR, SCALE_KWIDTH)
        tdm.async_load(b_right_desc, [0, load_k * BK_B], b_right_buf.index(write_slot))
        _issue_mxfp_scale_load(bs_right_desc, bs_right_ptrs, load_k, bs_right_buf, write_slot, BK_SCALE_PRESHUFFLED,
                               ASYNC_COPY_SCALE)
        consume_k += 1
        load_k += 1

    # Drain the prefetched K tiles using the same quadrant order.
    for i in gl.static_range(NUM_BUFFERS - 1):
        read_slot = (iter_max - (NUM_BUFFERS - 1 - i)) % nbuf
        acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_left, bs_left, DTYPE_B, acc_tl)
        if i < NUM_BUFFERS - 2:
            _wait_mxfp_scale_pipeline((NUM_BUFFERS - 2 - i) * wait_unit - 3, ASYNC_COPY_SCALE)
        else:
            _wait_mxfp_scale_pipeline(1, ASYNC_COPY_SCALE)
        a_bot = a_bot_buf.index(read_slot).load(layout=dot_a)
        if WITH_A_SCALE:
            as_bot = _load_mxfp_scale(as_bot_buf, read_slot, scale_a_layout, HALF_M, BK_SCALE, SCALE_PRESHUFFLE,
                                      PRESHUFFLE_FACTOR, SCALE_KWIDTH)
        else:
            as_bot = as_top
        acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_left, bs_left, DTYPE_B, acc_bl)
        if i < NUM_BUFFERS - 2:
            _wait_mxfp_scale_pipeline((NUM_BUFFERS - 2 - i) * wait_unit - 4, ASYNC_COPY_SCALE)
        else:
            _wait_mxfp_scale_pipeline(0, ASYNC_COPY_SCALE)
        b_right = b_right_buf.index(read_slot).permute([1, 0]).load(layout=dot_b)
        bs_right = _load_mxfp_scale(bs_right_buf, read_slot, scale_b_layout, HALF_N, BK_SCALE, SCALE_PRESHUFFLE,
                                    PRESHUFFLE_FACTOR, SCALE_KWIDTH)
        acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_right, bs_right, DTYPE_B, acc_tr)
        if i < NUM_BUFFERS - 2:
            next_slot = (iter_max - (NUM_BUFFERS - 1 - i) + 1) % nbuf
            _wait_mxfp_scale_pipeline((NUM_BUFFERS - 2 - i) * wait_unit - 5, ASYNC_COPY_SCALE)
            b_left = b_left_buf.index(next_slot).permute([1, 0]).load(layout=dot_b)
            bs_left = _load_mxfp_scale(bs_left_buf, next_slot, scale_b_layout, HALF_N, BK_SCALE, SCALE_PRESHUFFLE,
                                       PRESHUFFLE_FACTOR, SCALE_KWIDTH)
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_right, bs_right, DTYPE_B, acc_br)
            _wait_mxfp_scale_pipeline((NUM_BUFFERS - 2 - i) * wait_unit - 6, ASYNC_COPY_SCALE)
            a_top = a_top_buf.index(next_slot).load(layout=dot_a)
            if WITH_A_SCALE:
                as_top = _load_mxfp_scale(as_top_buf, next_slot, scale_a_layout, HALF_M, BK_SCALE, SCALE_PRESHUFFLE,
                                          PRESHUFFLE_FACTOR, SCALE_KWIDTH)
        else:
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_right, bs_right, DTYPE_B, acc_br)

    _store_quadrants(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc_tl, acc_bl, acc_tr, acc_br, store_layout,
                     BLOCK_M, BLOCK_N)


def _event_probe(fn, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _capture_graph(fn, n_per_graph):
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


def _run_benchmark(launch, M, N, K, warmup, probe_iters, graph_ms, n_replays, iters_per_graph):
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    probe_ms = _event_probe(launch, probe_iters)
    if iters_per_graph is None:
        n_per_graph = max(1, int(graph_ms / max(probe_ms, 1e-6)))
    else:
        n_per_graph = iters_per_graph
    if n_per_graph <= 0:
        raise ValueError("--iters-per-graph must be positive")

    graph = _capture_graph(launch, n_per_graph)
    total_iters = n_replays * n_per_graph

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_replays):
        graph.replay()
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0

    per_iter_s = elapsed_s / total_iters
    tflops = (2 * M * N * K) / per_iter_s / 1e12

    print(f"probe per-iter   : {probe_ms * 1e3:.2f} us")
    print(f"iters per graph  : {n_per_graph}")
    print(f"replays          : {n_replays}")
    print(f"total iters      : {total_iters}")
    print()
    print(f"total elapsed    : {elapsed_s:.6f} s")
    print(f"per-iter         : {per_iter_s * 1e6:.2f} us")
    print(f"TFLOPS           : {tflops:.3f}")


def _make_f16_case(args):
    if args.dtype_b != "float16":
        raise ValueError("The 16-bit path requires --dtype-a float16 --dtype-b float16")
    if args.scale_preshuffled or args.async_copy_scale:
        raise ValueError("--scale-preshuffled and --async-copy-scale apply only to the MXFP path")
    if not args.transpose_b:
        raise ValueError("The sliceMN fp16 kernel expects --transpose-b so K is contiguous in B")
    if (args.BM, args.BN, args.BK) != (256, 256, 64):
        raise ValueError("The fp16 sliceMN port currently expects -BM 256 -BN 256 -BK 64")
    if args.num_warps != 8:
        raise ValueError("The fp16 inter-wave sliceMN port expects --num-warps 8")
    if args.num_buffers < 2:
        raise ValueError("--num-buffers must be at least 2")
    if triton.cdiv(args.K, args.BK) <= 3:
        raise ValueError("K/BLOCK_K must be greater than 3 for the 2x-unrolled sliceMN pipeline")

    torch.manual_seed(args.seed)
    a = torch.randn((args.M, args.K), dtype=torch.float16, device="cuda")
    b = torch.randn((args.K, args.N), dtype=torch.float16)
    if args.transpose_b:
        b = b.T.contiguous()
    b = b.cuda()
    c = torch.zeros((args.M, args.N), dtype=torch.float32, device="cuda")
    grid = (triton.cdiv(args.M, args.BM) * triton.cdiv(args.N, args.BN), 1)

    def launch():
        return f16_slice_mn_warp_pipeline_kernel_gfx1250[grid](
            a, b, c, args.M, args.N, args.K, a.stride(0), a.stride(1), b.stride(1), b.stride(0), c.stride(0),
            c.stride(1), BLOCK_M=args.BM, BLOCK_N=args.BN, BLOCK_K=args.BK, GRID_MN=grid[0], NUM_XCDS=args.num_xcds,
            GROUP_SIZE_M=args.group_size_m, WARP_BASES=((0, 1), (1, 0), (2, 0)), num_warps=args.num_warps,
            NUM_BUFFERS=args.num_buffers, RESOLVE_PARTITION_CONFLICTS=args.resolve_partition_conflicts,
            NUM_WARPS=args.num_warps, waves_per_eu=args.num_warps // 4)

    def check():
        c.zero_()
        launch()
        torch.cuda.synchronize()
        ref_b = b.cpu().T.to(torch.float32) if args.transpose_b else b.cpu().to(torch.float32)
        ref = a.cpu().to(torch.float32) @ ref_b
        torch.testing.assert_close(c.cpu(), ref, rtol=1e-4, atol=1e-4)
        print("result verified", flush=True)

    return launch, check


def _make_mxfp_case(args):
    if args.dtype_a not in MXFP_DTYPE_TO_KERNEL or args.dtype_b not in MXFP_DTYPE_TO_KERNEL:
        raise ValueError("The 8/4-bit path supports float8_e5m2, float8_e4m3, and float4 inputs")
    if not args.transpose_b:
        raise ValueError("The sliceMN MXFP kernel expects --transpose-b so K is contiguous in B")
    if args.num_warps not in (4, 8):
        raise ValueError("The sliceMN MXFP kernel supports --num-warps 4 or 8")
    if args.num_buffers < 2:
        raise ValueError("--num-buffers must be at least 2")
    if (args.BM, args.BN) != (256, 256):
        raise ValueError("The MXFP sliceMN port currently expects -BM 256 -BN 256")
    if args.BK % args.scale_block != 0:
        raise ValueError("BLOCK_K must be divisible by --scale-block")
    scale_k = triton.cdiv(args.K, args.scale_block)
    scale_kwidth = 4 if scale_k >= 4 else scale_k
    if args.scale_preshuffled:
        if args.with_a_scale and args.M % 128 != 0:
            raise ValueError("--scale-preshuffled requires M to be divisible by 128 when --with-a-scale is set")
        if args.N % 128 != 0:
            raise ValueError("--scale-preshuffled requires N to be divisible by 128")
        if scale_k % scale_kwidth != 0:
            raise ValueError("--scale-preshuffled requires K/scale_block to be divisible by the scale K-width")
    if args.async_copy_scale and args.num_warps != 4:
        raise ValueError("--async-copy-scale is currently supported only with --num-warps 4")
    if triton.cdiv(args.K, args.BK) <= 3:
        raise ValueError("K/BLOCK_K must be greater than 3 for the 2x-unrolled sliceMN pipeline")

    torch.manual_seed(args.seed)
    a = init_data(args.dtype_a, args.M, args.K)
    b = init_data(args.dtype_b, args.K, args.N)
    if args.with_a_scale:
        a_scale_obj = MXScaleTensor(size=(args.M, scale_k)).random(low=1.0, high=32.0)
        a_scale = a_scale_obj.data
    else:
        a_scale_obj = None
        a_scale = None
    b_scale_obj = MXScaleTensor(size=(args.N, scale_k)).random(low=1.0, high=32.0)
    b_scale = b_scale_obj.data

    c_ref = None
    if args.check:
        c_ref = torch_gemm_mxfp(a, b, a_scale_obj, b_scale_obj, args.scale_block, args.M, args.N, args.K)

    if args.scale_preshuffled:
        a_scale = pack_scale(a_scale)
        b_scale = pack_scale(b_scale)

    if args.dtype_a == "float4":
        a = a.to_packed_tensor(dim=1)
    if args.dtype_b == "float4":
        b = b.to_packed_tensor(dim=0)

    a_d = a.data.contiguous().cuda()
    if args.transpose_b:
        b_d = b.data.T.contiguous().cuda()
    else:
        b_d = b.data.contiguous().cuda()
    a_scale_d = a_scale.cuda() if args.with_a_scale else None
    b_scale_d = b_scale.cuda()
    c_d = torch.zeros((args.M, args.N), dtype=torch.float32, device="cuda")

    stride_am, stride_ak = a_d.stride(0), a_d.stride(1)
    stride_bk, stride_bn = (b_d.stride(1), b_d.stride(0)) if args.transpose_b else (b_d.stride(0), b_d.stride(1))
    stride_cm, stride_cn = c_d.stride(0), c_d.stride(1)
    stride_scale = b_scale_d.stride(0)
    grid = (triton.cdiv(args.M, args.BM) * triton.cdiv(args.N, args.BN), 1)

    def launch():
        return mxfp_slice_mn_warp_pipeline_kernel_gfx1250[grid](
            a_d, b_d, c_d, a_scale_d, b_scale_d, args.M, args.N, args.K, stride_am, stride_ak, stride_bk, stride_bn,
            stride_cm, stride_cn, stride_scale, MXFP_DTYPE_TO_KERNEL[args.dtype_a], MXFP_DTYPE_TO_KERNEL[args.dtype_b],
            args.scale_block, args.BM, args.BN, args.BK, args.group_size_m, GRID_MN=grid[0], NUM_XCDS=args.num_xcds,
            TRANSPOSE_B=args.transpose_b, WITH_A_SCALE=args.with_a_scale, NUM_WARPS=args.num_warps,
            NUM_BUFFERS=args.num_buffers, RESOLVE_PARTITION_CONFLICTS=args.resolve_partition_conflicts,
            SCALE_PRESHUFFLE=args.scale_preshuffled, ASYNC_COPY_SCALE=args.async_copy_scale,
            num_warps=args.num_warps, waves_per_eu=args.num_warps // 4)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(c_d.cpu(), c_ref.cpu(), rtol=1e-5, atol=1e-8)
        print("result verified", flush=True)

    return launch, check


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="gfx1250 warp-pipelined GEMM examples for 16/8/4-bit inputs")
    parser.add_argument("-M", type=int, default=8192, help="problem M size")
    parser.add_argument("-N", type=int, default=8192, help="problem N size")
    parser.add_argument("-K", type=int, default=1024, help="problem K size")
    parser.add_argument("-BM", type=int, default=256, help="BLOCK_M")
    parser.add_argument("-BN", type=int, default=256, help="BLOCK_N")
    parser.add_argument("-BK", type=int, default=64, help="BLOCK_K")
    parser.add_argument("--dtype-a", default="float16", choices=["float16", "float8_e4m3", "float8_e5m2", "float4"])
    parser.add_argument("--dtype-b", default=None, choices=["float16", "float8_e4m3", "float8_e5m2", "float4"])
    parser.add_argument("--num-warps", type=int, default=8, choices=[4, 8])
    parser.add_argument("--num-buffers", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--group-size-m", type=int, default=8, choices=[1, 2, 4, 8])
    parser.add_argument("--num-xcds", type=int, default=8)
    parser.add_argument("--transpose-b", action="store_true", default=True)
    parser.add_argument("--no-transpose-b", action="store_false", dest="transpose_b")
    parser.add_argument("--scale-block", type=int, default=32, help="MXFP scale block")
    parser.add_argument("--with-a-scale", action="store_true", help="Use A scales on the MXFP path")
    parser.add_argument("--scale-preshuffled", "--scale_preshuffled", dest="scale_preshuffled", action="store_true",
                        help="Use preshuffled MXFP scale tensors")
    parser.add_argument("--async-copy-scale", "--async_copy_scale", dest="async_copy_scale", action="store_true",
                        help="Stage MXFP scale tensors with async copy instead of TDM")
    parser.add_argument("--resolve-partition-conflicts", action="store_true",
                        help="Use partition-aware gfx1250 WMMA/shared layouts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true", help="Check output against torch")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark with CUDA graph replay")
    parser.add_argument("--warmup", type=int, default=10, help="Benchmark warmup iterations")
    parser.add_argument("--probe-iters", type=int, default=20, help="Iterations for CUDA event timing probe")
    parser.add_argument("--graph-ms", type=float, default=100.0, help="Target CUDA graph body duration in ms")
    parser.add_argument("--n-replays", type=int, default=20, help="Number of CUDA graph replays to time")
    parser.add_argument("--iters-per-graph", type=int, default=None, help="Override graph body iteration count")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.dtype_b is None:
        args.dtype_b = args.dtype_a
    if not args.check and not args.benchmark:
        args.check = True

    print(
        f"({args.M=}, {args.N=}, {args.K=}), ({args.BM=}, {args.BN=}, {args.BK=}), "
        f"{args.dtype_a=}, {args.dtype_b=}, {args.num_warps=}, {args.num_buffers=}, "
        f"{args.transpose_b=}, {args.scale_preshuffled=}, {args.async_copy_scale=}, "
        f"{args.resolve_partition_conflicts=}, sliceMN=True"
    )

    if args.dtype_a == "float16":
        launch, check = _make_f16_case(args)
    else:
        launch, check = _make_mxfp_case(args)

    if args.check:
        check()
    if args.benchmark:
        _run_benchmark(launch, args.M, args.N, args.K, args.warmup, args.probe_iters, args.graph_ms, args.n_replays,
                       args.iters_per_graph)
