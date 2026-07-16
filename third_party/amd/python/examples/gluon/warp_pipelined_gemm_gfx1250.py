import argparse
import time

import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm

try:
    from .f16_gemm_warp_pipeline_gfx1250 import get_xcd_swizzled_pids
    from .mxfp_gemm_gfx1250 import (
        get_wmma_layout,
        MXFPGEMMConfig,
        MXScaleTensor,
        init_data,
        torch_gemm_mxfp,
    )
except ImportError:
    from f16_gemm_warp_pipeline_gfx1250 import get_xcd_swizzled_pids
    from mxfp_gemm_gfx1250 import (
        get_wmma_layout,
        MXFPGEMMConfig,
        MXScaleTensor,
        init_data,
        torch_gemm_mxfp,
    )


MXFP_DTYPE_TO_KERNEL = {
    "float8_e5m2": "e5m2",
    "float8_e4m3": "e4m3",
    "float4": "e2m1",
}


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
def f16_slice_mn_warp_pipeline_kernel_gfx1250(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk,
                                             stride_bn, stride_cm, stride_cn, BLOCK_M: gl.constexpr,
                                             BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GRID_MN: gl.constexpr,
                                             NUM_XCDS: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
                                             WARP_BASES: gl.constexpr):
    gl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 64)
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    shared_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M // 2, BLOCK_K], [1, 0])
    shared_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_N // 2, BLOCK_K], [1, 0])
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma_layout, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma_layout, 8)
    store_layout: gl.constexpr = gl.BlockedLayout([4, 8], [4, 16], [2, 4], [1, 0])

    nbuf: gl.constexpr = 2
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

    for i in gl.static_range(2):
        tdm.async_load(b_left_desc, [0, i * BLOCK_K], b_left_buf.index(i))
        tdm.async_load(a_top_desc, [0, i * BLOCK_K], a_top_buf.index(i))
        tdm.async_load(a_bot_desc, [0, i * BLOCK_K], a_bot_buf.index(i))
        tdm.async_load(b_right_desc, [0, i * BLOCK_K], b_right_buf.index(i))

    tdm.async_wait(6)
    a_top = a_top_buf.index(0).load(layout=dot_a)
    b_left = b_left_buf.index(0).permute([1, 0]).load(layout=dot_b)

    acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), dtype=gl.float32, layout=wmma_layout)
    acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), dtype=gl.float32, layout=wmma_layout)
    acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), dtype=gl.float32, layout=wmma_layout)
    acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), dtype=gl.float32, layout=wmma_layout)

    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max > 3)
    load_k = 2
    for _ in range(0, iter_max - 2, 2):
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.gfx1250.wmma(a_top, b_left, acc_tl)
        tdm.async_wait(5)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = a_bot_buf.index(0).load(layout=dot_a)
            tdm.async_load(b_left_desc, [0, load_k * BLOCK_K], b_left_buf.index(0))

        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.gfx1250.wmma(a_bot, b_left, acc_bl)
        tdm.async_wait(5)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = b_right_buf.index(0).permute([1, 0]).load(layout=dot_b)
            tdm.async_load(a_top_desc, [0, load_k * BLOCK_K], a_top_buf.index(0))

        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.gfx1250.wmma(a_top, b_right, acc_tr)
        tdm.async_wait(5)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = b_left_buf.index(1).permute([1, 0]).load(layout=dot_b)
            tdm.async_load(a_bot_desc, [0, load_k * BLOCK_K], a_bot_buf.index(0))

        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.gfx1250.wmma(a_bot, b_right, acc_br)
        tdm.async_wait(5)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = a_top_buf.index(1).load(layout=dot_a)
            tdm.async_load(b_right_desc, [0, load_k * BLOCK_K], b_right_buf.index(0))

        load_k += 1
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.gfx1250.wmma(a_top, b_left, acc_tl)
        tdm.async_wait(5)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = a_bot_buf.index(1).load(layout=dot_a)
            tdm.async_load(b_left_desc, [0, load_k * BLOCK_K], b_left_buf.index(1))

        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.gfx1250.wmma(a_bot, b_left, acc_bl)
        tdm.async_wait(5)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = b_right_buf.index(1).permute([1, 0]).load(layout=dot_b)
            tdm.async_load(a_top_desc, [0, load_k * BLOCK_K], a_top_buf.index(1))

        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.gfx1250.wmma(a_top, b_right, acc_tr)
        tdm.async_wait(5)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = b_left_buf.index(0).permute([1, 0]).load(layout=dot_b)
            tdm.async_load(a_bot_desc, [0, load_k * BLOCK_K], a_bot_buf.index(1))

        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.gfx1250.wmma(a_bot, b_right, acc_br)
        tdm.async_wait(5)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = a_top_buf.index(0).load(layout=dot_a)
            tdm.async_load(b_right_desc, [0, load_k * BLOCK_K], b_right_buf.index(1))
        load_k += 1

    acc_tl = gl.amd.gfx1250.wmma(a_top, b_left, acc_tl)
    tdm.async_wait(5)
    l_idx = (iter_max - 2) % 2
    a_bot = a_bot_buf.index(l_idx).load(layout=dot_a)
    acc_bl = gl.amd.gfx1250.wmma(a_bot, b_left, acc_bl)
    tdm.async_wait(4)
    b_right = b_right_buf.index(l_idx).permute([1, 0]).load(layout=dot_b)
    acc_tr = gl.amd.gfx1250.wmma(a_top, b_right, acc_tr)
    tdm.async_wait(3)
    g_idx = 1 - l_idx
    b_left = b_left_buf.index(g_idx).permute([1, 0]).load(layout=dot_b)
    acc_br = gl.amd.gfx1250.wmma(a_bot, b_right, acc_br)
    tdm.async_wait(2)
    a_top = a_top_buf.index(g_idx).load(layout=dot_a)

    acc_tl = gl.amd.gfx1250.wmma(a_top, b_left, acc_tl)
    tdm.async_wait(1)
    a_bot = a_bot_buf.index(g_idx).load(layout=dot_a)
    acc_bl = gl.amd.gfx1250.wmma(a_bot, b_left, acc_bl)
    tdm.async_wait(0)
    b_right = b_right_buf.index(g_idx).permute([1, 0]).load(layout=dot_b)
    acc_tr = gl.amd.gfx1250.wmma(a_top, b_right, acc_tr)
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
                                              WITH_A_SCALE: gl.constexpr, NUM_WARPS: gl.constexpr):
    gl.static_assert(TRANSPOSE_B)
    cfg: gl.constexpr = MXFPGEMMConfig(BLOCK_M, BLOCK_N, BLOCK_K, DTYPE_A, DTYPE_B, SCALE_BLOCK, 2, TRANSPOSE_B,
                                       WITH_A_SCALE, False, NUM_WARPS, False, (2, 2, 1), -1, "", False)
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    HALF_M: gl.constexpr = BLOCK_M // 2
    HALF_N: gl.constexpr = BLOCK_N // 2
    BK_A: gl.constexpr = BLOCK_K // cfg.DIV_FACTOR_A
    BK_B: gl.constexpr = BLOCK_K // cfg.DIV_FACTOR_B
    BK_SCALE: gl.constexpr = BLOCK_K // SCALE_BLOCK

    shared_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BK_A if BK_A >= 256 else 256, 16]],
                                                                     [HALF_M, BK_A], [1, 0])
    shared_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BK_B if BK_B >= 256 else 256, 16]],
                                                                     [HALF_N, BK_B], [1, 0])
    shared_scale: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[256, 8]], [128, BK_SCALE], [1, 0])
    INSTR_M: gl.constexpr = 32 if (DTYPE_A == "e2m1" and DTYPE_B == "e2m1") else 16
    wmma: gl.constexpr = get_wmma_layout(NUM_WARPS, False, False, INSTR_M)
    wmma_packed: gl.constexpr = get_wmma_layout(NUM_WARPS, True, False, INSTR_M)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma_packed if DTYPE_A == "e2m1" else wmma, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma_packed if DTYPE_B == "e2m1" else wmma, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_a, [HALF_M, BK_SCALE])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_b, [HALF_N, BK_SCALE])
    store_layout: gl.constexpr = gl.BlockedLayout([4, 8], [4, 16], [NUM_WARPS // 4, 4], [1, 0])

    nbuf: gl.constexpr = 2
    a_top_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, HALF_M, BK_A], shared_a)
    a_bot_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, HALF_M, BK_A], shared_a)
    b_left_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, HALF_N, BK_B], shared_b)
    b_right_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, HALF_N, BK_B], shared_b)
    if WITH_A_SCALE:
        as_top_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty, [nbuf, HALF_M, BK_SCALE], shared_scale)
        as_bot_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty, [nbuf, HALF_M, BK_SCALE], shared_scale)
    else:
        as_top_buf = gl.constexpr(0)
        as_bot_buf = gl.constexpr(0)
    bs_left_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty, [nbuf, HALF_N, BK_SCALE], shared_scale)
    bs_right_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty, [nbuf, HALF_N, BK_SCALE], shared_scale)

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
        as_top_desc = tdm.make_tensor_descriptor(base=a_scale_ptr + pid_m * BLOCK_M * stride_scale,
                                                 shape=(M, K // SCALE_BLOCK), strides=(stride_scale, 1),
                                                 block_shape=(HALF_M, BK_SCALE), layout=shared_scale)
        as_bot_desc = tdm.make_tensor_descriptor(base=a_scale_ptr + (pid_m * BLOCK_M + HALF_M) * stride_scale,
                                                 shape=(M, K // SCALE_BLOCK), strides=(stride_scale, 1),
                                                 block_shape=(HALF_M, BK_SCALE), layout=shared_scale)
    else:
        as_top_desc = gl.constexpr(0)
        as_bot_desc = gl.constexpr(0)
    bs_left_desc = tdm.make_tensor_descriptor(base=b_scale_ptr + pid_n * BLOCK_N * stride_scale,
                                              shape=(N, K // SCALE_BLOCK), strides=(stride_scale, 1),
                                              block_shape=(HALF_N, BK_SCALE), layout=shared_scale)
    bs_right_desc = tdm.make_tensor_descriptor(base=b_scale_ptr + (pid_n * BLOCK_N + HALF_N) * stride_scale,
                                               shape=(N, K // SCALE_BLOCK), strides=(stride_scale, 1),
                                               block_shape=(HALF_N, BK_SCALE), layout=shared_scale)

    for i in gl.static_range(2):
        tdm.async_load(b_left_desc, [0, i * BK_B], b_left_buf.index(i))
        tdm.async_load(bs_left_desc, [0, i * BK_SCALE], bs_left_buf.index(i))
        tdm.async_load(a_top_desc, [0, i * BK_A], a_top_buf.index(i))
        if WITH_A_SCALE:
            tdm.async_load(as_top_desc, [0, i * BK_SCALE], as_top_buf.index(i))
        tdm.async_load(a_bot_desc, [0, i * BK_A], a_bot_buf.index(i))
        if WITH_A_SCALE:
            tdm.async_load(as_bot_desc, [0, i * BK_SCALE], as_bot_buf.index(i))
        tdm.async_load(b_right_desc, [0, i * BK_B], b_right_buf.index(i))
        tdm.async_load(bs_right_desc, [0, i * BK_SCALE], bs_right_buf.index(i))

    wait_unit: gl.constexpr = 8 if WITH_A_SCALE else 6
    tdm.async_wait(wait_unit - 2)
    a_top = a_top_buf.index(0).load(layout=dot_a)
    if WITH_A_SCALE:
        as_top = as_top_buf.index(0).load(layout=scale_a_layout)
    else:
        as_top = 0
        as_top = as_top.to(gl.uint8)
    b_left = b_left_buf.index(0).permute([1, 0]).load(layout=dot_b)
    bs_left = bs_left_buf.index(0).load(layout=scale_b_layout)

    acc_tl = gl.zeros((HALF_M, HALF_N), dtype=gl.float32, layout=wmma)
    acc_bl = gl.zeros((HALF_M, HALF_N), dtype=gl.float32, layout=wmma)
    acc_tr = gl.zeros((HALF_M, HALF_N), dtype=gl.float32, layout=wmma)
    acc_br = gl.zeros((HALF_M, HALF_N), dtype=gl.float32, layout=wmma)

    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max > 3)
    load_k = 2
    for _ in range(0, iter_max - 2, 2):
        with gl.amd.warp_pipeline_stage("wmma", priority=0):
            acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_left, bs_left, DTYPE_B, acc_tl)
        tdm.async_wait(wait_unit - 3)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = a_bot_buf.index(0).load(layout=dot_a)
            if WITH_A_SCALE:
                as_bot = as_bot_buf.index(0).load(layout=scale_a_layout)
            else:
                as_bot = as_top
            tdm.async_load(b_left_desc, [0, load_k * BK_B], b_left_buf.index(0))
            tdm.async_load(bs_left_desc, [0, load_k * BK_SCALE], bs_left_buf.index(0))

        acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_left, bs_left, DTYPE_B, acc_bl)
        tdm.async_wait(wait_unit - 3)
        b_right = b_right_buf.index(0).permute([1, 0]).load(layout=dot_b)
        bs_right = bs_right_buf.index(0).load(layout=scale_b_layout)
        tdm.async_load(a_top_desc, [0, load_k * BK_A], a_top_buf.index(0))
        if WITH_A_SCALE:
            tdm.async_load(as_top_desc, [0, load_k * BK_SCALE], as_top_buf.index(0))

        acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_right, bs_right, DTYPE_B, acc_tr)
        tdm.async_wait(wait_unit - 3)
        b_left = b_left_buf.index(1).permute([1, 0]).load(layout=dot_b)
        bs_left = bs_left_buf.index(1).load(layout=scale_b_layout)
        tdm.async_load(a_bot_desc, [0, load_k * BK_A], a_bot_buf.index(0))
        if WITH_A_SCALE:
            tdm.async_load(as_bot_desc, [0, load_k * BK_SCALE], as_bot_buf.index(0))

        acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_right, bs_right, DTYPE_B, acc_br)
        tdm.async_wait(wait_unit - 3)
        a_top = a_top_buf.index(1).load(layout=dot_a)
        if WITH_A_SCALE:
            as_top = as_top_buf.index(1).load(layout=scale_a_layout)
        tdm.async_load(b_right_desc, [0, load_k * BK_B], b_right_buf.index(0))
        tdm.async_load(bs_right_desc, [0, load_k * BK_SCALE], bs_right_buf.index(0))

        load_k += 1
        acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_left, bs_left, DTYPE_B, acc_tl)
        tdm.async_wait(wait_unit - 3)
        a_bot = a_bot_buf.index(1).load(layout=dot_a)
        if WITH_A_SCALE:
            as_bot = as_bot_buf.index(1).load(layout=scale_a_layout)
        tdm.async_load(b_left_desc, [0, load_k * BK_B], b_left_buf.index(1))
        tdm.async_load(bs_left_desc, [0, load_k * BK_SCALE], bs_left_buf.index(1))

        acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_left, bs_left, DTYPE_B, acc_bl)
        tdm.async_wait(wait_unit - 3)
        b_right = b_right_buf.index(1).permute([1, 0]).load(layout=dot_b)
        bs_right = bs_right_buf.index(1).load(layout=scale_b_layout)
        tdm.async_load(a_top_desc, [0, load_k * BK_A], a_top_buf.index(1))
        if WITH_A_SCALE:
            tdm.async_load(as_top_desc, [0, load_k * BK_SCALE], as_top_buf.index(1))

        acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_right, bs_right, DTYPE_B, acc_tr)
        tdm.async_wait(wait_unit - 3)
        b_left = b_left_buf.index(0).permute([1, 0]).load(layout=dot_b)
        bs_left = bs_left_buf.index(0).load(layout=scale_b_layout)
        tdm.async_load(a_bot_desc, [0, load_k * BK_A], a_bot_buf.index(1))
        if WITH_A_SCALE:
            tdm.async_load(as_bot_desc, [0, load_k * BK_SCALE], as_bot_buf.index(1))

        acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_right, bs_right, DTYPE_B, acc_br)
        tdm.async_wait(wait_unit - 3)
        a_top = a_top_buf.index(0).load(layout=dot_a)
        if WITH_A_SCALE:
            as_top = as_top_buf.index(0).load(layout=scale_a_layout)
        tdm.async_load(b_right_desc, [0, load_k * BK_B], b_right_buf.index(1))
        tdm.async_load(bs_right_desc, [0, load_k * BK_SCALE], bs_right_buf.index(1))
        load_k += 1

    # Drain the last two prefetched K tiles using the same quadrant order.
    acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_left, bs_left, DTYPE_B, acc_tl)
    tdm.async_wait(wait_unit - 3)
    l_idx = (iter_max - 2) % 2
    a_bot = a_bot_buf.index(l_idx).load(layout=dot_a)
    if WITH_A_SCALE:
        as_bot = as_bot_buf.index(l_idx).load(layout=scale_a_layout)
    else:
        as_bot = as_top
    acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_left, bs_left, DTYPE_B, acc_bl)
    tdm.async_wait(wait_unit - 4)
    b_right = b_right_buf.index(l_idx).permute([1, 0]).load(layout=dot_b)
    bs_right = bs_right_buf.index(l_idx).load(layout=scale_b_layout)
    acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_right, bs_right, DTYPE_B, acc_tr)
    tdm.async_wait(wait_unit - 5)
    g_idx = 1 - l_idx
    b_left = b_left_buf.index(g_idx).permute([1, 0]).load(layout=dot_b)
    bs_left = bs_left_buf.index(g_idx).load(layout=scale_b_layout)
    acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_right, bs_right, DTYPE_B, acc_br)
    tdm.async_wait(wait_unit - 6)
    a_top = a_top_buf.index(g_idx).load(layout=dot_a)
    if WITH_A_SCALE:
        as_top = as_top_buf.index(g_idx).load(layout=scale_a_layout)

    acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_left, bs_left, DTYPE_B, acc_tl)
    tdm.async_wait(1)
    a_bot = a_bot_buf.index(g_idx).load(layout=dot_a)
    if WITH_A_SCALE:
        as_bot = as_bot_buf.index(g_idx).load(layout=scale_a_layout)
    else:
        as_bot = as_top
    acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_left, bs_left, DTYPE_B, acc_bl)
    tdm.async_wait(0)
    b_right = b_right_buf.index(g_idx).permute([1, 0]).load(layout=dot_b)
    bs_right = bs_right_buf.index(g_idx).load(layout=scale_b_layout)
    acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_right, bs_right, DTYPE_B, acc_tr)
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
    if not args.transpose_b:
        raise ValueError("The sliceMN fp16 kernel expects --transpose-b so K is contiguous in B")
    if (args.BM, args.BN, args.BK) != (256, 256, 64):
        raise ValueError("The fp16 sliceMN port currently expects -BM 256 -BN 256 -BK 64")
    if args.num_warps != 8:
        raise ValueError("The fp16 inter-wave sliceMN port expects --num-warps 8")
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
            waves_per_eu=args.num_warps // 4)

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
    if (args.BM, args.BN) != (256, 256):
        raise ValueError("The MXFP sliceMN port currently expects -BM 256 -BN 256")
    if args.BK % args.scale_block != 0:
        raise ValueError("BLOCK_K must be divisible by --scale-block")
    if triton.cdiv(args.K, args.BK) <= 3:
        raise ValueError("K/BLOCK_K must be greater than 3 for the 2x-unrolled sliceMN pipeline")

    torch.manual_seed(args.seed)
    a = init_data(args.dtype_a, args.M, args.K)
    b = init_data(args.dtype_b, args.K, args.N)
    scale_k = triton.cdiv(args.K, args.scale_block)
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
    parser.add_argument("--group-size-m", type=int, default=8, choices=[1, 2, 4, 8])
    parser.add_argument("--num-xcds", type=int, default=8)
    parser.add_argument("--transpose-b", action="store_true", default=True)
    parser.add_argument("--no-transpose-b", action="store_false", dest="transpose_b")
    parser.add_argument("--scale-block", type=int, default=32, help="MXFP scale block")
    parser.add_argument("--with-a-scale", action="store_true", help="Use A scales on the MXFP path")
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
        f"{args.dtype_a=}, {args.dtype_b=}, {args.num_warps=}, {args.transpose_b=}, sliceMN=True"
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
