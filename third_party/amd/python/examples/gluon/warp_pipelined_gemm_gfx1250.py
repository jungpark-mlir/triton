import argparse
import time

import torch
import triton
from triton._C.libtriton.gluon_ir import make_cga_layout
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import async_copy as cp
from triton.experimental.gluon.language.amd.gfx1250 import tdm
from triton.tools.mxfp import MXFP4Tensor

try:
    from .mxfp_gemm_cdna5 import (
        get_scale_blocked_layout,
        get_wmma_layout,
        MXFPGEMMConfig,
        MXScaleTensor,
        init_data,
        pack_scale,
        torch_gemm_mxfp,
    )
except ImportError:
    from mxfp_gemm_cdna5 import (
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
def get_xcd_swizzled_pids(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, GRID_MN: gl.constexpr,
                          NUM_XCDS: gl.constexpr, GROUP_SIZE_M: gl.constexpr, Z_ORDER: gl.constexpr = False,
                          SWIZZLE_BLOCK_M: gl.constexpr = 0, SWIZZLE_BLOCK_N: gl.constexpr = 0):
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

    if SWIZZLE_BLOCK_M > 0:
        block_area: gl.constexpr = SWIZZLE_BLOCK_M * SWIZZLE_BLOCK_N
        blocks_n = num_pid_n // SWIZZLE_BLOCK_N
        block_id = pid // block_area
        local_pid = pid % block_area
        pid_m = (block_id // blocks_n) * SWIZZLE_BLOCK_M + local_pid % SWIZZLE_BLOCK_M
        pid_n = (block_id % blocks_n) * SWIZZLE_BLOCK_N + local_pid // SWIZZLE_BLOCK_M
    elif Z_ORDER:
        # Decode a Morton index by compacting its alternating bits. The host
        # restricts this mode to square power-of-two CTA grids.
        pid_m = 0
        pid_n = 0
        for bit in gl.static_range(0, 16):
            pid_m |= ((pid >> (2 * bit)) & 1) << bit
            pid_n |= ((pid >> (2 * bit + 1)) & 1) << bit
    elif GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
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
def _store_n_halves(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc_l, acc_r,
                    STORE_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, STORE_LAYOUT))
    offs_n = gl.arange(0, BLOCK_N // 2, layout=gl.SliceLayout(0, STORE_LAYOUT))
    offsets = stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    mask_l = (pid_m * BLOCK_M + offs_m[:, None] < M) & (pid_n * BLOCK_N + offs_n[None, :] < N)
    mask_r = (pid_m * BLOCK_M + offs_m[:, None] < M) & (
        pid_n * BLOCK_N + BLOCK_N // 2 + offs_n[None, :] < N)

    gl.store(base + offsets, acc_l.to(c_ptr.type.element_ty), mask=mask_l)
    gl.store(base + (BLOCK_N // 2) * stride_cn + offsets, acc_r.to(c_ptr.type.element_ty), mask=mask_r)


@gluon.jit
def _store_full_tile(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc, STORE_LAYOUT: gl.constexpr,
                     BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, STORE_LAYOUT))
    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, STORE_LAYOUT))
    offsets = stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    base_m = pid_m * BLOCK_M
    base_n = pid_n * BLOCK_N
    base = c_ptr + base_m * stride_cm + base_n * stride_cn
    mask = (base_m + offs_m[:, None] < M) & (base_n + offs_n[None, :] < N)
    gl.store(base + offsets, acc.to(c_ptr.type.element_ty), mask=mask)


@gluon.jit
def _tdm_store_full_tile(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
                         STORE_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                         CTA_N: gl.constexpr):
    # Pad each CTA-local row by eight BF16 elements. This avoids the LDS bank
    # conflicts of the identity layout and outperforms the swizzled TDM store.
    c_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N // CTA_N, 8]], [BLOCK_M, BLOCK_N], [1, 0], STORE_LAYOUT.cga_layout)
    c_shared = gl.allocate_shared_memory(c_ptr.type.element_ty, [BLOCK_M, BLOCK_N], c_shared_layout)
    c_shared.store(acc.to(c_ptr.type.element_ty))
    c_desc = tdm.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                        block_shape=(BLOCK_M, BLOCK_N), layout=c_shared_layout)
    tdm.async_store(c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], c_shared)
    tdm.async_wait(0)


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
def _wait_fp8_scaled_warp_pipeline(waitcnt: int, ASYNC_COPY_SCALE: gl.constexpr):
    if ASYNC_COPY_SCALE:
        tdm.async_wait(0)
        with gl.amd.warp_pipeline_stage("wait_split", priority=1):
            pass
        cp.wait_group(0)
    else:
        tdm.async_wait(waitcnt)


@gluon.jit
def _cluster_wait(waitcnt: gl.constexpr):
    gl.amd.gfx1250.cluster.arrive()
    gl.amd.gfx1250.cluster.wait()


@gluon.jit
def _issue_mxfp_scale_load(scale_desc, scale_ptrs, load_k, scale_buffer, slot, BK_SCALE_PRESHUFFLED: gl.constexpr,
                           ASYNC_COPY_SCALE: gl.constexpr):
    if ASYNC_COPY_SCALE:
        cp.global_to_shared(scale_buffer.index(slot), scale_ptrs + load_k * BK_SCALE_PRESHUFFLED)
        cp.commit_group()
    else:
        tdm.async_load(scale_desc, [0, load_k * BK_SCALE_PRESHUFFLED], scale_buffer.index(slot))


@gluon.jit
def _issue_mxfp4_fused_scale_load(as_top_desc, as_bot_desc, bs_left_desc, bs_right_desc, load_k, as_top_buf,
                                  as_bot_buf, bs_left_buf, bs_right_buf, slot,
                                  BK_SCALE_PRESHUFFLED: gl.constexpr):
    scale_k = load_k * BK_SCALE_PRESHUFFLED
    as_top_load_desc = tdm.update_tensor_descriptor(as_top_desc, add_offsets=[0, scale_k])
    as_bot_load_desc = tdm.update_tensor_descriptor(as_bot_desc, add_offsets=[0, scale_k])
    bs_left_load_desc = tdm.update_tensor_descriptor(bs_left_desc, add_offsets=[0, scale_k])
    bs_right_load_desc = tdm.update_tensor_descriptor(bs_right_desc, add_offsets=[0, scale_k])
    tdm.async_load_fused([
        (as_top_load_desc, as_top_buf.index(slot), 0b00000011),
        (as_bot_load_desc, as_bot_buf.index(slot), 0b00001100),
        (bs_left_load_desc, bs_left_buf.index(slot), 0b00110000),
        (bs_right_load_desc, bs_right_buf.index(slot), 0b11000000),
    ])


@gluon.jit
def f16_slice_mn_warp_pipeline_kernel_gfx1250(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk,
                                             stride_bn, stride_cm, stride_cn, BLOCK_M: gl.constexpr,
                                             BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GRID_MN: gl.constexpr,
                                             NUM_XCDS: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
                                             WARP_BASES: gl.constexpr, NUM_BUFFERS: gl.constexpr,
                                             RESOLVE_PARTITION_CONFLICTS: gl.constexpr, NUM_WARPS: gl.constexpr,
                                             MAX_ITER: gl.constexpr):
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
    gl.assume(iter_max <= MAX_ITER)
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
def _issue_f16_cluster_tile(a_buf, b_buf, slot, acc, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
                            WAIT_COUNT: gl.constexpr):
    tdm.async_wait(WAIT_COUNT)
    _cluster_wait(WAIT_COUNT)
    a = a_buf.index(slot).load(layout=DOT_A)
    b = b_buf.index(slot).permute([1, 0]).load(layout=DOT_B)
    return gl.amd.gfx1250.wmma(a, b, acc)


@gluon.jit
def f16_cluster_kernelC_gfx1250(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                                stride_cm, stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                                BLOCK_K: gl.constexpr, GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr,
                                GROUP_SIZE_M: gl.constexpr, NUM_BUFFERS: gl.constexpr, NUM_WARPS: gl.constexpr,
                                SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr,
                                WMMA_LAYOUT: gl.constexpr):
    gl.static_assert(BLOCK_M == 512 and BLOCK_N == 512 and BLOCK_K == 64)
    gl.static_assert(NUM_WARPS == 8)
    gl.static_assert(NUM_BUFFERS == 3 or NUM_BUFFERS == 4,
                     "F16 cluster KernelC requires three or four LDS buffers")
    gl.static_assert(gl.num_ctas() == 4, "F16 cluster KernelC requires a four-CTA cluster")
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    shared_a: gl.constexpr = SHARED_LAYOUT_A
    shared_b: gl.constexpr = SHARED_LAYOUT_B
    wmma: gl.constexpr = WMMA_LAYOUT
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma, 8)
    nbuf: gl.constexpr = NUM_BUFFERS

    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, BLOCK_M, BLOCK_K], shared_a)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, BLOCK_N, BLOCK_K], shared_b)
    a_base = pid_m * BLOCK_M * stride_am
    b_base = pid_n * BLOCK_N * stride_bn
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base, shape=(M, K), strides=(stride_am, stride_ak),
                                        block_shape=(BLOCK_M, BLOCK_K), layout=shared_a)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base, shape=(N, K), strides=(stride_bn, stride_bk),
                                        block_shape=(BLOCK_N, BLOCK_K), layout=shared_b)

    producer = 0
    for _ in gl.static_range(NUM_BUFFERS - 1):
        slot = producer % nbuf
        tdm.async_load(a_desc, [0, producer * BLOCK_K], a_buf.index(slot), warp_used_hint=0b00001111)
        tdm.async_load(b_desc, [0, producer * BLOCK_K], b_buf.index(slot), warp_used_hint=0b00001111)
        producer += 1

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=wmma)
    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max >= NUM_BUFFERS)
    consumer = 0

    for _ in range(0, iter_max - NUM_BUFFERS):
        slot = producer % nbuf
        tdm.async_load(a_desc, [0, producer * BLOCK_K], a_buf.index(slot), warp_used_hint=0b00001111)
        tdm.async_load(b_desc, [0, producer * BLOCK_K], b_buf.index(slot), warp_used_hint=0b00001111)
        producer += 1
        acc = _issue_f16_cluster_tile(a_buf, b_buf, consumer % nbuf, acc, dot_a, dot_b,
                                      2 * (NUM_BUFFERS - 1))
        consumer += 1

    slot = producer % nbuf
    tdm.async_load(a_desc, [0, producer * BLOCK_K], a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(b_desc, [0, producer * BLOCK_K], b_buf.index(slot), warp_used_hint=0b00001111)

    for i in gl.static_range(NUM_BUFFERS):
        acc = _issue_f16_cluster_tile(a_buf, b_buf, consumer % nbuf, acc, dot_a, dot_b,
                                      2 * (NUM_BUFFERS - 1 - i))
        consumer += 1

    _store_full_tile(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc, wmma, BLOCK_M, BLOCK_N)


@gluon.jit
def _bf16_kernelc_3pf_issue_loads(a_desc, b_desc, a_dst, b_dst, BLOCK_K: gl.constexpr):
    tdm.async_load(a_desc, [0, 0], a_dst, warp_used_hint=0b00001111)
    tdm.async_load(b_desc, [0, 0], b_dst, warp_used_hint=0b00001111)
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K])
    return a_desc, b_desc


@gluon.jit
def _bf16_kernelc_3pf_load_slot(a_buf, b_buf, slot, DOT_A: gl.constexpr, DOT_B: gl.constexpr):
    a = a_buf.index(slot).load(layout=DOT_A)
    b = b_buf.index(slot).permute([1, 0]).load(layout=DOT_B)
    return a, b


@gluon.jit
def _bf16_kernelc_3pf_load_dynamic(phase, a_buf, b_buf, DOT_A: gl.constexpr, DOT_B: gl.constexpr):
    a, b = _bf16_kernelc_3pf_load_slot(a_buf, b_buf, phase % 3, DOT_A, DOT_B)
    return phase + 1, a, b


@gluon.jit
def bf16_kernelc_3pf_gfx1250(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                             stride_cm, stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                             BLOCK_K: gl.constexpr, GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr,
                             GROUP_SIZE_M: gl.constexpr, NUM_WARPS: gl.constexpr):
    """Materialized form of the partitioned KernelC that measured 3 PFLOPS."""
    gl.static_assert(a_ptr.type.element_ty.is_bf16() and b_ptr.type.element_ty.is_bf16(),
                     "bf16_kernelc_3pf_gfx1250 requires BF16 inputs")
    gl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 64)
    gl.static_assert(NUM_WARPS == 8)
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    padded_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0])
    padded_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_N, BLOCK_K], [1, 0])
    layouts: gl.constexpr = gl.amd.gfx1250.make_partitioned_dot_layouts(
        BLOCK_M, BLOCK_N, padded_a, padded_b, NUM_WARPS, [16, 16, 32], a_transposed=False, b_transposed=True,
        transposed=True)
    shared_a: gl.constexpr = layouts[0]
    shared_b: gl.constexpr = layouts[1]
    wmma: gl.constexpr = layouts[2]
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma, 8)

    a_base = pid_m * BLOCK_M * stride_am
    b_base = pid_n * BLOCK_N * stride_bn
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base, shape=(M, K), strides=(stride_am, stride_ak),
                                        block_shape=(BLOCK_M, BLOCK_K), layout=shared_a)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base, shape=(N, K), strides=(stride_bn, stride_bk),
                                        block_shape=(BLOCK_N, BLOCK_K), layout=shared_b)
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [3, BLOCK_M, BLOCK_K], shared_a)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [3, BLOCK_N, BLOCK_K], shared_b)

    a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(a_desc, b_desc, a_buf.index(0), b_buf.index(0), BLOCK_K)
    a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(a_desc, b_desc, a_buf.index(1), b_buf.index(1), BLOCK_K)
    tdm.async_wait(2)

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=wmma)

    # Preserve the exact three-slot refill-first schedule used by the measured
    # partitioned experiment. Complete ring rotations use static LDS slots.
    for _ in range(0, (gl.cdiv(K, BLOCK_K) - 2) // 3):
        with gl.amd.warp_pipeline_stage("stage0", priority=0):
            a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(
                a_desc, b_desc, a_buf.index(2), b_buf.index(2), BLOCK_K)
            a, b = _bf16_kernelc_3pf_load_slot(a_buf, b_buf, 0, dot_a, dot_b)
        with gl.amd.warp_pipeline_stage("stage1", priority=1):
            acc = gl.amd.gfx1250.wmma(a, b, acc)
        tdm.async_wait(2)

        with gl.amd.warp_pipeline_stage("stage0", priority=0):
            a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(
                a_desc, b_desc, a_buf.index(0), b_buf.index(0), BLOCK_K)
            a, b = _bf16_kernelc_3pf_load_slot(a_buf, b_buf, 1, dot_a, dot_b)
        with gl.amd.warp_pipeline_stage("stage1", priority=1):
            acc = gl.amd.gfx1250.wmma(a, b, acc)
        tdm.async_wait(2)

        with gl.amd.warp_pipeline_stage("stage0", priority=0):
            a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(
                a_desc, b_desc, a_buf.index(1), b_buf.index(1), BLOCK_K)
            a, b = _bf16_kernelc_3pf_load_slot(a_buf, b_buf, 2, dot_a, dot_b)
        with gl.amd.warp_pipeline_stage("stage1", priority=1):
            acc = gl.amd.gfx1250.wmma(a, b, acc)
        tdm.async_wait(2)

    phase = ((gl.cdiv(K, BLOCK_K) - 2) // 3) * 3
    for _ in range(0, (gl.cdiv(K, BLOCK_K) - 2) % 3):
        with gl.amd.warp_pipeline_stage("stage0", priority=0):
            write_phase = phase + 2
            phase, a, b = _bf16_kernelc_3pf_load_dynamic(phase, a_buf, b_buf, dot_a, dot_b)
            a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(
                a_desc, b_desc, a_buf.index(write_phase % 3), b_buf.index(write_phase % 3), BLOCK_K)
        with gl.amd.warp_pipeline_stage("stage1", priority=1):
            acc = gl.amd.gfx1250.wmma(a, b, acc)
        tdm.async_wait(2)

    for _ in gl.static_range(2):
        with gl.amd.warp_pipeline_stage("stage0", priority=0):
            phase, a, b = _bf16_kernelc_3pf_load_dynamic(phase, a_buf, b_buf, dot_a, dot_b)
        with gl.amd.warp_pipeline_stage("stage1", priority=1):
            acc = gl.amd.gfx1250.wmma(a, b, acc)
        tdm.async_wait(0)

    # The historical 3-PFLOPS result used this one-CTA TDM-store epilogue.
    c_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 4]], [BLOCK_M, BLOCK_N],
                                                                           [1, 0])
    c_shared = gl.allocate_shared_memory(c_ptr.type.element_ty, [BLOCK_M, BLOCK_N], c_shared_layout)
    c_shared.store(acc.to(c_ptr.type.element_ty))
    c_desc = tdm.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                        block_shape=(BLOCK_M, BLOCK_N), layout=c_shared_layout)
    tdm.async_store(c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], c_shared)
    tdm.async_wait(0)


@gluon.jit
def _bf16_kernelc_2pf_consume_slot(a_buf, b_buf, slot, acc, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
                                   HALF_K: gl.constexpr):
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(0, HALF_K, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(0, HALF_K, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)

    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(HALF_K, HALF_K, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(HALF_K, HALF_K, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    return acc


@gluon.jit
def _bf16_kernelc_2pf_consume_and_refill(a_buf, b_buf, slot, a_desc, b_desc, acc, DOT_A: gl.constexpr,
                                        DOT_B: gl.constexpr, HALF_K: gl.constexpr, BLOCK_K: gl.constexpr):
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(0, HALF_K, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(0, HALF_K, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)

    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(HALF_K, HALF_K, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(HALF_K, HALF_K, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        # The stage0 boundary waits for the LDS reads before the refill starts
        # overwriting this slot. WMMA only consumes the register operands.
        a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(
            a_desc, b_desc, a_buf.index(slot), b_buf.index(slot), BLOCK_K)
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    return a_desc, b_desc, acc


@gluon.jit
def bf16_kernelc_2pf_gfx1250(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                             stride_cm, stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                             BLOCK_K: gl.constexpr, GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr,
                             GROUP_SIZE_M: gl.constexpr, NUM_WARPS: gl.constexpr):
    """KernelC full-tile schedule with double-buffered BK128 TDM copies."""
    gl.static_assert(a_ptr.type.element_ty.is_bf16() and b_ptr.type.element_ty.is_bf16(),
                     "bf16_kernelc_2pf_gfx1250 requires BF16 inputs")
    gl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 128)
    gl.static_assert(NUM_WARPS == 8)
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    padded_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0])
    padded_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_N, BLOCK_K], [1, 0])
    layouts: gl.constexpr = gl.amd.gfx1250.make_partitioned_dot_layouts(
        BLOCK_M, BLOCK_N, padded_a, padded_b, NUM_WARPS, [16, 16, 32], a_transposed=False, b_transposed=True,
        transposed=True)
    shared_a: gl.constexpr = layouts[0]
    shared_b: gl.constexpr = layouts[1]
    wmma: gl.constexpr = layouts[2]
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma, 8)

    a_base = pid_m * BLOCK_M * stride_am
    b_base = pid_n * BLOCK_N * stride_bn
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base, shape=(M, K), strides=(stride_am, stride_ak),
                                        block_shape=(BLOCK_M, BLOCK_K), layout=shared_a)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base, shape=(N, K), strides=(stride_bn, stride_bk),
                                        block_shape=(BLOCK_N, BLOCK_K), layout=shared_b)
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [2, BLOCK_M, BLOCK_K], shared_a)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [2, BLOCK_N, BLOCK_K], shared_b)

    a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(a_desc, b_desc, a_buf.index(0), b_buf.index(0), BLOCK_K)
    a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(a_desc, b_desc, a_buf.index(1), b_buf.index(1), BLOCK_K)
    tdm.async_wait(2)

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=wmma)
    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max >= 2)
    half_k: gl.constexpr = BLOCK_K // 2

    # Consume and refill complete two-slot rotations with compile-time LDS
    # addresses. Split each BK128 tile into two full-MN BK64 WMMAs to keep
    # operand register pressure equal to the BK64 KernelC.
    for _ in range(0, (iter_max - 2) // 2):
        a_desc, b_desc, acc = _bf16_kernelc_2pf_consume_and_refill(
            a_buf, b_buf, 0, a_desc, b_desc, acc, dot_a, dot_b, half_k, BLOCK_K)
        tdm.async_wait(2)

        a_desc, b_desc, acc = _bf16_kernelc_2pf_consume_and_refill(
            a_buf, b_buf, 1, a_desc, b_desc, acc, dot_a, dot_b, half_k, BLOCK_K)
        tdm.async_wait(2)

    phase = ((iter_max - 2) // 2) * 2
    for _ in range(0, (iter_max - 2) % 2):
        slot = phase % 2
        a_desc, b_desc, acc = _bf16_kernelc_2pf_consume_and_refill(
            a_buf, b_buf, slot, a_desc, b_desc, acc, dot_a, dot_b, half_k, BLOCK_K)
        tdm.async_wait(2)
        phase += 1

    tdm.async_wait(0)
    for _ in gl.static_range(2):
        acc = _bf16_kernelc_2pf_consume_slot(
            a_buf, b_buf, phase % 2, acc, dot_a, dot_b, half_k)
        phase += 1

    c_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 4]], [BLOCK_M, BLOCK_N],
                                                                           [1, 0])
    c_shared = gl.allocate_shared_memory(c_ptr.type.element_ty, [BLOCK_M, BLOCK_N], c_shared_layout)
    c_shared.store(acc.to(c_ptr.type.element_ty))
    c_desc = tdm.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                        block_shape=(BLOCK_M, BLOCK_N), layout=c_shared_layout)
    tdm.async_store(c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], c_shared)
    tdm.async_wait(0)


@gluon.jit
def _bf16_cluster_4x4_wait(wait_count: gl.constexpr):
    gl.amd.gfx1250.cluster.arrive()
    tdm.async_wait(wait_count)
    gl.amd.gfx1250.cluster.wait()


@gluon.jit
def bf16_cluster_4x4_tdm_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GRID_MN: gl.constexpr,
        NUM_XCDS: gl.constexpr, GROUP_SIZE_M: gl.constexpr, NUM_WARPS: gl.constexpr,
        SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr):
    """Standalone 4x4 cluster KernelC with a CGA-aware TDM-store epilogue."""
    gl.static_assert(a_ptr.type.element_ty.is_bf16() and b_ptr.type.element_ty.is_bf16(),
                     "bf16_cluster_4x4_tdm_kernel_gfx1250 requires BF16 inputs")
    gl.static_assert(BLOCK_M == 1024 and BLOCK_N == 1024 and BLOCK_K == 64)
    gl.static_assert(NUM_WARPS == 8)
    gl.static_assert(gl.num_ctas() == 16, "BF16 4x4 cluster KernelC requires 16 CTAs")
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 8)
    a_base = pid_m * BLOCK_M * stride_am
    b_base = pid_n * BLOCK_N * stride_bn
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base, shape=(M, K), strides=(stride_am, stride_ak),
                                        block_shape=(BLOCK_M, BLOCK_K), layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base, shape=(N, K), strides=(stride_bn, stride_bk),
                                        block_shape=(BLOCK_N, BLOCK_K), layout=SHARED_LAYOUT_B)
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [3, BLOCK_M, BLOCK_K], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [3, BLOCK_N, BLOCK_K], SHARED_LAYOUT_B)

    a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(a_desc, b_desc, a_buf.index(0), b_buf.index(0), BLOCK_K)
    a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(a_desc, b_desc, a_buf.index(1), b_buf.index(1), BLOCK_K)
    _bf16_cluster_4x4_wait(2)

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    for _ in range(0, (gl.cdiv(K, BLOCK_K) - 2) // 3):
        with gl.amd.warp_pipeline_stage("stage0", priority=0):
            a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(
                a_desc, b_desc, a_buf.index(2), b_buf.index(2), BLOCK_K)
            a, b = _bf16_kernelc_3pf_load_slot(a_buf, b_buf, 0, dot_a, dot_b)
        with gl.amd.warp_pipeline_stage("stage1", priority=1):
            acc = gl.amd.gfx1250.wmma(a, b, acc)
        _bf16_cluster_4x4_wait(2)

        with gl.amd.warp_pipeline_stage("stage0", priority=0):
            a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(
                a_desc, b_desc, a_buf.index(0), b_buf.index(0), BLOCK_K)
            a, b = _bf16_kernelc_3pf_load_slot(a_buf, b_buf, 1, dot_a, dot_b)
        with gl.amd.warp_pipeline_stage("stage1", priority=1):
            acc = gl.amd.gfx1250.wmma(a, b, acc)
        _bf16_cluster_4x4_wait(2)

        with gl.amd.warp_pipeline_stage("stage0", priority=0):
            a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(
                a_desc, b_desc, a_buf.index(1), b_buf.index(1), BLOCK_K)
            a, b = _bf16_kernelc_3pf_load_slot(a_buf, b_buf, 2, dot_a, dot_b)
        with gl.amd.warp_pipeline_stage("stage1", priority=1):
            acc = gl.amd.gfx1250.wmma(a, b, acc)
        _bf16_cluster_4x4_wait(2)

    phase = ((gl.cdiv(K, BLOCK_K) - 2) // 3) * 3
    for _ in range(0, (gl.cdiv(K, BLOCK_K) - 2) % 3):
        with gl.amd.warp_pipeline_stage("stage0", priority=0):
            write_phase = phase + 2
            phase, a, b = _bf16_kernelc_3pf_load_dynamic(phase, a_buf, b_buf, dot_a, dot_b)
            a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(
                a_desc, b_desc, a_buf.index(write_phase % 3), b_buf.index(write_phase % 3), BLOCK_K)
        with gl.amd.warp_pipeline_stage("stage1", priority=1):
            acc = gl.amd.gfx1250.wmma(a, b, acc)
        _bf16_cluster_4x4_wait(2)

    for _ in gl.static_range(2):
        with gl.amd.warp_pipeline_stage("stage0", priority=0):
            phase, a, b = _bf16_kernelc_3pf_load_dynamic(phase, a_buf, b_buf, dot_a, dot_b)
        with gl.amd.warp_pipeline_stage("stage1", priority=1):
            acc = gl.amd.gfx1250.wmma(a, b, acc)
        _bf16_cluster_4x4_wait(0)

    c_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0], WMMA_LAYOUT.cga_layout)
    c_shared = gl.allocate_shared_memory(c_ptr.type.element_ty, [BLOCK_M, BLOCK_N], c_shared_layout)
    c_shared.store(acc.to(c_ptr.type.element_ty))
    c_desc = tdm.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                        block_shape=(BLOCK_M, BLOCK_N), layout=c_shared_layout)
    tdm.async_store(c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], c_shared)
    tdm.async_wait(0)


@gluon.jit
def bf16_cluster_4x4_identity_tdm_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, NUM_WARPS: gl.constexpr,
        SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr):
    """4x4 cluster KernelC with direct two-dimensional cluster coordinates."""
    gl.static_assert(a_ptr.type.element_ty.is_bf16() and b_ptr.type.element_ty.is_bf16(),
                     "bf16_cluster_4x4_identity_tdm_kernel_gfx1250 requires BF16 inputs")
    gl.static_assert(BLOCK_M == 1024 and BLOCK_N == 1024 and BLOCK_K == 64)
    gl.static_assert(NUM_WARPS == 8)
    gl.static_assert(gl.num_ctas() == 16, "BF16 4x4 identity cluster KernelC requires 16 CTAs")
    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)

    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 8)
    a_base = pid_m * BLOCK_M * stride_am
    b_base = pid_n * BLOCK_N * stride_bn
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base, shape=(M, K), strides=(stride_am, stride_ak),
                                        block_shape=(BLOCK_M, BLOCK_K), layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base, shape=(N, K), strides=(stride_bn, stride_bk),
                                        block_shape=(BLOCK_N, BLOCK_K), layout=SHARED_LAYOUT_B)
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [3, BLOCK_M, BLOCK_K], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [3, BLOCK_N, BLOCK_K], SHARED_LAYOUT_B)

    a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(a_desc, b_desc, a_buf.index(0), b_buf.index(0), BLOCK_K)
    a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(a_desc, b_desc, a_buf.index(1), b_buf.index(1), BLOCK_K)
    _bf16_cluster_4x4_wait(2)

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    for _ in range(0, (gl.cdiv(K, BLOCK_K) - 2) // 3):
        with gl.amd.warp_pipeline_stage("stage0", priority=0):
            a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(
                a_desc, b_desc, a_buf.index(2), b_buf.index(2), BLOCK_K)
            a, b = _bf16_kernelc_3pf_load_slot(a_buf, b_buf, 0, dot_a, dot_b)
        with gl.amd.warp_pipeline_stage("stage1", priority=1):
            acc = gl.amd.gfx1250.wmma(a, b, acc)
        _bf16_cluster_4x4_wait(2)

        with gl.amd.warp_pipeline_stage("stage0", priority=0):
            a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(
                a_desc, b_desc, a_buf.index(0), b_buf.index(0), BLOCK_K)
            a, b = _bf16_kernelc_3pf_load_slot(a_buf, b_buf, 1, dot_a, dot_b)
        with gl.amd.warp_pipeline_stage("stage1", priority=1):
            acc = gl.amd.gfx1250.wmma(a, b, acc)
        _bf16_cluster_4x4_wait(2)

        with gl.amd.warp_pipeline_stage("stage0", priority=0):
            a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(
                a_desc, b_desc, a_buf.index(1), b_buf.index(1), BLOCK_K)
            a, b = _bf16_kernelc_3pf_load_slot(a_buf, b_buf, 2, dot_a, dot_b)
        with gl.amd.warp_pipeline_stage("stage1", priority=1):
            acc = gl.amd.gfx1250.wmma(a, b, acc)
        _bf16_cluster_4x4_wait(2)

    phase = ((gl.cdiv(K, BLOCK_K) - 2) // 3) * 3
    for _ in range(0, (gl.cdiv(K, BLOCK_K) - 2) % 3):
        with gl.amd.warp_pipeline_stage("stage0", priority=0):
            write_phase = phase + 2
            phase, a, b = _bf16_kernelc_3pf_load_dynamic(phase, a_buf, b_buf, dot_a, dot_b)
            a_desc, b_desc = _bf16_kernelc_3pf_issue_loads(
                a_desc, b_desc, a_buf.index(write_phase % 3), b_buf.index(write_phase % 3), BLOCK_K)
        with gl.amd.warp_pipeline_stage("stage1", priority=1):
            acc = gl.amd.gfx1250.wmma(a, b, acc)
        _bf16_cluster_4x4_wait(2)

    for _ in gl.static_range(2):
        with gl.amd.warp_pipeline_stage("stage0", priority=0):
            phase, a, b = _bf16_kernelc_3pf_load_dynamic(phase, a_buf, b_buf, dot_a, dot_b)
        with gl.amd.warp_pipeline_stage("stage1", priority=1):
            acc = gl.amd.gfx1250.wmma(a, b, acc)
        _bf16_cluster_4x4_wait(0)

    c_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0], WMMA_LAYOUT.cga_layout)
    c_shared = gl.allocate_shared_memory(c_ptr.type.element_ty, [BLOCK_M, BLOCK_N], c_shared_layout)
    c_shared.store(acc.to(c_ptr.type.element_ty))
    c_desc = tdm.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                        block_shape=(BLOCK_M, BLOCK_N), layout=c_shared_layout)
    tdm.async_store(c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], c_shared)
    tdm.async_wait(0)


@gluon.jit
def _consume_bf16_slice_mnk_tile(a_buf, b_buf, slot, acc_tl, acc_bl, acc_tr, acc_br, DOT_A: gl.constexpr,
                                 DOT_B: gl.constexpr, SUBTILE_M: gl.constexpr, SUBTILE_N: gl.constexpr,
                                 SUBTILE_K: gl.constexpr):
    with gl.amd.warp_pipeline_stage("mem", priority=1):
        a00 = a_buf.index(slot).slice(0, SUBTILE_M, 0).slice(0, SUBTILE_K, 1).load(layout=DOT_A)
        b00 = b_buf.index(slot).slice(0, SUBTILE_N, 0).slice(
            0, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_tl = gl.amd.gfx1250.wmma(a00, b00, acc_tl)

    with gl.amd.warp_pipeline_stage("mem", priority=1):
        b01 = b_buf.index(slot).slice(SUBTILE_N, SUBTILE_N, 0).slice(
            0, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_tr = gl.amd.gfx1250.wmma(a00, b01, acc_tr)

    with gl.amd.warp_pipeline_stage("mem", priority=1):
        a10 = a_buf.index(slot).slice(SUBTILE_M, SUBTILE_M, 0).slice(0, SUBTILE_K, 1).load(layout=DOT_A)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_bl = gl.amd.gfx1250.wmma(a10, b00, acc_bl)

    with gl.amd.warp_pipeline_stage("mem", priority=1):
        b10 = b_buf.index(slot).slice(0, SUBTILE_N, 0).slice(
            SUBTILE_K, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_br = gl.amd.gfx1250.wmma(a10, b01, acc_br)

    with gl.amd.warp_pipeline_stage("mem", priority=1):
        a01 = a_buf.index(slot).slice(0, SUBTILE_M, 0).slice(SUBTILE_K, SUBTILE_K, 1).load(layout=DOT_A)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_tl = gl.amd.gfx1250.wmma(a01, b10, acc_tl)

    with gl.amd.warp_pipeline_stage("mem", priority=1):
        b11 = b_buf.index(slot).slice(SUBTILE_N, SUBTILE_N, 0).slice(
            SUBTILE_K, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_tr = gl.amd.gfx1250.wmma(a01, b11, acc_tr)

    with gl.amd.warp_pipeline_stage("mem", priority=1):
        a11 = a_buf.index(slot).slice(SUBTILE_M, SUBTILE_M, 0).slice(
            SUBTILE_K, SUBTILE_K, 1).load(layout=DOT_A)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_bl = gl.amd.gfx1250.wmma(a11, b10, acc_bl)
        acc_br = gl.amd.gfx1250.wmma(a11, b11, acc_br)
    return acc_tl, acc_bl, acc_tr, acc_br


@gluon.jit
def bf16_slice_mnk_warp_pipeline_kernel_gfx1250(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk,
                                                stride_bn, stride_cm, stride_cn, BLOCK_M: gl.constexpr,
                                                BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                                                GROUP_SIZE_M: gl.constexpr, GRID_MN: gl.constexpr,
                                                NUM_XCDS: gl.constexpr, NUM_WARPS: gl.constexpr,
                                                NUM_BUFFERS: gl.constexpr):
    gl.static_assert(a_ptr.type.element_ty.is_bf16() and b_ptr.type.element_ty.is_bf16(),
                     "bf16_slice_mnk_warp_pipeline_kernel_gfx1250 requires BF16 inputs")
    gl.static_assert(BLOCK_M == 256 and BLOCK_N == 256)
    gl.static_assert((BLOCK_K == 128 and NUM_BUFFERS == 2) or (BLOCK_K == 64 and NUM_BUFFERS == 3))
    gl.static_assert(NUM_WARPS == 8)
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    subtile_m: gl.constexpr = BLOCK_M // 2
    subtile_n: gl.constexpr = BLOCK_N // 2
    subtile_k: gl.constexpr = BLOCK_K // 2
    padded_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0])
    padded_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_N, BLOCK_K], [1, 0])
    layouts: gl.constexpr = gl.amd.gfx1250.make_partitioned_dot_layouts(
        BLOCK_M, BLOCK_N, padded_a, padded_b, NUM_WARPS, [16, 16, 32], a_transposed=False, b_transposed=True,
        slice_m=subtile_m, slice_n=subtile_n)
    shared_a: gl.constexpr = layouts[0]
    shared_b: gl.constexpr = layouts[1]
    wmma: gl.constexpr = layouts[2]
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma, 8)

    nbuf: gl.constexpr = NUM_BUFFERS
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, BLOCK_M, BLOCK_K], shared_a)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, BLOCK_N, BLOCK_K], shared_b)

    a_base = pid_m * BLOCK_M * stride_am
    b_base = pid_n * BLOCK_N * stride_bn
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base, shape=(M, K), strides=(stride_am, stride_ak),
                                        block_shape=(BLOCK_M, BLOCK_K), layout=shared_a)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base, shape=(N, K), strides=(stride_bn, stride_bk),
                                        block_shape=(BLOCK_N, BLOCK_K), layout=shared_b)

    for prefetch_idx in gl.static_range(NUM_BUFFERS):
        tdm.async_load(a_desc, [0, prefetch_idx * BLOCK_K], a_buf.index(prefetch_idx),
                       warp_used_hint=0b00001111)
        tdm.async_load(b_desc, [0, prefetch_idx * BLOCK_K], b_buf.index(prefetch_idx),
                       warp_used_hint=0b00001111)
    tdm.async_wait(2 * (NUM_BUFFERS - 1))

    acc_tl = gl.zeros((subtile_m, subtile_n), dtype=gl.float32, layout=wmma)
    acc_bl = gl.zeros((subtile_m, subtile_n), dtype=gl.float32, layout=wmma)
    acc_tr = gl.zeros((subtile_m, subtile_n), dtype=gl.float32, layout=wmma)
    acc_br = gl.zeros((subtile_m, subtile_n), dtype=gl.float32, layout=wmma)

    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max >= NUM_BUFFERS)
    for tile_idx in range(0, iter_max - NUM_BUFFERS):
        slot = tile_idx % nbuf

        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a00 = a_buf.index(slot).slice(0, subtile_m, 0).slice(0, subtile_k, 1).load(layout=dot_a)
            b00 = b_buf.index(slot).slice(0, subtile_n, 0).slice(
                0, subtile_k, 1).permute([1, 0]).load(layout=dot_b)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.gfx1250.wmma(a00, b00, acc_tl)

        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b01 = b_buf.index(slot).slice(subtile_n, subtile_n, 0).slice(
                0, subtile_k, 1).permute([1, 0]).load(layout=dot_b)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.gfx1250.wmma(a00, b01, acc_tr)

        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a10 = a_buf.index(slot).slice(subtile_m, subtile_m, 0).slice(0, subtile_k, 1).load(layout=dot_a)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.gfx1250.wmma(a10, b00, acc_bl)

        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b10 = b_buf.index(slot).slice(0, subtile_n, 0).slice(
                subtile_k, subtile_k, 1).permute([1, 0]).load(layout=dot_b)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.gfx1250.wmma(a10, b01, acc_br)

        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a01 = a_buf.index(slot).slice(0, subtile_m, 0).slice(
                subtile_k, subtile_k, 1).load(layout=dot_a)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.gfx1250.wmma(a01, b10, acc_tl)

        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b11 = b_buf.index(slot).slice(subtile_n, subtile_n, 0).slice(
                subtile_k, subtile_k, 1).permute([1, 0]).load(layout=dot_b)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.gfx1250.wmma(a01, b11, acc_tr)

        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a11 = a_buf.index(slot).slice(subtile_m, subtile_m, 0).slice(
                subtile_k, subtile_k, 1).load(layout=dot_a)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.gfx1250.wmma(a11, b10, acc_bl)
            acc_br = gl.amd.gfx1250.wmma(a11, b11, acc_br)

        refill_idx = tile_idx + NUM_BUFFERS
        with gl.amd.warp_pipeline_stage("tdm", priority=1):
            tdm.async_load(a_desc, [0, refill_idx * BLOCK_K], a_buf.index(slot), warp_used_hint=0b00001111)
            tdm.async_load(b_desc, [0, refill_idx * BLOCK_K], b_buf.index(slot), warp_used_hint=0b00001111)
        tdm.async_wait(2 * (NUM_BUFFERS - 1))

    for tail_idx in gl.static_range(NUM_BUFFERS):
        tile_idx = iter_max - NUM_BUFFERS + tail_idx
        acc_tl, acc_bl, acc_tr, acc_br = _consume_bf16_slice_mnk_tile(
            a_buf, b_buf, tile_idx % nbuf, acc_tl, acc_bl, acc_tr, acc_br, dot_a, dot_b, subtile_m, subtile_n,
            subtile_k)
        if tail_idx < NUM_BUFFERS - 1:
            tdm.async_wait(2 * (NUM_BUFFERS - 2 - tail_idx))

    _store_quadrants(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc_tl, acc_bl, acc_tr, acc_br, wmma,
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
                                              SCALE_PRESHUFFLE: gl.constexpr, ASYNC_COPY_SCALE: gl.constexpr,
                                              USE_WARP_PIPELINE: gl.constexpr):
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
        if USE_WARP_PIPELINE:
            with gl.amd.warp_pipeline_stage("mfma", priority=0):
                acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_left, bs_left, DTYPE_B, acc_tl)
        else:
            acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_left, bs_left, DTYPE_B, acc_tl)
        _wait_mxfp_scale_pipeline(steady_wait, ASYNC_COPY_SCALE)
        if USE_WARP_PIPELINE:
            with gl.amd.warp_pipeline_stage("mem", priority=1):
                a_bot = a_bot_buf.index(read_slot).load(layout=dot_a)
                if WITH_A_SCALE:
                    as_bot = _load_mxfp_scale(as_bot_buf, read_slot, scale_a_layout, HALF_M, BK_SCALE,
                                              SCALE_PRESHUFFLE, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
                else:
                    as_bot = as_top
                tdm.async_load(b_left_desc, [0, load_k * BK_B], b_left_buf.index(write_slot))
                _issue_mxfp_scale_load(bs_left_desc, bs_left_ptrs, load_k, bs_left_buf, write_slot,
                                       BK_SCALE_PRESHUFFLED, ASYNC_COPY_SCALE)
        else:
            a_bot = a_bot_buf.index(read_slot).load(layout=dot_a)
            if WITH_A_SCALE:
                as_bot = _load_mxfp_scale(as_bot_buf, read_slot, scale_a_layout, HALF_M, BK_SCALE, SCALE_PRESHUFFLE,
                                          PRESHUFFLE_FACTOR, SCALE_KWIDTH)
            else:
                as_bot = as_top
            tdm.async_load(b_left_desc, [0, load_k * BK_B], b_left_buf.index(write_slot))
            _issue_mxfp_scale_load(bs_left_desc, bs_left_ptrs, load_k, bs_left_buf, write_slot, BK_SCALE_PRESHUFFLED,
                                   ASYNC_COPY_SCALE)

        if USE_WARP_PIPELINE:
            with gl.amd.warp_pipeline_stage("mfma", priority=0):
                acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_left, bs_left, DTYPE_B, acc_bl)
        else:
            acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_left, bs_left, DTYPE_B, acc_bl)
        _wait_mxfp_scale_pipeline(steady_wait, ASYNC_COPY_SCALE)
        if USE_WARP_PIPELINE:
            with gl.amd.warp_pipeline_stage("mem", priority=1):
                b_right = b_right_buf.index(read_slot).permute([1, 0]).load(layout=dot_b)
                bs_right = _load_mxfp_scale(bs_right_buf, read_slot, scale_b_layout, HALF_N, BK_SCALE,
                                            SCALE_PRESHUFFLE, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
                tdm.async_load(a_top_desc, [0, load_k * BK_A], a_top_buf.index(write_slot))
                if WITH_A_SCALE:
                    _issue_mxfp_scale_load(as_top_desc, as_top_ptrs, load_k, as_top_buf, write_slot,
                                           BK_SCALE_PRESHUFFLED, ASYNC_COPY_SCALE)
        else:
            b_right = b_right_buf.index(read_slot).permute([1, 0]).load(layout=dot_b)
            bs_right = _load_mxfp_scale(bs_right_buf, read_slot, scale_b_layout, HALF_N, BK_SCALE, SCALE_PRESHUFFLE,
                                        PRESHUFFLE_FACTOR, SCALE_KWIDTH)
            tdm.async_load(a_top_desc, [0, load_k * BK_A], a_top_buf.index(write_slot))
            if WITH_A_SCALE:
                _issue_mxfp_scale_load(as_top_desc, as_top_ptrs, load_k, as_top_buf, write_slot,
                                       BK_SCALE_PRESHUFFLED, ASYNC_COPY_SCALE)

        if USE_WARP_PIPELINE:
            with gl.amd.warp_pipeline_stage("mfma", priority=0):
                acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_right, bs_right, DTYPE_B, acc_tr)
        else:
            acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_right, bs_right, DTYPE_B, acc_tr)
        _wait_mxfp_scale_pipeline(steady_wait, ASYNC_COPY_SCALE)
        if USE_WARP_PIPELINE:
            with gl.amd.warp_pipeline_stage("mem", priority=1):
                b_left = b_left_buf.index(next_slot).permute([1, 0]).load(layout=dot_b)
                bs_left = _load_mxfp_scale(bs_left_buf, next_slot, scale_b_layout, HALF_N, BK_SCALE,
                                           SCALE_PRESHUFFLE, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
                tdm.async_load(a_bot_desc, [0, load_k * BK_A], a_bot_buf.index(write_slot))
                if WITH_A_SCALE:
                    _issue_mxfp_scale_load(as_bot_desc, as_bot_ptrs, load_k, as_bot_buf, write_slot,
                                           BK_SCALE_PRESHUFFLED, ASYNC_COPY_SCALE)
        else:
            b_left = b_left_buf.index(next_slot).permute([1, 0]).load(layout=dot_b)
            bs_left = _load_mxfp_scale(bs_left_buf, next_slot, scale_b_layout, HALF_N, BK_SCALE, SCALE_PRESHUFFLE,
                                       PRESHUFFLE_FACTOR, SCALE_KWIDTH)
            tdm.async_load(a_bot_desc, [0, load_k * BK_A], a_bot_buf.index(write_slot))
            if WITH_A_SCALE:
                _issue_mxfp_scale_load(as_bot_desc, as_bot_ptrs, load_k, as_bot_buf, write_slot,
                                       BK_SCALE_PRESHUFFLED, ASYNC_COPY_SCALE)

        if USE_WARP_PIPELINE:
            with gl.amd.warp_pipeline_stage("mfma", priority=0):
                acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_right, bs_right, DTYPE_B, acc_br)
        else:
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_right, bs_right, DTYPE_B, acc_br)
        _wait_mxfp_scale_pipeline(steady_wait, ASYNC_COPY_SCALE)
        if USE_WARP_PIPELINE:
            with gl.amd.warp_pipeline_stage("mem", priority=1):
                a_top = a_top_buf.index(next_slot).load(layout=dot_a)
                if WITH_A_SCALE:
                    as_top = _load_mxfp_scale(as_top_buf, next_slot, scale_a_layout, HALF_M, BK_SCALE,
                                              SCALE_PRESHUFFLE, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
                tdm.async_load(b_right_desc, [0, load_k * BK_B], b_right_buf.index(write_slot))
                _issue_mxfp_scale_load(bs_right_desc, bs_right_ptrs, load_k, bs_right_buf, write_slot,
                                       BK_SCALE_PRESHUFFLED, ASYNC_COPY_SCALE)
        else:
            a_top = a_top_buf.index(next_slot).load(layout=dot_a)
            if WITH_A_SCALE:
                as_top = _load_mxfp_scale(as_top_buf, next_slot, scale_a_layout, HALF_M, BK_SCALE, SCALE_PRESHUFFLE,
                                          PRESHUFFLE_FACTOR, SCALE_KWIDTH)
            tdm.async_load(b_right_desc, [0, load_k * BK_B], b_right_buf.index(write_slot))
            _issue_mxfp_scale_load(bs_right_desc, bs_right_ptrs, load_k, bs_right_buf, write_slot,
                                   BK_SCALE_PRESHUFFLED, ASYNC_COPY_SCALE)
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


@gluon.jit
def fp8_slice_mn_warp_pipeline_kernel_gfx1250(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk,
                                             stride_bn, stride_cm, stride_cn, DTYPE_A: gl.constexpr,
                                             DTYPE_B: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                                             BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
                                             GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr,
                                             NUM_WARPS: gl.constexpr):
    gl.static_assert(DTYPE_A != "e2m1" and DTYPE_B != "e2m1",
                     "fp8_slice_mn_warp_pipeline_kernel_gfx1250 requires FP8 inputs")
    gl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 128)
    gl.static_assert(NUM_WARPS == 8)
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    half_m: gl.constexpr = BLOCK_M // 2
    half_n: gl.constexpr = BLOCK_N // 2
    padded_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 16]], [half_m, BLOCK_K],
                                                                    [1, 0])
    padded_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 16]], [half_n, BLOCK_K],
                                                                    [1, 0])
    layouts: gl.constexpr = gl.amd.gfx1250.make_partitioned_dot_layouts(
        half_m, half_n, padded_a, padded_b, NUM_WARPS, [16, 16, 128], a_transposed=False, b_transposed=True)
    shared_a: gl.constexpr = layouts[0]
    shared_b: gl.constexpr = layouts[1]
    wmma: gl.constexpr = layouts[2]
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma, 16)

    nbuf: gl.constexpr = 2
    a_top_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, half_m, BLOCK_K], shared_a)
    a_bot_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, half_m, BLOCK_K], shared_a)
    b_left_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, half_n, BLOCK_K], shared_b)
    b_right_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, half_n, BLOCK_K], shared_b)

    a_base = pid_m * BLOCK_M * stride_am
    b_base = pid_n * BLOCK_N * stride_bn
    a_top_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base, shape=(M, K), strides=(stride_am, stride_ak),
                                            block_shape=(half_m, BLOCK_K), layout=shared_a)
    a_bot_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base + half_m * stride_am, shape=(M, K),
                                            strides=(stride_am, stride_ak), block_shape=(half_m, BLOCK_K),
                                            layout=shared_a)
    b_left_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base, shape=(N, K), strides=(stride_bn, stride_bk),
                                             block_shape=(half_n, BLOCK_K), layout=shared_b)
    b_right_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base + half_n * stride_bn, shape=(N, K),
                                              strides=(stride_bn, stride_bk), block_shape=(half_n, BLOCK_K),
                                              layout=shared_b)

    acc_tl = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    acc_bl = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    acc_tr = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    acc_br = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)

    iter_max = gl.cdiv(K, BLOCK_K)

    # Prologue: K-steps 0 and 1 occupy fixed LDS slots 0 and 1.
    tdm.async_load(b_left_desc, [0, 0], b_left_buf.index(0))
    tdm.async_load(a_top_desc, [0, 0], a_top_buf.index(0))
    tdm.async_load(a_bot_desc, [0, 0], a_bot_buf.index(0))
    tdm.async_load(b_right_desc, [0, 0], b_right_buf.index(0))
    tdm.async_load(b_left_desc, [0, BLOCK_K], b_left_buf.index(1))
    tdm.async_load(a_top_desc, [0, BLOCK_K], a_top_buf.index(1))
    tdm.async_load(a_bot_desc, [0, BLOCK_K], a_bot_buf.index(1))
    tdm.async_load(b_right_desc, [0, BLOCK_K], b_right_buf.index(1))

    tdm.async_wait(6)
    b_left = b_left_buf.index(0).permute([1, 0]).load(layout=dot_b)
    a_top = a_top_buf.index(0).load(layout=dot_a)
    gl.assume(iter_max > 3)

    # Tutorial schedule: two K-steps and eight MFMA/memory stage pairs per loop.
    for k in range(0, iter_max - 2, 2):
        tdm.async_wait(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, None, DTYPE_A, b_left, None, DTYPE_B, acc_tl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = a_bot_buf.index(0).load(layout=dot_a)
            tdm.async_load(b_left_desc, [0, (k + 2) * BLOCK_K], b_left_buf.index(0))

        tdm.async_wait(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, None, DTYPE_A, b_left, None, DTYPE_B, acc_bl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = b_right_buf.index(0).permute([1, 0]).load(layout=dot_b)
            tdm.async_load(a_top_desc, [0, (k + 2) * BLOCK_K], a_top_buf.index(0))

        tdm.async_wait(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, None, DTYPE_A, b_right, None, DTYPE_B, acc_tr)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = b_left_buf.index(1).permute([1, 0]).load(layout=dot_b)
            tdm.async_load(a_bot_desc, [0, (k + 2) * BLOCK_K], a_bot_buf.index(0))

        tdm.async_wait(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, None, DTYPE_A, b_right, None, DTYPE_B, acc_br)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = a_top_buf.index(1).load(layout=dot_a)
            tdm.async_load(b_right_desc, [0, (k + 2) * BLOCK_K], b_right_buf.index(0))

        tdm.async_wait(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, None, DTYPE_A, b_left, None, DTYPE_B, acc_tl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = a_bot_buf.index(1).load(layout=dot_a)
            tdm.async_load(b_left_desc, [0, (k + 3) * BLOCK_K], b_left_buf.index(1))

        tdm.async_wait(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, None, DTYPE_A, b_left, None, DTYPE_B, acc_bl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = b_right_buf.index(1).permute([1, 0]).load(layout=dot_b)
            tdm.async_load(a_top_desc, [0, (k + 3) * BLOCK_K], a_top_buf.index(1))

        tdm.async_wait(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, None, DTYPE_A, b_right, None, DTYPE_B, acc_tr)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = b_left_buf.index(0).permute([1, 0]).load(layout=dot_b)
            tdm.async_load(a_bot_desc, [0, (k + 3) * BLOCK_K], a_bot_buf.index(1))

        tdm.async_wait(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, None, DTYPE_A, b_right, None, DTYPE_B, acc_br)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = a_top_buf.index(0).load(layout=dot_a)
            tdm.async_load(b_right_desc, [0, (k + 3) * BLOCK_K], b_right_buf.index(1))

    # Drain the final two prefetched K-steps.
    acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, None, DTYPE_A, b_left, None, DTYPE_B, acc_tl)
    tdm.async_wait(5)
    l_idx = (iter_max - 2) % 2
    a_bot = a_bot_buf.index(l_idx).load(layout=dot_a)

    acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, None, DTYPE_A, b_left, None, DTYPE_B, acc_bl)
    tdm.async_wait(4)
    b_right = b_right_buf.index(l_idx).permute([1, 0]).load(layout=dot_b)

    acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, None, DTYPE_A, b_right, None, DTYPE_B, acc_tr)
    tdm.async_wait(3)
    g_idx = 1 - l_idx
    b_left = b_left_buf.index(g_idx).permute([1, 0]).load(layout=dot_b)

    acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, None, DTYPE_A, b_right, None, DTYPE_B, acc_br)
    tdm.async_wait(2)
    a_top = a_top_buf.index(g_idx).load(layout=dot_a)

    acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, None, DTYPE_A, b_left, None, DTYPE_B, acc_tl)
    tdm.async_wait(1)
    a_bot = a_bot_buf.index(g_idx).load(layout=dot_a)

    acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, None, DTYPE_A, b_left, None, DTYPE_B, acc_bl)
    tdm.async_wait(0)
    b_right = b_right_buf.index(g_idx).permute([1, 0]).load(layout=dot_b)

    acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, None, DTYPE_A, b_right, None, DTYPE_B, acc_tr)
    acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, None, DTYPE_A, b_right, None, DTYPE_B, acc_br)

    _store_quadrants(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc_tl, acc_bl, acc_tr, acc_br, wmma,
                     BLOCK_M, BLOCK_N)


@gluon.jit
def _fp8_scaled_slice_mn_warp_pipeline_bk128_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
        stride_cn, stride_scale, DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr, SCALE_BLOCK: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
        GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr, NUM_WARPS: gl.constexpr, SCALE_PRESHUFFLE: gl.constexpr,
        ASYNC_COPY_SCALE: gl.constexpr):
    gl.static_assert(DTYPE_A != "e2m1" and DTYPE_B != "e2m1",
                     "fp8_scaled_slice_mn_warp_pipeline_kernel_gfx1250 requires FP8 inputs")
    gl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 128)
    gl.static_assert(BLOCK_K % SCALE_BLOCK == 0)
    gl.static_assert(NUM_WARPS == 8)
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    half_m: gl.constexpr = BLOCK_M // 2
    half_n: gl.constexpr = BLOCK_N // 2
    bk_scale: gl.constexpr = BLOCK_K // SCALE_BLOCK
    preshuffle_factor: gl.constexpr = 128 if SCALE_PRESHUFFLE else 1
    bk_scale_preshuffled: gl.constexpr = bk_scale * preshuffle_factor
    half_m_preshuffled: gl.constexpr = half_m // preshuffle_factor
    half_n_preshuffled: gl.constexpr = half_n // preshuffle_factor
    scale_kwidth: gl.constexpr = 4 if bk_scale >= 4 else bk_scale

    padded_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 16]], [half_m, BLOCK_K],
                                                                    [1, 0])
    padded_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 16]], [half_n, BLOCK_K],
                                                                    [1, 0])
    shared_scale: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [half_m_preshuffled, bk_scale_preshuffled], [1, 0])
    layouts: gl.constexpr = gl.amd.gfx1250.make_partitioned_dot_layouts(
        half_m, half_n, padded_a, padded_b, NUM_WARPS, [16, 16, 128], a_transposed=False, b_transposed=True)
    shared_a: gl.constexpr = layouts[0]
    shared_b: gl.constexpr = layouts[1]
    wmma: gl.constexpr = layouts[2]
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_a, [half_m, bk_scale])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_b, [half_n, bk_scale])

    nbuf: gl.constexpr = 2
    a_top_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, half_m, BLOCK_K], shared_a)
    a_bot_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, half_m, BLOCK_K], shared_a)
    b_left_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, half_n, BLOCK_K], shared_b)
    b_right_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, half_n, BLOCK_K], shared_b)
    as_top_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty,
                                           [nbuf, half_m_preshuffled, bk_scale_preshuffled], shared_scale)
    as_bot_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty,
                                           [nbuf, half_m_preshuffled, bk_scale_preshuffled], shared_scale)
    bs_left_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty,
                                            [nbuf, half_n_preshuffled, bk_scale_preshuffled], shared_scale)
    bs_right_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty,
                                             [nbuf, half_n_preshuffled, bk_scale_preshuffled], shared_scale)

    a_base = pid_m * BLOCK_M * stride_am
    b_base = pid_n * BLOCK_N * stride_bn
    a_top_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base, shape=(M, K), strides=(stride_am, stride_ak),
                                            block_shape=(half_m, BLOCK_K), layout=shared_a)
    a_bot_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base + half_m * stride_am, shape=(M, K),
                                            strides=(stride_am, stride_ak), block_shape=(half_m, BLOCK_K),
                                            layout=shared_a)
    b_left_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base, shape=(N, K), strides=(stride_bn, stride_bk),
                                             block_shape=(half_n, BLOCK_K), layout=shared_b)
    b_right_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base + half_n * stride_bn, shape=(N, K),
                                              strides=(stride_bn, stride_bk), block_shape=(half_n, BLOCK_K),
                                              layout=shared_b)

    as_top_base = (pid_m * BLOCK_M) // preshuffle_factor * stride_scale
    as_bot_base = (pid_m * BLOCK_M + half_m) // preshuffle_factor * stride_scale
    bs_left_base = (pid_n * BLOCK_N) // preshuffle_factor * stride_scale
    bs_right_base = (pid_n * BLOCK_N + half_n) // preshuffle_factor * stride_scale
    as_top_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + as_top_base, shape=(M // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(half_m_preshuffled, bk_scale_preshuffled), layout=shared_scale)
    as_bot_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + as_bot_base, shape=(M // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(half_m_preshuffled, bk_scale_preshuffled), layout=shared_scale)
    bs_left_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + bs_left_base, shape=(N // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(half_n_preshuffled, bk_scale_preshuffled), layout=shared_scale)
    bs_right_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + bs_right_base, shape=(N // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(half_n_preshuffled, bk_scale_preshuffled), layout=shared_scale)

    scale_copy_layout: gl.constexpr = get_scale_blocked_layout(NUM_WARPS)
    scale_m = gl.arange(0, half_m_preshuffled, layout=gl.SliceLayout(1, scale_copy_layout))
    scale_n = gl.arange(0, half_n_preshuffled, layout=gl.SliceLayout(1, scale_copy_layout))
    scale_k = gl.arange(0, bk_scale_preshuffled, layout=gl.SliceLayout(0, scale_copy_layout))
    as_top_ptrs = a_scale_ptr + as_top_base + scale_m[:, None] * stride_scale + scale_k[None, :]
    as_bot_ptrs = a_scale_ptr + as_bot_base + scale_m[:, None] * stride_scale + scale_k[None, :]
    bs_left_ptrs = b_scale_ptr + bs_left_base + scale_n[:, None] * stride_scale + scale_k[None, :]
    bs_right_ptrs = b_scale_ptr + bs_right_base + scale_n[:, None] * stride_scale + scale_k[None, :]

    for i in gl.static_range(2):
        tdm.async_load(b_left_desc, [0, i * BLOCK_K], b_left_buf.index(i))
        _issue_mxfp_scale_load(bs_left_desc, bs_left_ptrs, i, bs_left_buf, i, bk_scale_preshuffled,
                               ASYNC_COPY_SCALE)
        tdm.async_load(a_top_desc, [0, i * BLOCK_K], a_top_buf.index(i))
        _issue_mxfp_scale_load(as_top_desc, as_top_ptrs, i, as_top_buf, i, bk_scale_preshuffled, ASYNC_COPY_SCALE)
        tdm.async_load(a_bot_desc, [0, i * BLOCK_K], a_bot_buf.index(i))
        _issue_mxfp_scale_load(as_bot_desc, as_bot_ptrs, i, as_bot_buf, i, bk_scale_preshuffled, ASYNC_COPY_SCALE)
        tdm.async_load(b_right_desc, [0, i * BLOCK_K], b_right_buf.index(i))
        _issue_mxfp_scale_load(bs_right_desc, bs_right_ptrs, i, bs_right_buf, i, bk_scale_preshuffled,
                               ASYNC_COPY_SCALE)

    _wait_mxfp_scale_pipeline(12, ASYNC_COPY_SCALE)
    b_left = b_left_buf.index(0).permute([1, 0]).load(layout=dot_b)
    bs_left = _load_mxfp_scale(bs_left_buf, 0, scale_b_layout, half_n, bk_scale, SCALE_PRESHUFFLE,
                               preshuffle_factor, scale_kwidth)
    a_top = a_top_buf.index(0).load(layout=dot_a)
    as_top = _load_mxfp_scale(as_top_buf, 0, scale_a_layout, half_m, bk_scale, SCALE_PRESHUFFLE,
                              preshuffle_factor, scale_kwidth)

    acc_tl = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    acc_bl = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    acc_tr = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    acc_br = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max > 3)

    for k in range(0, iter_max - 2, 2):
        _wait_fp8_scaled_warp_pipeline(10, ASYNC_COPY_SCALE)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_left, bs_left, DTYPE_B, acc_tl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = a_bot_buf.index(0).load(layout=dot_a)
            as_bot = _load_mxfp_scale(as_bot_buf, 0, scale_a_layout, half_m, bk_scale, SCALE_PRESHUFFLE,
                                      preshuffle_factor, scale_kwidth)
            tdm.async_load(b_left_desc, [0, (k + 2) * BLOCK_K], b_left_buf.index(0))
            _issue_mxfp_scale_load(bs_left_desc, bs_left_ptrs, k + 2, bs_left_buf, 0, bk_scale_preshuffled,
                                   ASYNC_COPY_SCALE)

        _wait_fp8_scaled_warp_pipeline(10, ASYNC_COPY_SCALE)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_left, bs_left, DTYPE_B, acc_bl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = b_right_buf.index(0).permute([1, 0]).load(layout=dot_b)
            bs_right = _load_mxfp_scale(bs_right_buf, 0, scale_b_layout, half_n, bk_scale, SCALE_PRESHUFFLE,
                                        preshuffle_factor, scale_kwidth)
            tdm.async_load(a_top_desc, [0, (k + 2) * BLOCK_K], a_top_buf.index(0))
            _issue_mxfp_scale_load(as_top_desc, as_top_ptrs, k + 2, as_top_buf, 0, bk_scale_preshuffled,
                                   ASYNC_COPY_SCALE)

        _wait_fp8_scaled_warp_pipeline(10, ASYNC_COPY_SCALE)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_right, bs_right, DTYPE_B, acc_tr)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = b_left_buf.index(1).permute([1, 0]).load(layout=dot_b)
            bs_left = _load_mxfp_scale(bs_left_buf, 1, scale_b_layout, half_n, bk_scale, SCALE_PRESHUFFLE,
                                       preshuffle_factor, scale_kwidth)
            tdm.async_load(a_bot_desc, [0, (k + 2) * BLOCK_K], a_bot_buf.index(0))
            _issue_mxfp_scale_load(as_bot_desc, as_bot_ptrs, k + 2, as_bot_buf, 0, bk_scale_preshuffled,
                                   ASYNC_COPY_SCALE)

        _wait_fp8_scaled_warp_pipeline(10, ASYNC_COPY_SCALE)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_right, bs_right, DTYPE_B, acc_br)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = a_top_buf.index(1).load(layout=dot_a)
            as_top = _load_mxfp_scale(as_top_buf, 1, scale_a_layout, half_m, bk_scale, SCALE_PRESHUFFLE,
                                      preshuffle_factor, scale_kwidth)
            tdm.async_load(b_right_desc, [0, (k + 2) * BLOCK_K], b_right_buf.index(0))
            _issue_mxfp_scale_load(bs_right_desc, bs_right_ptrs, k + 2, bs_right_buf, 0, bk_scale_preshuffled,
                                   ASYNC_COPY_SCALE)

        _wait_fp8_scaled_warp_pipeline(10, ASYNC_COPY_SCALE)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_left, bs_left, DTYPE_B, acc_tl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = a_bot_buf.index(1).load(layout=dot_a)
            as_bot = _load_mxfp_scale(as_bot_buf, 1, scale_a_layout, half_m, bk_scale, SCALE_PRESHUFFLE,
                                      preshuffle_factor, scale_kwidth)
            tdm.async_load(b_left_desc, [0, (k + 3) * BLOCK_K], b_left_buf.index(1))
            _issue_mxfp_scale_load(bs_left_desc, bs_left_ptrs, k + 3, bs_left_buf, 1, bk_scale_preshuffled,
                                   ASYNC_COPY_SCALE)

        _wait_fp8_scaled_warp_pipeline(10, ASYNC_COPY_SCALE)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_left, bs_left, DTYPE_B, acc_bl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = b_right_buf.index(1).permute([1, 0]).load(layout=dot_b)
            bs_right = _load_mxfp_scale(bs_right_buf, 1, scale_b_layout, half_n, bk_scale, SCALE_PRESHUFFLE,
                                        preshuffle_factor, scale_kwidth)
            tdm.async_load(a_top_desc, [0, (k + 3) * BLOCK_K], a_top_buf.index(1))
            _issue_mxfp_scale_load(as_top_desc, as_top_ptrs, k + 3, as_top_buf, 1, bk_scale_preshuffled,
                                   ASYNC_COPY_SCALE)

        _wait_fp8_scaled_warp_pipeline(10, ASYNC_COPY_SCALE)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_right, bs_right, DTYPE_B, acc_tr)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = b_left_buf.index(0).permute([1, 0]).load(layout=dot_b)
            bs_left = _load_mxfp_scale(bs_left_buf, 0, scale_b_layout, half_n, bk_scale, SCALE_PRESHUFFLE,
                                       preshuffle_factor, scale_kwidth)
            tdm.async_load(a_bot_desc, [0, (k + 3) * BLOCK_K], a_bot_buf.index(1))
            _issue_mxfp_scale_load(as_bot_desc, as_bot_ptrs, k + 3, as_bot_buf, 1, bk_scale_preshuffled,
                                   ASYNC_COPY_SCALE)

        _wait_fp8_scaled_warp_pipeline(10, ASYNC_COPY_SCALE)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_right, bs_right, DTYPE_B, acc_br)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = a_top_buf.index(0).load(layout=dot_a)
            as_top = _load_mxfp_scale(as_top_buf, 0, scale_a_layout, half_m, bk_scale, SCALE_PRESHUFFLE,
                                      preshuffle_factor, scale_kwidth)
            tdm.async_load(b_right_desc, [0, (k + 3) * BLOCK_K], b_right_buf.index(1))
            _issue_mxfp_scale_load(bs_right_desc, bs_right_ptrs, k + 3, bs_right_buf, 1, bk_scale_preshuffled,
                                   ASYNC_COPY_SCALE)

    acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_left, bs_left, DTYPE_B, acc_tl)
    _wait_mxfp_scale_pipeline(10, ASYNC_COPY_SCALE)
    l_idx = (iter_max - 2) % 2
    a_bot = a_bot_buf.index(l_idx).load(layout=dot_a)
    as_bot = _load_mxfp_scale(as_bot_buf, l_idx, scale_a_layout, half_m, bk_scale, SCALE_PRESHUFFLE,
                              preshuffle_factor, scale_kwidth)

    acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_left, bs_left, DTYPE_B, acc_bl)
    _wait_mxfp_scale_pipeline(8, ASYNC_COPY_SCALE)
    b_right = b_right_buf.index(l_idx).permute([1, 0]).load(layout=dot_b)
    bs_right = _load_mxfp_scale(bs_right_buf, l_idx, scale_b_layout, half_n, bk_scale, SCALE_PRESHUFFLE,
                                preshuffle_factor, scale_kwidth)

    acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_right, bs_right, DTYPE_B, acc_tr)
    _wait_mxfp_scale_pipeline(6, ASYNC_COPY_SCALE)
    g_idx = 1 - l_idx
    b_left = b_left_buf.index(g_idx).permute([1, 0]).load(layout=dot_b)
    bs_left = _load_mxfp_scale(bs_left_buf, g_idx, scale_b_layout, half_n, bk_scale, SCALE_PRESHUFFLE,
                               preshuffle_factor, scale_kwidth)

    acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_right, bs_right, DTYPE_B, acc_br)
    _wait_mxfp_scale_pipeline(4, ASYNC_COPY_SCALE)
    a_top = a_top_buf.index(g_idx).load(layout=dot_a)
    as_top = _load_mxfp_scale(as_top_buf, g_idx, scale_a_layout, half_m, bk_scale, SCALE_PRESHUFFLE,
                              preshuffle_factor, scale_kwidth)

    acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_left, bs_left, DTYPE_B, acc_tl)
    _wait_mxfp_scale_pipeline(2, ASYNC_COPY_SCALE)
    a_bot = a_bot_buf.index(g_idx).load(layout=dot_a)
    as_bot = _load_mxfp_scale(as_bot_buf, g_idx, scale_a_layout, half_m, bk_scale, SCALE_PRESHUFFLE,
                              preshuffle_factor, scale_kwidth)

    acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_left, bs_left, DTYPE_B, acc_bl)
    _wait_mxfp_scale_pipeline(0, ASYNC_COPY_SCALE)
    b_right = b_right_buf.index(g_idx).permute([1, 0]).load(layout=dot_b)
    bs_right = _load_mxfp_scale(bs_right_buf, g_idx, scale_b_layout, half_n, bk_scale, SCALE_PRESHUFFLE,
                                preshuffle_factor, scale_kwidth)

    acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_right, bs_right, DTYPE_B, acc_tr)
    acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_right, bs_right, DTYPE_B, acc_br)
    _store_quadrants(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc_tl, acc_bl, acc_tr, acc_br, wmma,
                     BLOCK_M, BLOCK_N)


@gluon.jit
def _load_fp8_scaled_slice_mnk_scale(scale_buffer, slot, start_nonk: gl.constexpr, start_k: gl.constexpr,
                                     LAYOUT: gl.constexpr, BLOCK_NONK: gl.constexpr, BK_SCALE: gl.constexpr,
                                     SUBTILE_NONK: gl.constexpr, SUBTILE_SCALE_K: gl.constexpr,
                                     PRESHUFFLE_FACTOR: gl.constexpr, SCALE_KWIDTH: gl.constexpr):
    scale_slice = scale_buffer.index(slot).reshape(
        (BLOCK_NONK // PRESHUFFLE_FACTOR, BK_SCALE // SCALE_KWIDTH, PRESHUFFLE_FACTOR // 4, 4,
         SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((BLOCK_NONK, BK_SCALE))
    return scale_slice.slice(start_nonk, SUBTILE_NONK, 0).slice(
        start_k, SUBTILE_SCALE_K, 1).load(layout=LAYOUT)


@gluon.jit
def _load_fp8_scaled_slice_mnk_scale_hipblaslt(
        scale_buffer, slot, start_nonk: gl.constexpr, start_k: gl.constexpr, LAYOUT: gl.constexpr,
        BLOCK_NONK: gl.constexpr, BK_SCALE: gl.constexpr, SUBTILE_NONK: gl.constexpr,
        SUBTILE_SCALE_K: gl.constexpr, SCALE_KWIDTH: gl.constexpr):
    # hipBLASLt gfx1250 layout: [BK_SCALE / 4, BLOCK_NONK, 4].
    scale_slice = scale_buffer.index(slot).reshape(
        (BK_SCALE // SCALE_KWIDTH, BLOCK_NONK, SCALE_KWIDTH)).permute(
            (1, 0, 2)).reshape((BLOCK_NONK, BK_SCALE))
    return scale_slice.slice(start_nonk, SUBTILE_NONK, 0).slice(
        start_k, SUBTILE_SCALE_K, 1).load(layout=LAYOUT)


@gluon.jit
def _consume_fp8_scaled_slice_mnk_tile(
        a_buf, b_buf, as_buf, bs_buf, slot, acc_tl, acc_bl, acc_tr, acc_br, DTYPE_A: gl.constexpr,
        DTYPE_B: gl.constexpr, DOT_A: gl.constexpr, DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
        BK_SCALE: gl.constexpr, SUBTILE_M: gl.constexpr, SUBTILE_N: gl.constexpr, SUBTILE_K: gl.constexpr,
        SUBTILE_SCALE_K: gl.constexpr, PRESHUFFLE_FACTOR: gl.constexpr, SCALE_KWIDTH: gl.constexpr):
    with gl.amd.warp_pipeline_stage("mem", priority=1):
        a00 = a_buf.index(slot).slice(0, SUBTILE_M, 0).slice(0, SUBTILE_K, 1).load(layout=DOT_A)
        as00 = _load_fp8_scaled_slice_mnk_scale(as_buf, slot, 0, 0, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE,
                                               SUBTILE_M, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
        b00 = b_buf.index(slot).slice(0, SUBTILE_N, 0).slice(
            0, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
        bs00 = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, 0, 0, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE,
                                               SUBTILE_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_tl = gl.amd.gfx1250.wmma_scaled(a00, as00, DTYPE_A, b00, bs00, DTYPE_B, acc_tl)

    with gl.amd.warp_pipeline_stage("mem", priority=1):
        b01 = b_buf.index(slot).slice(SUBTILE_N, SUBTILE_N, 0).slice(
            0, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
        bs01 = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, SUBTILE_N, 0, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE,
                                               SUBTILE_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_tr = gl.amd.gfx1250.wmma_scaled(a00, as00, DTYPE_A, b01, bs01, DTYPE_B, acc_tr)

    with gl.amd.warp_pipeline_stage("mem", priority=1):
        a10 = a_buf.index(slot).slice(SUBTILE_M, SUBTILE_M, 0).slice(
            0, SUBTILE_K, 1).load(layout=DOT_A)
        as10 = _load_fp8_scaled_slice_mnk_scale(as_buf, slot, SUBTILE_M, 0, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE,
                                               SUBTILE_M, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_bl = gl.amd.gfx1250.wmma_scaled(a10, as10, DTYPE_A, b00, bs00, DTYPE_B, acc_bl)

    with gl.amd.warp_pipeline_stage("mem", priority=1):
        b10 = b_buf.index(slot).slice(0, SUBTILE_N, 0).slice(
            SUBTILE_K, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
        bs10 = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, 0, SUBTILE_SCALE_K, SCALE_B_LAYOUT, BLOCK_N,
                                               BK_SCALE, SUBTILE_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR,
                                               SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_br = gl.amd.gfx1250.wmma_scaled(a10, as10, DTYPE_A, b01, bs01, DTYPE_B, acc_br)

    with gl.amd.warp_pipeline_stage("mem", priority=1):
        a01 = a_buf.index(slot).slice(0, SUBTILE_M, 0).slice(
            SUBTILE_K, SUBTILE_K, 1).load(layout=DOT_A)
        as01 = _load_fp8_scaled_slice_mnk_scale(as_buf, slot, 0, SUBTILE_SCALE_K, SCALE_A_LAYOUT, BLOCK_M,
                                               BK_SCALE, SUBTILE_M, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR,
                                               SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_tl = gl.amd.gfx1250.wmma_scaled(a01, as01, DTYPE_A, b10, bs10, DTYPE_B, acc_tl)

    with gl.amd.warp_pipeline_stage("mem", priority=1):
        b11 = b_buf.index(slot).slice(SUBTILE_N, SUBTILE_N, 0).slice(
            SUBTILE_K, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
        bs11 = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, SUBTILE_N, SUBTILE_SCALE_K, SCALE_B_LAYOUT,
                                               BLOCK_N, BK_SCALE, SUBTILE_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR,
                                               SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_tr = gl.amd.gfx1250.wmma_scaled(a01, as01, DTYPE_A, b11, bs11, DTYPE_B, acc_tr)

    with gl.amd.warp_pipeline_stage("mem", priority=1):
        a11 = a_buf.index(slot).slice(SUBTILE_M, SUBTILE_M, 0).slice(
            SUBTILE_K, SUBTILE_K, 1).load(layout=DOT_A)
        as11 = _load_fp8_scaled_slice_mnk_scale(as_buf, slot, SUBTILE_M, SUBTILE_SCALE_K, SCALE_A_LAYOUT,
                                               BLOCK_M, BK_SCALE, SUBTILE_M, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR,
                                               SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_bl = gl.amd.gfx1250.wmma_scaled(a11, as11, DTYPE_A, b10, bs10, DTYPE_B, acc_bl)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_br = gl.amd.gfx1250.wmma_scaled(a11, as11, DTYPE_A, b11, bs11, DTYPE_B, acc_br)
    return acc_tl, acc_bl, acc_tr, acc_br


@gluon.jit
def _issue_fp8_scaled_slice_mnk_fused_scale_load(as_desc, bs_desc, as_buf, bs_buf, tile_idx, slot,
                                                 BK_SCALE_PRESHUFFLED: gl.constexpr):
    scale_k = tile_idx * BK_SCALE_PRESHUFFLED
    as_load_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, scale_k])
    bs_load_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, scale_k])
    tdm.async_load_fused([
        (as_load_desc, as_buf.index(slot), 0b00001111),
        (bs_load_desc, bs_buf.index(slot), 0b11110000),
    ])


@gluon.jit
def _issue_fp8_scaled_slice_mnk_fused_scale_load_hipblaslt(
        as_desc, bs_desc, as_buf, bs_buf, tile_idx, slot, BK_SCALE: gl.constexpr,
        SCALE_KWIDTH: gl.constexpr):
    scale_tile = tile_idx * (BK_SCALE // SCALE_KWIDTH)
    as_load_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[scale_tile, 0])
    bs_load_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[scale_tile, 0])
    tdm.async_load_fused([
        (as_load_desc, as_buf.index(slot), 0b00001111),
        (bs_load_desc, bs_buf.index(slot), 0b11110000),
    ])


@gluon.jit
def _issue_fp8_scaled_slice_mnk_leading4_fused_scale_load(
        as_desc, bs_desc, as_buf, bs_buf, tile_idx, slot, BK_SCALE_PRESHUFFLED: gl.constexpr):
    scale_k = tile_idx * BK_SCALE_PRESHUFFLED
    as_load_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, scale_k])
    bs_load_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, scale_k])
    tdm.async_load_fused([
        (as_load_desc, as_buf.index(slot), 0b00000011),
        (bs_load_desc, bs_buf.index(slot), 0b00001100),
    ])


@gluon.jit
def _issue_fp8_scaled_slice_mnk_leading4_fused_scale_load_hipblaslt(
        as_desc, bs_desc, as_buf, bs_buf, tile_idx, slot, BK_SCALE: gl.constexpr,
        SCALE_KWIDTH: gl.constexpr):
    scale_tile = tile_idx * (BK_SCALE // SCALE_KWIDTH)
    as_load_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[scale_tile, 0])
    bs_load_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[scale_tile, 0])
    tdm.async_load_fused([
        (as_load_desc, as_buf.index(slot), 0b00000011),
        (bs_load_desc, bs_buf.index(slot), 0b00001100),
    ])


@gluon.jit
def _issue_fp8_scaled_slice_mnk_fused_data_load(a_desc, b_desc, a_buf, b_buf, tile_idx, slot,
                                                BLOCK_K: gl.constexpr):
    tile_k = tile_idx * BLOCK_K
    a_load_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, tile_k])
    b_load_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, tile_k])
    tdm.async_load_fused([
        (a_load_desc, a_buf.index(slot), 0b00001111),
        (b_load_desc, b_buf.index(slot), 0b11110000),
    ])


@gluon.jit
def _issue_fp8_scaled_slice_mnk_leading4_fused_data_load(a_desc, b_desc, a_buf, b_buf, tile_idx, slot,
                                                         BLOCK_K: gl.constexpr):
    tile_k = tile_idx * BLOCK_K
    a_load_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, tile_k])
    b_load_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, tile_k])
    tdm.async_load_fused([
        (a_load_desc, a_buf.index(slot), 0b00000011),
        (b_load_desc, b_buf.index(slot), 0b00001100),
    ])


@gluon.jit
def _issue_fp8_scaled_slice_mnk_separate_data_load(a_desc, b_desc, a_buf, b_buf, tile_idx, slot,
                                                   BLOCK_K: gl.constexpr):
    tile_k = tile_idx * BLOCK_K
    a_load_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, tile_k])
    b_load_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, tile_k])
    tdm.async_load(a_load_desc, dest=a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(b_load_desc, dest=b_buf.index(slot), warp_used_hint=0b11110000)


@gluon.jit
def _issue_fp8_scaled_slice_mnk_l2_prefetch(a_desc, b_desc, tile_idx, BLOCK_K: gl.constexpr):
    tile_k = tile_idx * BLOCK_K
    tdm.prefetch(a_desc, [0, tile_k])
    tdm.prefetch(b_desc, [0, tile_k])


def _build_fp8_scaled_cluster_layouts(BLOCK_M, BLOCK_N, BLOCK_K, SCALE_BLOCK, NUM_WARPS, cga_layout_c,
                                      CTA_M=2, CTA_N=1, HIPBLASLT_SCALE_LAYOUT=False, USE_PARTITIONED=False):
    """Build layouts for an FP8 tile distributed across M and optionally N.

    USE_PARTITIONED gives each CTA-local A and B slice two physical LDS
    partitions matching the partition-aware WMMA mapping. Scale tiles retain
    their padded CGA layouts.
    """
    assert NUM_WARPS == 8
    padded_a_local = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 16]], [BLOCK_M, BLOCK_K], [1, 0])
    padded_b_local = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 16]], [BLOCK_N, BLOCK_K], [1, 0])
    local_layouts = gl.amd.gfx1250.make_partitioned_dot_layouts(
        BLOCK_M, BLOCK_N, padded_a_local, padded_b_local, NUM_WARPS, [16, 16, 128], a_transposed=False,
        b_transposed=True, slice_m=BLOCK_M // CTA_M, slice_n=BLOCK_N // CTA_N)
    local_wmma = local_layouts[2]
    wmma = gl.amd.AMDWMMALayout(3, local_wmma.transposed, local_wmma.warp_bases, local_wmma.reg_bases,
                                local_wmma.instr_shape, cga_layout_c)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma, 16)
    cga_layout_a = dot_a.cga_layout
    cga_layout_b = dot_b.cga_layout
    # B and B-scale LDS tiles are stored as [N, K], while dot-B's CGA bases
    # describe the post-permute [K, N] view.
    cga_layout_b_transposed = tuple(tuple([basis[1], basis[0]]) for basis in cga_layout_b)

    padded_a = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 16]], [BLOCK_M, BLOCK_K], [1, 0],
                                                       cga_layout_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 16]], [BLOCK_N, BLOCK_K], [1, 0],
                                                       cga_layout_b_transposed)
    shared_a = padded_a
    shared_b = padded_b
    if USE_PARTITIONED:
        # Physically partition each CTA-local operand slice to match the
        # partition-aware WMMA mapping derived above.
        shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
            BLOCK_M // CTA_M, BLOCK_N // CTA_N, padded_a, padded_b, NUM_WARPS, [16, 16, 128],
            a_transposed=False, b_transposed=True, slice_m=BLOCK_M // CTA_M, slice_n=BLOCK_N // CTA_N)

    bk_scale = BLOCK_K // SCALE_BLOCK
    preshuffle_factor = 128
    if HIPBLASLT_SCALE_LAYOUT:
        scale_kwidth = 4 if bk_scale >= 4 else bk_scale
        cga_layout_scale_a = tuple(tuple([basis[1], basis[0]]) for basis in cga_layout_a)
        cga_layout_scale_b = tuple(tuple([basis[1], basis[0]]) for basis in cga_layout_b_transposed)
        shared_scale_a = gl.PaddedSharedLayout.with_identity_for(
            [[256, 8]], [bk_scale // scale_kwidth, BLOCK_M * scale_kwidth], [1, 0],
            cga_layout_scale_a)
        shared_scale_b = gl.PaddedSharedLayout.with_identity_for(
            [[256, 8]], [bk_scale // scale_kwidth, BLOCK_N * scale_kwidth], [1, 0],
            cga_layout_scale_b)
    else:
        shared_scale_a = gl.PaddedSharedLayout.with_identity_for(
            [[256, 8]], [BLOCK_M // preshuffle_factor, bk_scale * preshuffle_factor], [1, 0],
            cga_layout_a)
        shared_scale_b = gl.PaddedSharedLayout.with_identity_for(
            [[256, 8]], [BLOCK_N // preshuffle_factor, bk_scale * preshuffle_factor], [1, 0],
            cga_layout_b_transposed)
    return shared_a, shared_b, shared_scale_a, shared_scale_b, wmma


def _build_f16_cluster_layouts(BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, cga_layout_c):
    """Build layouts for an F16 tile distributed across a CTA cluster."""
    assert NUM_WARPS == 8
    cta_m = 4
    cta_n = 4
    slice_m = BLOCK_M // cta_m
    slice_n = BLOCK_N // cta_n

    # Derive the same partition-aware warp/register mapping used by the
    # historical 256x256 KernelC, then distribute that local tile across the
    # 4x4 CGA.
    padded_a_local = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0])
    padded_b_local = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_N, BLOCK_K], [1, 0])
    local_layouts = gl.amd.gfx1250.make_partitioned_dot_layouts(
        BLOCK_M, BLOCK_N, padded_a_local, padded_b_local, NUM_WARPS, [16, 16, 32], a_transposed=False,
        b_transposed=True, slice_m=slice_m, slice_n=slice_n, transposed=True)
    local_wmma = local_layouts[2]
    wmma = gl.amd.AMDWMMALayout(3, local_wmma.transposed, local_wmma.warp_bases, local_wmma.reg_bases,
                                local_wmma.instr_shape, cga_layout_c)
    dot_a = gl.DotOperandLayout(0, wmma, 8)
    dot_b = gl.DotOperandLayout(1, wmma, 8)
    cga_layout_a = dot_a.cga_layout
    cga_layout_b = dot_b.cga_layout
    cga_layout_b_transposed = tuple(tuple([basis[1], basis[0]]) for basis in cga_layout_b)

    padded_a = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0],
                                                       cga_layout_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_N, BLOCK_K], [1, 0],
                                                       cga_layout_b_transposed)
    return padded_a, padded_b, wmma


@gluon.jit
def fp8_scaled_slice_mn_warp_pipeline_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
        stride_cn, stride_scale, DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr, SCALE_BLOCK: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
        GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr, NUM_WARPS: gl.constexpr, SCALE_PRESHUFFLE: gl.constexpr,
        ASYNC_COPY_SCALE: gl.constexpr, L2_PREFETCH_DISTANCE: gl.constexpr, Z_ORDER: gl.constexpr = False,
        SWIZZLE_BLOCK_M: gl.constexpr = 0, SWIZZLE_BLOCK_N: gl.constexpr = 0):
    gl.static_assert(DTYPE_A != "e2m1" and DTYPE_B != "e2m1",
                     "fp8_scaled_slice_mn_warp_pipeline_kernel_gfx1250 requires FP8 inputs")
    gl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 256)
    gl.static_assert(SCALE_BLOCK == 16 or SCALE_BLOCK == 32)
    gl.static_assert(NUM_WARPS == 8)
    gl.static_assert(SCALE_PRESHUFFLE, "the BK256 scaled FP8 tutorial requires preshuffled scales")
    gl.static_assert(not ASYNC_COPY_SCALE, "the BK256 scaled FP8 tutorial stages scales with TDM")
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M, Z_ORDER,
                                        SWIZZLE_BLOCK_M, SWIZZLE_BLOCK_N)

    # Group two K=128 WMMA steps into each BK=256 tile. Full-block descriptors
    # and fused A/B scale copies reduce each slot to three TDM requests, while
    # partition-aware slices retain the 128x128 operand layouts. A dedicated
    # TDM stage refills only after all eight WMMAs consume the slot.
    # On the development gfx1250 system, scale fusion improved the 8192^3
    # benchmark median from 5,480 to 5,737 TFLOPS with zero scratch use.
    num_subtiles: gl.constexpr = (2, 2, 2)
    subtile_m: gl.constexpr = BLOCK_M // num_subtiles[0]
    subtile_n: gl.constexpr = BLOCK_N // num_subtiles[1]
    subtile_k: gl.constexpr = BLOCK_K // num_subtiles[2]
    bk_scale: gl.constexpr = BLOCK_K // SCALE_BLOCK
    subtile_scale_k: gl.constexpr = bk_scale // num_subtiles[2]
    preshuffle_factor: gl.constexpr = 128
    bk_scale_preshuffled: gl.constexpr = bk_scale * preshuffle_factor
    block_m_preshuffled: gl.constexpr = BLOCK_M // preshuffle_factor
    block_n_preshuffled: gl.constexpr = BLOCK_N // preshuffle_factor
    scale_kwidth: gl.constexpr = 4 if bk_scale >= 4 else bk_scale

    padded_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 16]], [BLOCK_M, BLOCK_K], [1, 0])
    padded_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 16]], [BLOCK_N, BLOCK_K], [1, 0])
    shared_scale: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_m_preshuffled, bk_scale_preshuffled], [1, 0])
    layouts: gl.constexpr = gl.amd.gfx1250.make_partitioned_dot_layouts(
        BLOCK_M, BLOCK_N, padded_a, padded_b, NUM_WARPS, [16, 16, 128], a_transposed=False, b_transposed=True,
        slice_m=subtile_m, slice_n=subtile_n)
    shared_a: gl.constexpr = layouts[0]
    shared_b: gl.constexpr = layouts[1]
    wmma: gl.constexpr = layouts[2]
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_a,
                                                                       [subtile_m, subtile_scale_k])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_b,
                                                                       [subtile_n, subtile_scale_k])

    nbuf: gl.constexpr = 2
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, BLOCK_M, BLOCK_K], shared_a)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, BLOCK_N, BLOCK_K], shared_b)
    as_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty,
                                       [nbuf, block_m_preshuffled, bk_scale_preshuffled], shared_scale)
    bs_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty,
                                       [nbuf, block_n_preshuffled, bk_scale_preshuffled], shared_scale)

    a_base = pid_m * BLOCK_M * stride_am
    b_base = pid_n * BLOCK_N * stride_bn
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base, shape=(M, K), strides=(stride_am, stride_ak),
                                        block_shape=(BLOCK_M, BLOCK_K), layout=shared_a)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base, shape=(N, K), strides=(stride_bn, stride_bk),
                                        block_shape=(BLOCK_N, BLOCK_K), layout=shared_b)
    as_base = (pid_m * BLOCK_M) // preshuffle_factor * stride_scale
    bs_base = (pid_n * BLOCK_N) // preshuffle_factor * stride_scale
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + as_base, shape=(M // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(block_m_preshuffled, bk_scale_preshuffled), layout=shared_scale)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + bs_base, shape=(N // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(block_n_preshuffled, bk_scale_preshuffled), layout=shared_scale)

    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max >= 2)
    for prologue_l2_idx in gl.static_range(2, 2 + L2_PREFETCH_DISTANCE):
        if prologue_l2_idx < iter_max:
            _issue_fp8_scaled_slice_mnk_l2_prefetch(a_desc, b_desc, prologue_l2_idx, BLOCK_K)

    for prefetch_idx in gl.static_range(2):
        _issue_fp8_scaled_slice_mnk_fused_scale_load(as_desc, bs_desc, as_buf, bs_buf, prefetch_idx, prefetch_idx,
                                                     bk_scale_preshuffled)
        tdm.async_load(a_desc, [0, prefetch_idx * BLOCK_K], a_buf.index(prefetch_idx),
                       warp_used_hint=0b00001111)
        tdm.async_load(b_desc, [0, prefetch_idx * BLOCK_K], b_buf.index(prefetch_idx),
                       warp_used_hint=0b00001111)

    tdm.async_wait(3)
    acc_tl = gl.zeros((subtile_m, subtile_n), dtype=gl.float32, layout=wmma)
    acc_bl = gl.zeros((subtile_m, subtile_n), dtype=gl.float32, layout=wmma)
    acc_tr = gl.zeros((subtile_m, subtile_n), dtype=gl.float32, layout=wmma)
    acc_br = gl.zeros((subtile_m, subtile_n), dtype=gl.float32, layout=wmma)

    for tile_idx in range(0, iter_max - 2):
        slot = tile_idx % nbuf
        refill_idx = tile_idx + 2
        prefetch_tile = refill_idx + L2_PREFETCH_DISTANCE

        # Consume the BK256 slot as two explicit K128 tutorial steps. The
        # full-slot refill remains below because the second half is still live.
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a00 = a_buf.index(slot).slice(0, subtile_m, 0).slice(0, subtile_k, 1).load(layout=dot_a)
            as00 = _load_fp8_scaled_slice_mnk_scale(as_buf, slot, 0, 0, scale_a_layout, BLOCK_M, bk_scale,
                                                   subtile_m, subtile_scale_k, preshuffle_factor, scale_kwidth)
            b00 = b_buf.index(slot).slice(0, subtile_n, 0).slice(
                0, subtile_k, 1).permute([1, 0]).load(layout=dot_b)
            bs00 = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, 0, 0, scale_b_layout, BLOCK_N, bk_scale,
                                                   subtile_n, subtile_scale_k, preshuffle_factor, scale_kwidth)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.gfx1250.wmma_scaled(a00, as00, DTYPE_A, b00, bs00, DTYPE_B, acc_tl)

        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b01 = b_buf.index(slot).slice(subtile_n, subtile_n, 0).slice(
                0, subtile_k, 1).permute([1, 0]).load(layout=dot_b)
            bs01 = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, subtile_n, 0, scale_b_layout, BLOCK_N, bk_scale,
                                                   subtile_n, subtile_scale_k, preshuffle_factor, scale_kwidth)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.gfx1250.wmma_scaled(a00, as00, DTYPE_A, b01, bs01, DTYPE_B, acc_tr)

        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a10 = a_buf.index(slot).slice(subtile_m, subtile_m, 0).slice(
                0, subtile_k, 1).load(layout=dot_a)
            as10 = _load_fp8_scaled_slice_mnk_scale(as_buf, slot, subtile_m, 0, scale_a_layout, BLOCK_M, bk_scale,
                                                   subtile_m, subtile_scale_k, preshuffle_factor, scale_kwidth)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.gfx1250.wmma_scaled(a10, as10, DTYPE_A, b00, bs00, DTYPE_B, acc_bl)

        with gl.amd.warp_pipeline_stage("mem", priority=1):
            if L2_PREFETCH_DISTANCE > 0:
                if prefetch_tile < iter_max:
                    _issue_fp8_scaled_slice_mnk_l2_prefetch(a_desc, b_desc, prefetch_tile, BLOCK_K)
            b10 = b_buf.index(slot).slice(0, subtile_n, 0).slice(
                subtile_k, subtile_k, 1).permute([1, 0]).load(layout=dot_b)
            bs10 = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, 0, subtile_scale_k, scale_b_layout, BLOCK_N,
                                                   bk_scale, subtile_n, subtile_scale_k, preshuffle_factor,
                                                   scale_kwidth)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.gfx1250.wmma_scaled(a10, as10, DTYPE_A, b01, bs01, DTYPE_B, acc_br)

        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a01 = a_buf.index(slot).slice(0, subtile_m, 0).slice(
                subtile_k, subtile_k, 1).load(layout=dot_a)
            as01 = _load_fp8_scaled_slice_mnk_scale(as_buf, slot, 0, subtile_scale_k, scale_a_layout, BLOCK_M,
                                                   bk_scale, subtile_m, subtile_scale_k, preshuffle_factor,
                                                   scale_kwidth)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.gfx1250.wmma_scaled(a01, as01, DTYPE_A, b10, bs10, DTYPE_B, acc_tl)

        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b11 = b_buf.index(slot).slice(subtile_n, subtile_n, 0).slice(
                subtile_k, subtile_k, 1).permute([1, 0]).load(layout=dot_b)
            bs11 = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, subtile_n, subtile_scale_k, scale_b_layout,
                                                   BLOCK_N, bk_scale, subtile_n, subtile_scale_k,
                                                   preshuffle_factor, scale_kwidth)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.gfx1250.wmma_scaled(a01, as01, DTYPE_A, b11, bs11, DTYPE_B, acc_tr)

        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a11 = a_buf.index(slot).slice(subtile_m, subtile_m, 0).slice(
                subtile_k, subtile_k, 1).load(layout=dot_a)
            as11 = _load_fp8_scaled_slice_mnk_scale(as_buf, slot, subtile_m, subtile_scale_k, scale_a_layout,
                                                   BLOCK_M, bk_scale, subtile_m, subtile_scale_k,
                                                   preshuffle_factor, scale_kwidth)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.gfx1250.wmma_scaled(a11, as11, DTYPE_A, b10, bs10, DTYPE_B, acc_bl)
            acc_br = gl.amd.gfx1250.wmma_scaled(a11, as11, DTYPE_A, b11, bs11, DTYPE_B, acc_br)

        with gl.amd.warp_pipeline_stage("tdm", priority=1):
            _issue_fp8_scaled_slice_mnk_fused_scale_load(as_desc, bs_desc, as_buf, bs_buf, refill_idx, slot,
                                                         bk_scale_preshuffled)
            tdm.async_load(a_desc, [0, refill_idx * BLOCK_K], a_buf.index(slot), warp_used_hint=0b00001111)
            tdm.async_load(b_desc, [0, refill_idx * BLOCK_K], b_buf.index(slot), warp_used_hint=0b00001111)
        tdm.async_wait(3)

    penultimate_idx = iter_max - 2
    penultimate_slot = penultimate_idx % nbuf
    acc_tl, acc_bl, acc_tr, acc_br = _consume_fp8_scaled_slice_mnk_tile(
        a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc_tl, acc_bl, acc_tr, acc_br, DTYPE_A, DTYPE_B, dot_a,
        dot_b, scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N, BLOCK_K, bk_scale, subtile_m, subtile_n,
        subtile_k, subtile_scale_k, preshuffle_factor, scale_kwidth)

    tdm.async_wait(0)
    last_slot = (iter_max - 1) % nbuf
    acc_tl, acc_bl, acc_tr, acc_br = _consume_fp8_scaled_slice_mnk_tile(
        a_buf, b_buf, as_buf, bs_buf, last_slot, acc_tl, acc_bl, acc_tr, acc_br, DTYPE_A, DTYPE_B, dot_a, dot_b,
        scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N, BLOCK_K, bk_scale, subtile_m, subtile_n, subtile_k,
        subtile_scale_k, preshuffle_factor, scale_kwidth)

    _store_quadrants(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc_tl, acc_bl, acc_tr, acc_br, wmma,
                     BLOCK_M, BLOCK_N)


@gluon.jit
def _consume_fp8_scaled_cluster_tile(a_buf, b_buf, as_buf, bs_buf, slot, acc_l, acc_r,
                                     DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr, DOT_A: gl.constexpr,
                                     DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
                                     SCALE_B_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                                     BLOCK_K: gl.constexpr, BK_SCALE: gl.constexpr, SUBTILE_N: gl.constexpr,
                                     SUBTILE_K: gl.constexpr, SUBTILE_SCALE_K: gl.constexpr,
                                     PRESHUFFLE_FACTOR: gl.constexpr, SCALE_KWIDTH: gl.constexpr):
    a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(0, SUBTILE_K, 1).load(layout=DOT_A)
    as0 = _load_fp8_scaled_slice_mnk_scale(as_buf, slot, 0, 0, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE, BLOCK_M,
                                          SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    b0 = b_buf.index(slot).slice(0, SUBTILE_N, 0).slice(0, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
    bs0 = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, 0, 0, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE, SUBTILE_N,
                                          SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    acc_l = gl.amd.gfx1250.wmma_scaled(a0, as0, DTYPE_A, b0, bs0, DTYPE_B, acc_l)

    b1 = b_buf.index(slot).slice(SUBTILE_N, SUBTILE_N, 0).slice(
        0, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
    bs1 = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, SUBTILE_N, 0, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE,
                                          SUBTILE_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    acc_r = gl.amd.gfx1250.wmma_scaled(a0, as0, DTYPE_A, b1, bs1, DTYPE_B, acc_r)

    a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(SUBTILE_K, SUBTILE_K, 1).load(layout=DOT_A)
    as1 = _load_fp8_scaled_slice_mnk_scale(as_buf, slot, 0, SUBTILE_SCALE_K, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE,
                                          BLOCK_M, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    b2 = b_buf.index(slot).slice(0, SUBTILE_N, 0).slice(
        SUBTILE_K, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
    bs2 = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, 0, SUBTILE_SCALE_K, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE,
                                          SUBTILE_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    acc_l = gl.amd.gfx1250.wmma_scaled(a1, as1, DTYPE_A, b2, bs2, DTYPE_B, acc_l)

    b3 = b_buf.index(slot).slice(SUBTILE_N, SUBTILE_N, 0).slice(
        SUBTILE_K, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
    bs3 = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, SUBTILE_N, SUBTILE_SCALE_K, SCALE_B_LAYOUT, BLOCK_N,
                                          BK_SCALE, SUBTILE_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    acc_r = gl.amd.gfx1250.wmma_scaled(a1, as1, DTYPE_A, b3, bs3, DTYPE_B, acc_r)
    return acc_l, acc_r


@gluon.jit
def _consume_fp8_scaled_cluster_2x2_tile(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, BK_SCALE: gl.constexpr,
        SUBTILE_K: gl.constexpr, SUBTILE_SCALE_K: gl.constexpr, PRESHUFFLE_FACTOR: gl.constexpr,
        SCALE_KWIDTH: gl.constexpr):
    a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(0, SUBTILE_K, 1).load(layout=DOT_A)
    as0 = _load_fp8_scaled_slice_mnk_scale(as_buf, slot, 0, 0, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE, BLOCK_M,
                                          SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
        0, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
    bs0 = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, 0, 0, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE, BLOCK_N,
                                          SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    acc = gl.amd.gfx1250.wmma_scaled(a0, as0, DTYPE_A, b0, bs0, DTYPE_B, acc)

    a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(SUBTILE_K, SUBTILE_K, 1).load(layout=DOT_A)
    as1 = _load_fp8_scaled_slice_mnk_scale(as_buf, slot, 0, SUBTILE_SCALE_K, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE,
                                          BLOCK_M, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
        SUBTILE_K, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
    bs1 = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, 0, SUBTILE_SCALE_K, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE,
                                          BLOCK_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    return gl.amd.gfx1250.wmma_scaled(a1, as1, DTYPE_A, b1, bs1, DTYPE_B, acc)


@gluon.jit
def _consume_fp8_scaled_cluster_2x2_tile_hipblaslt(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, BK_SCALE: gl.constexpr,
        SUBTILE_K: gl.constexpr, SUBTILE_SCALE_K: gl.constexpr, SCALE_KWIDTH: gl.constexpr):
    a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(0, SUBTILE_K, 1).load(layout=DOT_A)
    as0 = _load_fp8_scaled_slice_mnk_scale_hipblaslt(
        as_buf, slot, 0, 0, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE, BLOCK_M, SUBTILE_SCALE_K, SCALE_KWIDTH)
    b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
        0, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
    bs0 = _load_fp8_scaled_slice_mnk_scale_hipblaslt(
        bs_buf, slot, 0, 0, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE, BLOCK_N, SUBTILE_SCALE_K, SCALE_KWIDTH)
    acc = gl.amd.gfx1250.wmma_scaled(a0, as0, DTYPE_A, b0, bs0, DTYPE_B, acc)

    a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(SUBTILE_K, SUBTILE_K, 1).load(layout=DOT_A)
    as1 = _load_fp8_scaled_slice_mnk_scale_hipblaslt(
        as_buf, slot, 0, SUBTILE_SCALE_K, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE, BLOCK_M, SUBTILE_SCALE_K,
        SCALE_KWIDTH)
    b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
        SUBTILE_K, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
    bs1 = _load_fp8_scaled_slice_mnk_scale_hipblaslt(
        bs_buf, slot, 0, SUBTILE_SCALE_K, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE, BLOCK_N, SUBTILE_SCALE_K,
        SCALE_KWIDTH)
    return gl.amd.gfx1250.wmma_scaled(a1, as1, DTYPE_A, b1, bs1, DTYPE_B, acc)


@gluon.jit
def fp8_scaled_slice_mn_cluster_warp_pipeline_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
        stride_cn, stride_scale, DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr, SCALE_BLOCK: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
        GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr, NUM_WARPS: gl.constexpr, SCALE_PRESHUFFLE: gl.constexpr,
        ASYNC_COPY_SCALE: gl.constexpr, L2_PREFETCH_DISTANCE: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr, SHARED_SCALE_B: gl.constexpr,
        WMMA_LAYOUT: gl.constexpr, Z_ORDER: gl.constexpr = False, SWIZZLE_BLOCK_M: gl.constexpr = 0,
        SWIZZLE_BLOCK_N: gl.constexpr = 0):
    gl.static_assert(DTYPE_A != "e2m1" and DTYPE_B != "e2m1",
                     "fp8_scaled_slice_mn_cluster_warp_pipeline_kernel_gfx1250 requires FP8 inputs")
    gl.static_assert(BLOCK_M == 1024 and BLOCK_N == 1024 and BLOCK_K == 256)
    gl.static_assert(SCALE_BLOCK == 16 or SCALE_BLOCK == 32)
    gl.static_assert(NUM_WARPS == 8)
    gl.static_assert(gl.num_ctas() == 16, "the FP8 cluster tutorial requires a 16-CTA cluster")
    gl.static_assert(SCALE_PRESHUFFLE, "the BK256 scaled FP8 cluster tutorial requires preshuffled scales")
    gl.static_assert(not ASYNC_COPY_SCALE, "the BK256 scaled FP8 cluster tutorial stages scales with TDM")
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M, Z_ORDER,
                                        SWIZZLE_BLOCK_M, SWIZZLE_BLOCK_N)

    # Keep M and N whole so the CGA layout gives each CTA one contiguous
    # 256x256 part of the aggregate output tile.
    num_subtiles: gl.constexpr = (1, 1, 2)
    subtile_m: gl.constexpr = BLOCK_M // num_subtiles[0]
    subtile_n: gl.constexpr = BLOCK_N // num_subtiles[1]
    subtile_k: gl.constexpr = BLOCK_K // num_subtiles[2]
    bk_scale: gl.constexpr = BLOCK_K // SCALE_BLOCK
    subtile_scale_k: gl.constexpr = bk_scale // num_subtiles[2]
    preshuffle_factor: gl.constexpr = 128
    bk_scale_preshuffled: gl.constexpr = bk_scale * preshuffle_factor
    block_m_preshuffled: gl.constexpr = BLOCK_M // preshuffle_factor
    block_n_preshuffled: gl.constexpr = BLOCK_N // preshuffle_factor
    scale_kwidth: gl.constexpr = 4 if bk_scale >= 4 else bk_scale

    shared_a: gl.constexpr = SHARED_LAYOUT_A
    shared_b: gl.constexpr = SHARED_LAYOUT_B
    shared_scale_a: gl.constexpr = SHARED_SCALE_A
    shared_scale_b: gl.constexpr = SHARED_SCALE_B
    wmma: gl.constexpr = WMMA_LAYOUT
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_a,
                                                                       [subtile_m, subtile_scale_k])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_b,
                                                                       [subtile_n, subtile_scale_k])

    nbuf: gl.constexpr = 2
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, BLOCK_M, BLOCK_K], shared_a)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, BLOCK_N, BLOCK_K], shared_b)
    as_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty,
                                       [nbuf, block_m_preshuffled, bk_scale_preshuffled], shared_scale_a)
    bs_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty,
                                       [nbuf, block_n_preshuffled, bk_scale_preshuffled], shared_scale_b)

    a_base = pid_m * BLOCK_M * stride_am
    b_base = pid_n * BLOCK_N * stride_bn
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base, shape=(M, K), strides=(stride_am, stride_ak),
                                        block_shape=(BLOCK_M, BLOCK_K), layout=shared_a)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base, shape=(N, K), strides=(stride_bn, stride_bk),
                                        block_shape=(BLOCK_N, BLOCK_K), layout=shared_b)
    as_base = (pid_m * BLOCK_M) // preshuffle_factor * stride_scale
    bs_base = (pid_n * BLOCK_N) // preshuffle_factor * stride_scale
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + as_base, shape=(M // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(block_m_preshuffled, bk_scale_preshuffled), layout=shared_scale_a)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + bs_base, shape=(N // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(block_n_preshuffled, bk_scale_preshuffled), layout=shared_scale_b)

    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max >= 2)
    for prologue_l2_idx in gl.static_range(2, 2 + L2_PREFETCH_DISTANCE):
        if prologue_l2_idx < iter_max:
            _issue_fp8_scaled_slice_mnk_l2_prefetch(a_desc, b_desc, prologue_l2_idx, BLOCK_K)

    for prefetch_idx in gl.static_range(2):
        _issue_fp8_scaled_slice_mnk_fused_scale_load(as_desc, bs_desc, as_buf, bs_buf, prefetch_idx, prefetch_idx,
                                                     bk_scale_preshuffled)
        tdm.async_load(a_desc, [0, prefetch_idx * BLOCK_K], a_buf.index(prefetch_idx),
                       warp_used_hint=0b00001111)
        tdm.async_load(b_desc, [0, prefetch_idx * BLOCK_K], b_buf.index(prefetch_idx),
                       warp_used_hint=0b00001111)

    tdm.async_wait(3)
    _cluster_wait(3)
    acc = gl.zeros((subtile_m, subtile_n), dtype=gl.float32, layout=wmma)

    for tile_idx in range(0, iter_max - 2):
        slot = tile_idx % nbuf
        refill_idx = tile_idx + 2
        prefetch_tile = refill_idx + L2_PREFETCH_DISTANCE

        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a0 = a_buf.index(slot).slice(0, subtile_m, 0).slice(0, subtile_k, 1).load(layout=dot_a)
            as0 = _load_fp8_scaled_slice_mnk_scale(as_buf, slot, 0, 0, scale_a_layout, BLOCK_M, bk_scale,
                                                  subtile_m, subtile_scale_k, preshuffle_factor, scale_kwidth)
            b0 = b_buf.index(slot).slice(0, subtile_n, 0).slice(
                0, subtile_k, 1).permute([1, 0]).load(layout=dot_b)
            bs0 = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, 0, 0, scale_b_layout, BLOCK_N, bk_scale,
                                                  subtile_n, subtile_scale_k, preshuffle_factor, scale_kwidth)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc = gl.amd.gfx1250.wmma_scaled(a0, as0, DTYPE_A, b0, bs0, DTYPE_B, acc)

        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a1 = a_buf.index(slot).slice(0, subtile_m, 0).slice(
                subtile_k, subtile_k, 1).load(layout=dot_a)
            as1 = _load_fp8_scaled_slice_mnk_scale(as_buf, slot, 0, subtile_scale_k, scale_a_layout, BLOCK_M,
                                                  bk_scale, subtile_m, subtile_scale_k, preshuffle_factor,
                                                  scale_kwidth)
            if L2_PREFETCH_DISTANCE > 0:
                if prefetch_tile < iter_max:
                    _issue_fp8_scaled_slice_mnk_l2_prefetch(a_desc, b_desc, prefetch_tile, BLOCK_K)
            b1 = b_buf.index(slot).slice(0, subtile_n, 0).slice(
                subtile_k, subtile_k, 1).permute([1, 0]).load(layout=dot_b)
            bs1 = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, 0, subtile_scale_k, scale_b_layout, BLOCK_N,
                                                  bk_scale, subtile_n, subtile_scale_k, preshuffle_factor,
                                                  scale_kwidth)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc = gl.amd.gfx1250.wmma_scaled(a1, as1, DTYPE_A, b1, bs1, DTYPE_B, acc)

        with gl.amd.warp_pipeline_stage("tdm", priority=1):
            _issue_fp8_scaled_slice_mnk_fused_scale_load(as_desc, bs_desc, as_buf, bs_buf, refill_idx, slot,
                                                         bk_scale_preshuffled)
            tdm.async_load(a_desc, [0, refill_idx * BLOCK_K], a_buf.index(slot), warp_used_hint=0b00001111)
            tdm.async_load(b_desc, [0, refill_idx * BLOCK_K], b_buf.index(slot), warp_used_hint=0b00001111)
            _cluster_wait(3)
        tdm.async_wait(3)

    penultimate_idx = iter_max - 2
    penultimate_slot = penultimate_idx % nbuf
    acc = _consume_fp8_scaled_cluster_2x2_tile(
        a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, DTYPE_A, DTYPE_B, dot_a, dot_b, scale_a_layout,
        scale_b_layout, BLOCK_M, BLOCK_N, BLOCK_K, bk_scale, subtile_k, subtile_scale_k, preshuffle_factor,
        scale_kwidth)

    tdm.async_wait(0)
    _cluster_wait(0)
    last_slot = (iter_max - 1) % nbuf
    acc = _consume_fp8_scaled_cluster_2x2_tile(
        a_buf, b_buf, as_buf, bs_buf, last_slot, acc, DTYPE_A, DTYPE_B, dot_a, dot_b, scale_a_layout,
        scale_b_layout, BLOCK_M, BLOCK_N, BLOCK_K, bk_scale, subtile_k, subtile_scale_k, preshuffle_factor,
        scale_kwidth)

    _store_full_tile(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc, wmma, BLOCK_M, BLOCK_N)


@gluon.jit
def _fp8_scaled_cluster_bf16_style_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, slot, a_desc, b_desc, as_desc, bs_desc, refill_idx, acc,
        DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, BK_SCALE: gl.constexpr,
        SUBTILE_K: gl.constexpr, SUBTILE_SCALE_K: gl.constexpr, PRESHUFFLE_FACTOR: gl.constexpr,
        BK_SCALE_PRESHUFFLED: gl.constexpr, SCALE_KWIDTH: gl.constexpr,
        HIPBLASLT_SCALE_LAYOUT: gl.constexpr, AB_SEPARATE_DATA: gl.constexpr,
        LEADING_FOUR_TDM: gl.constexpr, CTA_M: gl.constexpr, CTA_N: gl.constexpr):
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(0, SUBTILE_K, 1).load(layout=DOT_A)
        if HIPBLASLT_SCALE_LAYOUT:
            as0 = _load_fp8_scaled_slice_mnk_scale_hipblaslt(
                as_buf, slot, 0, 0, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE, BLOCK_M, SUBTILE_SCALE_K,
                SCALE_KWIDTH)
        else:
            as0 = _load_fp8_scaled_slice_mnk_scale(as_buf, slot, 0, 0, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE,
                                                  BLOCK_M, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
        b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            0, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
        if HIPBLASLT_SCALE_LAYOUT:
            bs0 = _load_fp8_scaled_slice_mnk_scale_hipblaslt(
                bs_buf, slot, 0, 0, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE, BLOCK_N, SUBTILE_SCALE_K,
                SCALE_KWIDTH)
        else:
            bs0 = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, 0, 0, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE,
                                                  BLOCK_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(a0, as0, DTYPE_A, b0, bs0, DTYPE_B, acc)

    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            SUBTILE_K, SUBTILE_K, 1).load(layout=DOT_A)
        if HIPBLASLT_SCALE_LAYOUT:
            as1 = _load_fp8_scaled_slice_mnk_scale_hipblaslt(
                as_buf, slot, 0, SUBTILE_SCALE_K, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE, BLOCK_M,
                SUBTILE_SCALE_K, SCALE_KWIDTH)
        else:
            as1 = _load_fp8_scaled_slice_mnk_scale(as_buf, slot, 0, SUBTILE_SCALE_K, SCALE_A_LAYOUT,
                                                  BLOCK_M, BK_SCALE, BLOCK_M, SUBTILE_SCALE_K,
                                                  PRESHUFFLE_FACTOR, SCALE_KWIDTH)
        b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            SUBTILE_K, SUBTILE_K, 1).permute([1, 0]).load(layout=DOT_B)
        if HIPBLASLT_SCALE_LAYOUT:
            bs1 = _load_fp8_scaled_slice_mnk_scale_hipblaslt(
                bs_buf, slot, 0, SUBTILE_SCALE_K, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE, BLOCK_N,
                SUBTILE_SCALE_K, SCALE_KWIDTH)
        else:
            bs1 = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, 0, SUBTILE_SCALE_K, SCALE_B_LAYOUT,
                                                  BLOCK_N, BK_SCALE, BLOCK_N, SUBTILE_SCALE_K,
                                                  PRESHUFFLE_FACTOR, SCALE_KWIDTH)
        if CTA_M * CTA_N > 1:
            gl.amd.gfx1250.cluster.arrive()
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(a1, as1, DTYPE_A, b1, bs1, DTYPE_B, acc)
        if CTA_M * CTA_N > 1:
            gl.amd.gfx1250.cluster.wait()
        if LEADING_FOUR_TDM:
            _issue_fp8_scaled_slice_mnk_leading4_fused_data_load(
                a_desc, b_desc, a_buf, b_buf, refill_idx, slot, BLOCK_K)
        elif AB_SEPARATE_DATA:
            _issue_fp8_scaled_slice_mnk_separate_data_load(
                a_desc, b_desc, a_buf, b_buf, refill_idx, slot, BLOCK_K)
        if LEADING_FOUR_TDM and HIPBLASLT_SCALE_LAYOUT:
            _issue_fp8_scaled_slice_mnk_leading4_fused_scale_load_hipblaslt(
                as_desc, bs_desc, as_buf, bs_buf, refill_idx, slot, BK_SCALE, SCALE_KWIDTH)
        elif LEADING_FOUR_TDM:
            _issue_fp8_scaled_slice_mnk_leading4_fused_scale_load(
                as_desc, bs_desc, as_buf, bs_buf, refill_idx, slot, BK_SCALE_PRESHUFFLED)
        elif HIPBLASLT_SCALE_LAYOUT:
            _issue_fp8_scaled_slice_mnk_fused_scale_load_hipblaslt(
                as_desc, bs_desc, as_buf, bs_buf, refill_idx, slot, BK_SCALE, SCALE_KWIDTH)
        else:
            _issue_fp8_scaled_slice_mnk_fused_scale_load(
                as_desc, bs_desc, as_buf, bs_buf, refill_idx, slot, BK_SCALE_PRESHUFFLED)
        if not LEADING_FOUR_TDM and not AB_SEPARATE_DATA:
            _issue_fp8_scaled_slice_mnk_fused_data_load(
                a_desc, b_desc, a_buf, b_buf, refill_idx, slot, BLOCK_K)
    return acc


@gluon.jit
def fp8_scaled_cluster_bf16_style_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
        stride_cm, stride_cn, stride_scale, DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr,
        SCALE_BLOCK: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
        GROUP_SIZE_M: gl.constexpr, GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr, NUM_WARPS: gl.constexpr,
        SCALE_PRESHUFFLE: gl.constexpr, ASYNC_COPY_SCALE: gl.constexpr,
        SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr, HIPBLASLT_SCALE_LAYOUT: gl.constexpr,
        AB_SEPARATE_DATA: gl.constexpr, LEADING_FOUR_TDM: gl.constexpr,
        CTA_M: gl.constexpr, CTA_N: gl.constexpr):
    gl.static_assert(DTYPE_A != "e2m1" and DTYPE_B != "e2m1",
                     "fp8_scaled_cluster_bf16_style_kernel_gfx1250 requires FP8 inputs")
    gl.static_assert(CTA_M == CTA_N and (CTA_M == 1 or CTA_M == 2 or CTA_M == 4),
                     "the BF16-style MXFP8 kernel supports 1x1, 2x2, and 4x4 clusters")
    gl.static_assert(BLOCK_M == CTA_M * 256 and BLOCK_N == CTA_N * 256 and BLOCK_K == 256)
    gl.static_assert(SCALE_BLOCK == 16 or SCALE_BLOCK == 32)
    gl.static_assert(NUM_WARPS == 8)
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N,
                     "the BF16-style MXFP8 kernel CTA count must match its cluster shape")
    gl.static_assert(SCALE_PRESHUFFLE, "the BF16-style MXFP8 kernel requires preshuffled scales")
    gl.static_assert(not ASYNC_COPY_SCALE, "the BF16-style MXFP8 kernel stages scales with TDM")
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    subtile_k: gl.constexpr = BLOCK_K // 2
    bk_scale: gl.constexpr = BLOCK_K // SCALE_BLOCK
    subtile_scale_k: gl.constexpr = bk_scale // 2
    preshuffle_factor: gl.constexpr = 128
    bk_scale_preshuffled: gl.constexpr = bk_scale * preshuffle_factor
    block_m_preshuffled: gl.constexpr = BLOCK_M // preshuffle_factor
    block_n_preshuffled: gl.constexpr = BLOCK_N // preshuffle_factor
    scale_kwidth: gl.constexpr = 4 if bk_scale >= 4 else bk_scale

    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [BLOCK_M, subtile_scale_k])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [BLOCK_N, subtile_scale_k])

    nbuf: gl.constexpr = 2
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, BLOCK_M, BLOCK_K], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, BLOCK_N, BLOCK_K], SHARED_LAYOUT_B)
    if HIPBLASLT_SCALE_LAYOUT:
        as_buf = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty, [nbuf, bk_scale // scale_kwidth, BLOCK_M * scale_kwidth],
            SHARED_SCALE_A)
        bs_buf = gl.allocate_shared_memory(
            b_scale_ptr.type.element_ty, [nbuf, bk_scale // scale_kwidth, BLOCK_N * scale_kwidth],
            SHARED_SCALE_B)
    else:
        as_buf = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty, [nbuf, block_m_preshuffled, bk_scale_preshuffled], SHARED_SCALE_A)
        bs_buf = gl.allocate_shared_memory(
            b_scale_ptr.type.element_ty, [nbuf, block_n_preshuffled, bk_scale_preshuffled], SHARED_SCALE_B)

    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K), strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, BLOCK_K), layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K), strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_N, BLOCK_K), layout=SHARED_LAYOUT_B)
    if HIPBLASLT_SCALE_LAYOUT:
        as_desc = tdm.make_tensor_descriptor(
            base=a_scale_ptr + pid_m * BLOCK_M * scale_kwidth,
            shape=(K // SCALE_BLOCK // scale_kwidth, M * scale_kwidth), strides=(stride_scale, 1),
            block_shape=(bk_scale // scale_kwidth, BLOCK_M * scale_kwidth), layout=SHARED_SCALE_A)
        bs_desc = tdm.make_tensor_descriptor(
            base=b_scale_ptr + pid_n * BLOCK_N * scale_kwidth,
            shape=(K // SCALE_BLOCK // scale_kwidth, N * scale_kwidth), strides=(stride_scale, 1),
            block_shape=(bk_scale // scale_kwidth, BLOCK_N * scale_kwidth), layout=SHARED_SCALE_B)
    else:
        as_desc = tdm.make_tensor_descriptor(
            base=a_scale_ptr + (pid_m * BLOCK_M) // preshuffle_factor * stride_scale,
            shape=(M // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor), strides=(stride_scale, 1),
            block_shape=(block_m_preshuffled, bk_scale_preshuffled), layout=SHARED_SCALE_A)
        bs_desc = tdm.make_tensor_descriptor(
            base=b_scale_ptr + (pid_n * BLOCK_N) // preshuffle_factor * stride_scale,
            shape=(N // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor), strides=(stride_scale, 1),
            block_shape=(block_n_preshuffled, bk_scale_preshuffled), layout=SHARED_SCALE_B)

    for prefetch_idx in gl.static_range(2):
        if LEADING_FOUR_TDM:
            _issue_fp8_scaled_slice_mnk_leading4_fused_data_load(
                a_desc, b_desc, a_buf, b_buf, prefetch_idx, prefetch_idx, BLOCK_K)
        elif AB_SEPARATE_DATA:
            _issue_fp8_scaled_slice_mnk_separate_data_load(
                a_desc, b_desc, a_buf, b_buf, prefetch_idx, prefetch_idx, BLOCK_K)
        if LEADING_FOUR_TDM and HIPBLASLT_SCALE_LAYOUT:
            _issue_fp8_scaled_slice_mnk_leading4_fused_scale_load_hipblaslt(
                as_desc, bs_desc, as_buf, bs_buf, prefetch_idx, prefetch_idx, bk_scale, scale_kwidth)
        elif LEADING_FOUR_TDM:
            _issue_fp8_scaled_slice_mnk_leading4_fused_scale_load(
                as_desc, bs_desc, as_buf, bs_buf, prefetch_idx, prefetch_idx, bk_scale_preshuffled)
        elif HIPBLASLT_SCALE_LAYOUT:
            _issue_fp8_scaled_slice_mnk_fused_scale_load_hipblaslt(
                as_desc, bs_desc, as_buf, bs_buf, prefetch_idx, prefetch_idx, bk_scale, scale_kwidth)
        else:
            _issue_fp8_scaled_slice_mnk_fused_scale_load(
                as_desc, bs_desc, as_buf, bs_buf, prefetch_idx, prefetch_idx, bk_scale_preshuffled)
        if not LEADING_FOUR_TDM and not AB_SEPARATE_DATA:
            _issue_fp8_scaled_slice_mnk_fused_data_load(
                a_desc, b_desc, a_buf, b_buf, prefetch_idx, prefetch_idx, BLOCK_K)

    wait_count: gl.constexpr = 2 if LEADING_FOUR_TDM or AB_SEPARATE_DATA else 1
    tdm.async_wait(wait_count)
    if CTA_M * CTA_N > 1:
        _cluster_wait(wait_count)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max >= 2)

    for tile_idx in range(0, iter_max - 2):
        slot = tile_idx % nbuf
        acc = _fp8_scaled_cluster_bf16_style_consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, slot, a_desc, b_desc, as_desc, bs_desc, tile_idx + 2, acc,
            DTYPE_A, DTYPE_B,
            dot_a, dot_b, scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N, BLOCK_K, bk_scale, subtile_k,
            subtile_scale_k, preshuffle_factor, bk_scale_preshuffled, scale_kwidth, HIPBLASLT_SCALE_LAYOUT,
            AB_SEPARATE_DATA, LEADING_FOUR_TDM, CTA_M, CTA_N)
        tdm.async_wait(wait_count)

    penultimate_slot = (iter_max - 2) % nbuf
    if HIPBLASLT_SCALE_LAYOUT:
        acc = _consume_fp8_scaled_cluster_2x2_tile_hipblaslt(
            a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, DTYPE_A, DTYPE_B, dot_a, dot_b,
            scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N, BLOCK_K, bk_scale, subtile_k,
            subtile_scale_k, scale_kwidth)
    else:
        acc = _consume_fp8_scaled_cluster_2x2_tile(
            a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, DTYPE_A, DTYPE_B, dot_a, dot_b,
            scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N, BLOCK_K, bk_scale, subtile_k,
            subtile_scale_k, preshuffle_factor, scale_kwidth)

    tdm.async_wait(0)
    if CTA_M * CTA_N > 1:
        _cluster_wait(0)
    last_slot = (iter_max - 1) % nbuf
    if HIPBLASLT_SCALE_LAYOUT:
        acc = _consume_fp8_scaled_cluster_2x2_tile_hipblaslt(
            a_buf, b_buf, as_buf, bs_buf, last_slot, acc, DTYPE_A, DTYPE_B, dot_a, dot_b,
            scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N, BLOCK_K, bk_scale, subtile_k,
            subtile_scale_k, scale_kwidth)
    else:
        acc = _consume_fp8_scaled_cluster_2x2_tile(
            a_buf, b_buf, as_buf, bs_buf, last_slot, acc, DTYPE_A, DTYPE_B, dot_a, dot_b,
            scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N, BLOCK_K, bk_scale, subtile_k,
            subtile_scale_k, preshuffle_factor, scale_kwidth)

    _tdm_store_full_tile(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc, WMMA_LAYOUT, BLOCK_M, BLOCK_N,
                         CTA_N)


@gluon.jit
def _issue_fp8_scaled_cluster_bk128_tile(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BK_SCALE: gl.constexpr, PRESHUFFLE_FACTOR: gl.constexpr,
        SCALE_KWIDTH: gl.constexpr, WAIT_COUNT: gl.constexpr):
    tdm.async_wait(WAIT_COUNT)
    _cluster_wait(WAIT_COUNT)
    a = a_buf.index(slot).load(layout=DOT_A)
    as_ = _load_fp8_scaled_slice_mnk_scale(as_buf, slot, 0, 0, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE, BLOCK_M,
                                          BK_SCALE, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    b = b_buf.index(slot).permute([1, 0]).load(layout=DOT_B)
    bs = _load_fp8_scaled_slice_mnk_scale(bs_buf, slot, 0, 0, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE, BLOCK_N,
                                         BK_SCALE, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    return gl.amd.gfx1250.wmma_scaled(a, as_, DTYPE_A, b, bs, DTYPE_B, acc)


@gluon.jit
def fp8_scaled_cluster_bk128_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
        stride_cn, stride_scale, DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr, SCALE_BLOCK: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
        GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr, NUM_WARPS: gl.constexpr, NUM_BUFFERS: gl.constexpr,
        SCALE_PRESHUFFLE: gl.constexpr, ASYNC_COPY_SCALE: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr, SHARED_SCALE_B: gl.constexpr,
        WMMA_LAYOUT: gl.constexpr, Z_ORDER: gl.constexpr = False, SWIZZLE_BLOCK_M: gl.constexpr = 0,
        SWIZZLE_BLOCK_N: gl.constexpr = 0):
    gl.static_assert(DTYPE_A != "e2m1" and DTYPE_B != "e2m1",
                     "fp8_scaled_cluster_bk128_kernel_gfx1250 requires FP8 inputs")
    gl.static_assert(BLOCK_M == 512 and BLOCK_N == 256 and BLOCK_K == 128)
    gl.static_assert(SCALE_BLOCK == 16 or SCALE_BLOCK == 32)
    gl.static_assert(NUM_WARPS == 8)
    gl.static_assert(NUM_BUFFERS == 3 or NUM_BUFFERS == 4,
                     "scaled FP8 BK128 cluster kernel requires three or four LDS buffers")
    gl.static_assert(gl.num_ctas() == 2, "scaled FP8 BK128 cluster kernel requires a two-CTA cluster")
    gl.static_assert(SCALE_PRESHUFFLE, "scaled FP8 BK128 cluster kernel requires preshuffled scales")
    gl.static_assert(not ASYNC_COPY_SCALE, "scaled FP8 BK128 cluster kernel stages scales with TDM")
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M, Z_ORDER,
                                        SWIZZLE_BLOCK_M, SWIZZLE_BLOCK_N)

    bk_scale: gl.constexpr = BLOCK_K // SCALE_BLOCK
    preshuffle_factor: gl.constexpr = 128
    bk_scale_preshuffled: gl.constexpr = bk_scale * preshuffle_factor
    block_m_preshuffled: gl.constexpr = BLOCK_M // preshuffle_factor
    block_n_preshuffled: gl.constexpr = BLOCK_N // preshuffle_factor
    scale_kwidth: gl.constexpr = 4 if bk_scale >= 4 else bk_scale
    requests_per_tile: gl.constexpr = 3
    nbuf: gl.constexpr = NUM_BUFFERS

    shared_a: gl.constexpr = SHARED_LAYOUT_A
    shared_b: gl.constexpr = SHARED_LAYOUT_B
    shared_scale_a: gl.constexpr = SHARED_SCALE_A
    shared_scale_b: gl.constexpr = SHARED_SCALE_B
    wmma: gl.constexpr = WMMA_LAYOUT
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_a, [BLOCK_M, bk_scale])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_b, [BLOCK_N, bk_scale])

    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, BLOCK_M, BLOCK_K], shared_a)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, BLOCK_N, BLOCK_K], shared_b)
    as_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty,
                                       [nbuf, block_m_preshuffled, bk_scale_preshuffled], shared_scale_a)
    bs_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty,
                                       [nbuf, block_n_preshuffled, bk_scale_preshuffled], shared_scale_b)

    a_base = pid_m * BLOCK_M * stride_am
    b_base = pid_n * BLOCK_N * stride_bn
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base, shape=(M, K), strides=(stride_am, stride_ak),
                                        block_shape=(BLOCK_M, BLOCK_K), layout=shared_a)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base, shape=(N, K), strides=(stride_bn, stride_bk),
                                        block_shape=(BLOCK_N, BLOCK_K), layout=shared_b)
    as_base = (pid_m * BLOCK_M) // preshuffle_factor * stride_scale
    bs_base = (pid_n * BLOCK_N) // preshuffle_factor * stride_scale
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + as_base, shape=(M // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(block_m_preshuffled, bk_scale_preshuffled), layout=shared_scale_a)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + bs_base, shape=(N // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(block_n_preshuffled, bk_scale_preshuffled), layout=shared_scale_b)

    producer = 0
    for _ in gl.static_range(NUM_BUFFERS - 1):
        slot = producer % nbuf
        _issue_fp8_scaled_slice_mnk_fused_scale_load(as_desc, bs_desc, as_buf, bs_buf, producer, slot,
                                                     bk_scale_preshuffled)
        tdm.async_load(a_desc, [0, producer * BLOCK_K], a_buf.index(slot), warp_used_hint=0b00001111)
        tdm.async_load(b_desc, [0, producer * BLOCK_K], b_buf.index(slot), warp_used_hint=0b00001111)
        producer += 1

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=wmma)
    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max >= NUM_BUFFERS)
    consumer = 0

    for _ in range(0, iter_max - NUM_BUFFERS):
        slot = producer % nbuf
        _issue_fp8_scaled_slice_mnk_fused_scale_load(as_desc, bs_desc, as_buf, bs_buf, producer, slot,
                                                     bk_scale_preshuffled)
        tdm.async_load(a_desc, [0, producer * BLOCK_K], a_buf.index(slot), warp_used_hint=0b00001111)
        tdm.async_load(b_desc, [0, producer * BLOCK_K], b_buf.index(slot), warp_used_hint=0b00001111)
        producer += 1
        acc = _issue_fp8_scaled_cluster_bk128_tile(
            a_buf, b_buf, as_buf, bs_buf, consumer % nbuf, acc, DTYPE_A, DTYPE_B, dot_a, dot_b, scale_a_layout,
            scale_b_layout, BLOCK_M, BLOCK_N, bk_scale, preshuffle_factor, scale_kwidth,
            requests_per_tile * (NUM_BUFFERS - 1))
        consumer += 1

    slot = producer % nbuf
    _issue_fp8_scaled_slice_mnk_fused_scale_load(as_desc, bs_desc, as_buf, bs_buf, producer, slot,
                                                 bk_scale_preshuffled)
    tdm.async_load(a_desc, [0, producer * BLOCK_K], a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(b_desc, [0, producer * BLOCK_K], b_buf.index(slot), warp_used_hint=0b00001111)

    for i in gl.static_range(NUM_BUFFERS):
        acc = _issue_fp8_scaled_cluster_bk128_tile(
            a_buf, b_buf, as_buf, bs_buf, consumer % nbuf, acc, DTYPE_A, DTYPE_B, dot_a, dot_b, scale_a_layout,
            scale_b_layout, BLOCK_M, BLOCK_N, bk_scale, preshuffle_factor, scale_kwidth,
            requests_per_tile * (NUM_BUFFERS - 1 - i))
        consumer += 1

    _store_full_tile(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc, wmma, BLOCK_M, BLOCK_N)


@gluon.jit
def _mxfp4_slice_mn_warp_pipeline_tutorial_bk128_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
        stride_cn, stride_scale, SCALE_BLOCK: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr, GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr,
        NUM_WARPS: gl.constexpr):
    gl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 128)
    gl.static_assert(SCALE_BLOCK == 16 or SCALE_BLOCK == 32)
    gl.static_assert(NUM_WARPS == 8)
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    half_m: gl.constexpr = BLOCK_M // 2
    half_n: gl.constexpr = BLOCK_N // 2
    bk_packed: gl.constexpr = BLOCK_K // 2
    bk_scale: gl.constexpr = BLOCK_K // SCALE_BLOCK
    preshuffle_factor: gl.constexpr = 128
    bk_scale_preshuffled: gl.constexpr = bk_scale * preshuffle_factor
    half_m_preshuffled: gl.constexpr = half_m // preshuffle_factor
    half_n_preshuffled: gl.constexpr = half_n // preshuffle_factor
    scale_kwidth: gl.constexpr = 4 if bk_scale >= 4 else bk_scale

    padded_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[bk_packed if bk_packed >= 256 else 256, 16]], [half_m, bk_packed], [1, 0])
    padded_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[bk_packed if bk_packed >= 256 else 256, 16]], [half_n, bk_packed], [1, 0])
    shared_scale: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [half_m_preshuffled, bk_scale_preshuffled], [1, 0])
    layouts: gl.constexpr = gl.amd.gfx1250.make_partitioned_dot_layouts(
        half_m, half_n, padded_a, padded_b, NUM_WARPS, [32, 16, 128], a_transposed=False, b_transposed=True)
    shared_a: gl.constexpr = layouts[0]
    shared_b: gl.constexpr = layouts[1]
    wmma: gl.constexpr = layouts[2]
    wmma_packed: gl.constexpr = gl.amd.AMDWMMALayout(3, True, wmma.warp_bases, wmma.reg_bases, [32, 16, 64])
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma_packed, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma_packed, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_a, [half_m, bk_scale])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_b, [half_n, bk_scale])

    nbuf: gl.constexpr = 2
    a_top_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, half_m, bk_packed], shared_a)
    a_bot_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, half_m, bk_packed], shared_a)
    b_left_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, half_n, bk_packed], shared_b)
    b_right_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, half_n, bk_packed], shared_b)
    as_top_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty,
                                           [nbuf, half_m_preshuffled, bk_scale_preshuffled], shared_scale)
    as_bot_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty,
                                           [nbuf, half_m_preshuffled, bk_scale_preshuffled], shared_scale)
    bs_left_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty,
                                            [nbuf, half_n_preshuffled, bk_scale_preshuffled], shared_scale)
    bs_right_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty,
                                             [nbuf, half_n_preshuffled, bk_scale_preshuffled], shared_scale)

    a_base = pid_m * BLOCK_M * stride_am
    b_base = pid_n * BLOCK_N * stride_bn
    a_top_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base, shape=(M, K // 2),
                                            strides=(stride_am, stride_ak), block_shape=(half_m, bk_packed),
                                            layout=shared_a)
    a_bot_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base + half_m * stride_am, shape=(M, K // 2),
                                            strides=(stride_am, stride_ak), block_shape=(half_m, bk_packed),
                                            layout=shared_a)
    b_left_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base, shape=(N, K // 2),
                                             strides=(stride_bn, stride_bk), block_shape=(half_n, bk_packed),
                                             layout=shared_b)
    b_right_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base + half_n * stride_bn, shape=(N, K // 2),
                                              strides=(stride_bn, stride_bk), block_shape=(half_n, bk_packed),
                                              layout=shared_b)

    as_top_base = (pid_m * BLOCK_M) // preshuffle_factor * stride_scale
    as_bot_base = (pid_m * BLOCK_M + half_m) // preshuffle_factor * stride_scale
    bs_left_base = (pid_n * BLOCK_N) // preshuffle_factor * stride_scale
    bs_right_base = (pid_n * BLOCK_N + half_n) // preshuffle_factor * stride_scale
    as_top_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + as_top_base, shape=(M // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(half_m_preshuffled, bk_scale_preshuffled), layout=shared_scale)
    as_bot_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + as_bot_base, shape=(M // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(half_m_preshuffled, bk_scale_preshuffled), layout=shared_scale)
    bs_left_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + bs_left_base, shape=(N // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(half_n_preshuffled, bk_scale_preshuffled), layout=shared_scale)
    bs_right_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + bs_right_base, shape=(N // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(half_n_preshuffled, bk_scale_preshuffled), layout=shared_scale)

    for i in gl.static_range(2):
        _issue_mxfp4_fused_scale_load(as_top_desc, as_bot_desc, bs_left_desc, bs_right_desc, i, as_top_buf, as_bot_buf,
                                      bs_left_buf, bs_right_buf, i, bk_scale_preshuffled)
        tdm.async_load(b_left_desc, [0, i * bk_packed], b_left_buf.index(i))
        tdm.async_load(a_top_desc, [0, i * bk_packed], a_top_buf.index(i))
        tdm.async_load(a_bot_desc, [0, i * bk_packed], a_bot_buf.index(i))
        tdm.async_load(b_right_desc, [0, i * bk_packed], b_right_buf.index(i))

    tdm.async_wait(7)
    b_left = b_left_buf.index(0).permute([1, 0]).load(layout=dot_b)
    bs_left = _load_mxfp_scale(bs_left_buf, 0, scale_b_layout, half_n, bk_scale, True, preshuffle_factor,
                               scale_kwidth)
    a_top = a_top_buf.index(0).load(layout=dot_a)
    as_top = _load_mxfp_scale(as_top_buf, 0, scale_a_layout, half_m, bk_scale, True, preshuffle_factor,
                              scale_kwidth)

    acc_tl = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    acc_bl = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    acc_tr = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    acc_br = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max > 3)

    for k in range(0, iter_max - 2, 2):
        tdm.async_wait(6)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            as_bot = _load_mxfp_scale(as_bot_buf, 0, scale_a_layout, half_m, bk_scale, True, preshuffle_factor,
                                      scale_kwidth)
            bs_right = _load_mxfp_scale(bs_right_buf, 0, scale_b_layout, half_n, bk_scale, True, preshuffle_factor,
                                        scale_kwidth)
            acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, "e2m1", b_left, bs_left, "e2m1", acc_tl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = a_bot_buf.index(0).load(layout=dot_a)
            tdm.async_load(b_left_desc, [0, (k + 2) * bk_packed], b_left_buf.index(0))

        tdm.async_wait(6)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, "e2m1", b_left, bs_left, "e2m1", acc_bl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = b_right_buf.index(0).permute([1, 0]).load(layout=dot_b)
            tdm.async_load(a_top_desc, [0, (k + 2) * bk_packed], a_top_buf.index(0))

        tdm.async_wait(6)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, "e2m1", b_right, bs_right, "e2m1", acc_tr)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = b_left_buf.index(1).permute([1, 0]).load(layout=dot_b)
            bs_left = _load_mxfp_scale(bs_left_buf, 1, scale_b_layout, half_n, bk_scale, True, preshuffle_factor,
                                       scale_kwidth)
            tdm.async_load(a_bot_desc, [0, (k + 2) * bk_packed], a_bot_buf.index(0))

        tdm.async_wait(6)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, "e2m1", b_right, bs_right, "e2m1", acc_br)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = a_top_buf.index(1).load(layout=dot_a)
            as_top = _load_mxfp_scale(as_top_buf, 1, scale_a_layout, half_m, bk_scale, True, preshuffle_factor,
                                      scale_kwidth)
            _issue_mxfp4_fused_scale_load(as_top_desc, as_bot_desc, bs_left_desc, bs_right_desc, k + 2, as_top_buf,
                                          as_bot_buf, bs_left_buf, bs_right_buf, 0, bk_scale_preshuffled)
            tdm.async_load(b_right_desc, [0, (k + 2) * bk_packed], b_right_buf.index(0))

        tdm.async_wait(6)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            as_bot = _load_mxfp_scale(as_bot_buf, 1, scale_a_layout, half_m, bk_scale, True, preshuffle_factor,
                                      scale_kwidth)
            bs_right = _load_mxfp_scale(bs_right_buf, 1, scale_b_layout, half_n, bk_scale, True, preshuffle_factor,
                                        scale_kwidth)
            acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, "e2m1", b_left, bs_left, "e2m1", acc_tl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = a_bot_buf.index(1).load(layout=dot_a)
            tdm.async_load(b_left_desc, [0, (k + 3) * bk_packed], b_left_buf.index(1))

        tdm.async_wait(6)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, "e2m1", b_left, bs_left, "e2m1", acc_bl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = b_right_buf.index(1).permute([1, 0]).load(layout=dot_b)
            tdm.async_load(a_top_desc, [0, (k + 3) * bk_packed], a_top_buf.index(1))

        tdm.async_wait(6)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, "e2m1", b_right, bs_right, "e2m1", acc_tr)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = b_left_buf.index(0).permute([1, 0]).load(layout=dot_b)
            bs_left = _load_mxfp_scale(bs_left_buf, 0, scale_b_layout, half_n, bk_scale, True, preshuffle_factor,
                                       scale_kwidth)
            tdm.async_load(a_bot_desc, [0, (k + 3) * bk_packed], a_bot_buf.index(1))

        tdm.async_wait(6)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, "e2m1", b_right, bs_right, "e2m1", acc_br)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = a_top_buf.index(0).load(layout=dot_a)
            as_top = _load_mxfp_scale(as_top_buf, 0, scale_a_layout, half_m, bk_scale, True, preshuffle_factor,
                                      scale_kwidth)
            _issue_mxfp4_fused_scale_load(as_top_desc, as_bot_desc, bs_left_desc, bs_right_desc, k + 3, as_top_buf,
                                          as_bot_buf, bs_left_buf, bs_right_buf, 1, bk_scale_preshuffled)
            tdm.async_load(b_right_desc, [0, (k + 3) * bk_packed], b_right_buf.index(1))

    acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, "e2m1", b_left, bs_left, "e2m1", acc_tl)
    tdm.async_wait(6)
    l_idx = (iter_max - 2) % 2
    a_bot = a_bot_buf.index(l_idx).load(layout=dot_a)
    as_bot = _load_mxfp_scale(as_bot_buf, l_idx, scale_a_layout, half_m, bk_scale, True, preshuffle_factor,
                              scale_kwidth)

    acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, "e2m1", b_left, bs_left, "e2m1", acc_bl)
    tdm.async_wait(5)
    b_right = b_right_buf.index(l_idx).permute([1, 0]).load(layout=dot_b)
    bs_right = _load_mxfp_scale(bs_right_buf, l_idx, scale_b_layout, half_n, bk_scale, True, preshuffle_factor,
                                scale_kwidth)

    acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, "e2m1", b_right, bs_right, "e2m1", acc_tr)
    tdm.async_wait(3)
    g_idx = 1 - l_idx
    b_left = b_left_buf.index(g_idx).permute([1, 0]).load(layout=dot_b)
    bs_left = _load_mxfp_scale(bs_left_buf, g_idx, scale_b_layout, half_n, bk_scale, True, preshuffle_factor,
                               scale_kwidth)

    acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, "e2m1", b_right, bs_right, "e2m1", acc_br)
    tdm.async_wait(2)
    a_top = a_top_buf.index(g_idx).load(layout=dot_a)
    as_top = _load_mxfp_scale(as_top_buf, g_idx, scale_a_layout, half_m, bk_scale, True, preshuffle_factor,
                              scale_kwidth)

    acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, "e2m1", b_left, bs_left, "e2m1", acc_tl)
    tdm.async_wait(1)
    a_bot = a_bot_buf.index(g_idx).load(layout=dot_a)
    as_bot = _load_mxfp_scale(as_bot_buf, g_idx, scale_a_layout, half_m, bk_scale, True, preshuffle_factor,
                              scale_kwidth)

    acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, "e2m1", b_left, bs_left, "e2m1", acc_bl)
    tdm.async_wait(0)
    b_right = b_right_buf.index(g_idx).permute([1, 0]).load(layout=dot_b)
    bs_right = _load_mxfp_scale(bs_right_buf, g_idx, scale_b_layout, half_n, bk_scale, True, preshuffle_factor,
                                scale_kwidth)

    acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, "e2m1", b_right, bs_right, "e2m1", acc_tr)
    acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, "e2m1", b_right, bs_right, "e2m1", acc_br)
    _store_quadrants(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc_tl, acc_bl, acc_tr, acc_br, wmma,
                     BLOCK_M, BLOCK_N)


# Optimized tutorial schedule compared with the original BK=128 implementation
# above:
#
# * Stage one full 256x256x256 A/B tile per LDS slot instead of four independent
#   128x128x128 quadrant allocations. The 2x2x2 SliceMNK views retain the four
#   output accumulators while rotating partition bases for each 128-row/column
#   subview.
# * Use one full A and one full B TDM descriptor, plus full non-partitioned scale
#   descriptors. Fusing the A/B scale copies reduces each BK=256 slot from four
#   requests to three, amortizing descriptor, wait, and refill overhead across
#   the work of two original BK=128 steps.
# * Interleave local SliceMNK operand loads with independent scaled WMMAs. A
#   dedicated TDM stage keeps fused multi-destination refills ordered after all
#   scale consumers while all eight waves participate in compute.
# * Refill a slot only after every M/N/K subtile consumer has completed. This is
#   required for scales as well as data because warp-pipeline phase shifting can
#   move the effective LDS load later than its source position.
#
# On the development gfx1250 system the BK=256 conversion changed the 8192^3
# benchmark median from 5,572 to 6,855 TFLOPS. Later TDM scheduling changes and
# fused scales reached 7,693 TFLOPS, with 378 VGPRs and zero scratch use.
@gluon.jit
def _load_mxfp4_slice_mnk_scale(scale_buffer, slot, start_nonk: gl.constexpr, start_k: gl.constexpr,
                                 LAYOUT: gl.constexpr, BLOCK_NONK: gl.constexpr, BK_SCALE: gl.constexpr,
                                 SUBTILE_NONK: gl.constexpr, SUBTILE_SCALE_K: gl.constexpr,
                                 PRESHUFFLE_FACTOR: gl.constexpr, SCALE_KWIDTH: gl.constexpr):
    scale_slice = scale_buffer.index(slot).reshape(
        (BLOCK_NONK // PRESHUFFLE_FACTOR, BK_SCALE // SCALE_KWIDTH, PRESHUFFLE_FACTOR // 4, 4,
         SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((BLOCK_NONK, BK_SCALE))
    return scale_slice.slice(start_nonk, SUBTILE_NONK, 0).slice(
        start_k, SUBTILE_SCALE_K, 1).load(layout=LAYOUT)


@gluon.jit
def _consume_mxfp4_slice_mnk_tile(a_buf, b_buf, as_buf, bs_buf, slot, acc_tl, acc_bl, acc_tr, acc_br,
                                  DOT_A: gl.constexpr, DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
                                  SCALE_B_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                                  BK_PACKED: gl.constexpr, BK_SCALE: gl.constexpr, SUBTILE_M: gl.constexpr,
                                  SUBTILE_N: gl.constexpr, SUBTILE_K_PACKED: gl.constexpr,
                                  SUBTILE_SCALE_K: gl.constexpr, PRESHUFFLE_FACTOR: gl.constexpr,
                                  SCALE_KWIDTH: gl.constexpr):
    with gl.amd.warp_pipeline_stage("mem", priority=1):
        a00 = a_buf.index(slot).slice(0, SUBTILE_M, 0).slice(0, SUBTILE_K_PACKED, 1).load(layout=DOT_A)
        as00 = _load_mxfp4_slice_mnk_scale(as_buf, slot, 0, 0, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE, SUBTILE_M,
                                           SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
        b00 = b_buf.index(slot).slice(0, SUBTILE_N, 0).slice(
            0, SUBTILE_K_PACKED, 1).permute([1, 0]).load(layout=DOT_B)
        bs00 = _load_mxfp4_slice_mnk_scale(bs_buf, slot, 0, 0, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE, SUBTILE_N,
                                           SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_tl = gl.amd.gfx1250.wmma_scaled(a00, as00, "e2m1", b00, bs00, "e2m1", acc_tl)

    with gl.amd.warp_pipeline_stage("mem", priority=1):
        b01 = b_buf.index(slot).slice(SUBTILE_N, SUBTILE_N, 0).slice(
            0, SUBTILE_K_PACKED, 1).permute([1, 0]).load(layout=DOT_B)
        bs01 = _load_mxfp4_slice_mnk_scale(bs_buf, slot, SUBTILE_N, 0, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE,
                                           SUBTILE_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_tr = gl.amd.gfx1250.wmma_scaled(a00, as00, "e2m1", b01, bs01, "e2m1", acc_tr)

    with gl.amd.warp_pipeline_stage("mem", priority=1):
        a10 = a_buf.index(slot).slice(SUBTILE_M, SUBTILE_M, 0).slice(
            0, SUBTILE_K_PACKED, 1).load(layout=DOT_A)
        as10 = _load_mxfp4_slice_mnk_scale(as_buf, slot, SUBTILE_M, 0, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE,
                                           SUBTILE_M, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_bl = gl.amd.gfx1250.wmma_scaled(a10, as10, "e2m1", b00, bs00, "e2m1", acc_bl)

    with gl.amd.warp_pipeline_stage("mem", priority=1):
        b10 = b_buf.index(slot).slice(0, SUBTILE_N, 0).slice(
            SUBTILE_K_PACKED, SUBTILE_K_PACKED, 1).permute([1, 0]).load(layout=DOT_B)
        bs10 = _load_mxfp4_slice_mnk_scale(bs_buf, slot, 0, SUBTILE_SCALE_K, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE,
                                           SUBTILE_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_br = gl.amd.gfx1250.wmma_scaled(a10, as10, "e2m1", b01, bs01, "e2m1", acc_br)

    with gl.amd.warp_pipeline_stage("mem", priority=1):
        a01 = a_buf.index(slot).slice(0, SUBTILE_M, 0).slice(
            SUBTILE_K_PACKED, SUBTILE_K_PACKED, 1).load(layout=DOT_A)
        as01 = _load_mxfp4_slice_mnk_scale(as_buf, slot, 0, SUBTILE_SCALE_K, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE,
                                           SUBTILE_M, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_tl = gl.amd.gfx1250.wmma_scaled(a01, as01, "e2m1", b10, bs10, "e2m1", acc_tl)

    with gl.amd.warp_pipeline_stage("mem", priority=1):
        b11 = b_buf.index(slot).slice(SUBTILE_N, SUBTILE_N, 0).slice(
            SUBTILE_K_PACKED, SUBTILE_K_PACKED, 1).permute([1, 0]).load(layout=DOT_B)
        bs11 = _load_mxfp4_slice_mnk_scale(bs_buf, slot, SUBTILE_N, SUBTILE_SCALE_K, SCALE_B_LAYOUT, BLOCK_N,
                                           BK_SCALE, SUBTILE_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_tr = gl.amd.gfx1250.wmma_scaled(a01, as01, "e2m1", b11, bs11, "e2m1", acc_tr)

    with gl.amd.warp_pipeline_stage("mem", priority=1):
        a11 = a_buf.index(slot).slice(SUBTILE_M, SUBTILE_M, 0).slice(
            SUBTILE_K_PACKED, SUBTILE_K_PACKED, 1).load(layout=DOT_A)
        as11 = _load_mxfp4_slice_mnk_scale(as_buf, slot, SUBTILE_M, SUBTILE_SCALE_K, SCALE_A_LAYOUT, BLOCK_M,
                                           BK_SCALE, SUBTILE_M, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_bl = gl.amd.gfx1250.wmma_scaled(a11, as11, "e2m1", b10, bs10, "e2m1", acc_bl)
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        acc_br = gl.amd.gfx1250.wmma_scaled(a11, as11, "e2m1", b11, bs11, "e2m1", acc_br)
    return acc_tl, acc_bl, acc_tr, acc_br


@gluon.jit
def _issue_mxfp4_slice_mnk_fused_scale_load(as_desc, bs_desc, as_buf, bs_buf, tile_idx, slot,
                                            BK_SCALE_PRESHUFFLED: gl.constexpr):
    scale_k = tile_idx * BK_SCALE_PRESHUFFLED
    as_load_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, scale_k])
    bs_load_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, scale_k])
    tdm.async_load_fused([
        (as_load_desc, as_buf.index(slot), 0b00001111),
        (bs_load_desc, bs_buf.index(slot), 0b11110000),
    ])


@gluon.jit
def mxfp4_slice_mn_warp_pipeline_tutorial_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
        stride_cn, stride_scale, SCALE_BLOCK: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr, GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr,
        NUM_WARPS: gl.constexpr):
    gl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 256)
    gl.static_assert(SCALE_BLOCK == 16 or SCALE_BLOCK == 32)
    gl.static_assert(NUM_WARPS == 8)
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    num_subtiles: gl.constexpr = (2, 2, 2)
    subtile_m: gl.constexpr = BLOCK_M // num_subtiles[0]
    subtile_n: gl.constexpr = BLOCK_N // num_subtiles[1]
    subtile_k: gl.constexpr = BLOCK_K // num_subtiles[2]
    bk_packed: gl.constexpr = BLOCK_K // 2
    subtile_k_packed: gl.constexpr = subtile_k // 2
    bk_scale: gl.constexpr = BLOCK_K // SCALE_BLOCK
    subtile_scale_k: gl.constexpr = bk_scale // num_subtiles[2]
    preshuffle_factor: gl.constexpr = 128
    bk_scale_preshuffled: gl.constexpr = bk_scale * preshuffle_factor
    block_m_preshuffled: gl.constexpr = BLOCK_M // preshuffle_factor
    block_n_preshuffled: gl.constexpr = BLOCK_N // preshuffle_factor
    scale_kwidth: gl.constexpr = 4 if bk_scale >= 4 else bk_scale

    padded_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256 if bk_packed <= 256 else bk_packed, 16]], [BLOCK_M, bk_packed], [1, 0])
    padded_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256 if bk_packed <= 256 else bk_packed, 16]], [BLOCK_N, bk_packed], [1, 0])
    shared_scale: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_m_preshuffled, bk_scale_preshuffled], [1, 0])
    layouts: gl.constexpr = gl.amd.gfx1250.make_partitioned_dot_layouts(
        BLOCK_M, BLOCK_N, padded_a, padded_b, NUM_WARPS, [32, 16, 128], a_transposed=False, b_transposed=True,
        slice_m=subtile_m, slice_n=subtile_n)
    shared_a: gl.constexpr = layouts[0]
    shared_b: gl.constexpr = layouts[1]
    wmma: gl.constexpr = layouts[2]
    wmma_packed: gl.constexpr = gl.amd.AMDWMMALayout(3, True, wmma.warp_bases, wmma.reg_bases, [32, 16, 64])
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma_packed, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma_packed, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_a,
                                                                       [subtile_m, subtile_scale_k])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_b,
                                                                       [subtile_n, subtile_scale_k])

    nbuf: gl.constexpr = 2
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, BLOCK_M, bk_packed], shared_a)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, BLOCK_N, bk_packed], shared_b)
    as_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty,
                                       [nbuf, block_m_preshuffled, bk_scale_preshuffled], shared_scale)
    bs_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty,
                                       [nbuf, block_n_preshuffled, bk_scale_preshuffled], shared_scale)

    a_base = pid_m * BLOCK_M * stride_am
    b_base = pid_n * BLOCK_N * stride_bn
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base, shape=(M, K // 2), strides=(stride_am, stride_ak),
                                        block_shape=(BLOCK_M, bk_packed), layout=shared_a)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base, shape=(N, K // 2), strides=(stride_bn, stride_bk),
                                        block_shape=(BLOCK_N, bk_packed), layout=shared_b)
    as_base = (pid_m * BLOCK_M) // preshuffle_factor * stride_scale
    bs_base = (pid_n * BLOCK_N) // preshuffle_factor * stride_scale
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + as_base, shape=(M // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(block_m_preshuffled, bk_scale_preshuffled), layout=shared_scale)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + bs_base, shape=(N // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(block_n_preshuffled, bk_scale_preshuffled), layout=shared_scale)

    # Three requests form one complete slot. Keep the larger A/B data copies
    # independent and fuse only the equally sized A/B scale transfers.
    for prefetch_idx in gl.static_range(2):
        tdm.async_load(b_desc, [0, prefetch_idx * bk_packed], b_buf.index(prefetch_idx),
                       warp_used_hint=0b00001111)
        tdm.async_load(a_desc, [0, prefetch_idx * bk_packed], a_buf.index(prefetch_idx),
                       warp_used_hint=0b00001111)
        _issue_mxfp4_slice_mnk_fused_scale_load(as_desc, bs_desc, as_buf, bs_buf, prefetch_idx, prefetch_idx,
                                                bk_scale_preshuffled)

    # Complete slot 0 while retaining the three slot-1 requests.
    tdm.async_wait(3)
    acc_tl = gl.zeros((subtile_m, subtile_n), dtype=gl.float32, layout=wmma)
    acc_bl = gl.zeros((subtile_m, subtile_n), dtype=gl.float32, layout=wmma)
    acc_tr = gl.zeros((subtile_m, subtile_n), dtype=gl.float32, layout=wmma)
    acc_br = gl.zeros((subtile_m, subtile_n), dtype=gl.float32, layout=wmma)

    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max >= 2)
    for tile_idx in range(0, iter_max - 2):
        slot = tile_idx % nbuf
        acc_tl, acc_bl, acc_tr, acc_br = _consume_mxfp4_slice_mnk_tile(
            a_buf, b_buf, as_buf, bs_buf, slot, acc_tl, acc_bl, acc_tr, acc_br, dot_a, dot_b, scale_a_layout,
            scale_b_layout, BLOCK_M, BLOCK_N, bk_packed, bk_scale, subtile_m, subtile_n, subtile_k_packed,
            subtile_scale_k, preshuffle_factor, scale_kwidth)

        # Refill only after all eight M/N/K consumer WMMAs have completed.
        refill_idx = tile_idx + 2
        with gl.amd.warp_pipeline_stage("tdm", priority=1):
            tdm.async_load(b_desc, [0, refill_idx * bk_packed], b_buf.index(slot),
                           warp_used_hint=0b00001111)
            tdm.async_load(a_desc, [0, refill_idx * bk_packed], a_buf.index(slot),
                           warp_used_hint=0b00001111)
            _issue_mxfp4_slice_mnk_fused_scale_load(as_desc, bs_desc, as_buf, bs_buf, refill_idx, slot,
                                                    bk_scale_preshuffled)
        # Complete the next read slot and retain exactly the three refill requests.
        tdm.async_wait(3)

    penultimate_idx = iter_max - 2
    penultimate_slot = penultimate_idx % nbuf
    acc_tl, acc_bl, acc_tr, acc_br = _consume_mxfp4_slice_mnk_tile(
        a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc_tl, acc_bl, acc_tr, acc_br, dot_a, dot_b,
        scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N, bk_packed, bk_scale, subtile_m, subtile_n,
        subtile_k_packed, subtile_scale_k, preshuffle_factor, scale_kwidth)

    # No refill is needed in the tail, so drain the final slot exactly.
    tdm.async_wait(0)
    last_slot = (iter_max - 1) % nbuf
    acc_tl, acc_bl, acc_tr, acc_br = _consume_mxfp4_slice_mnk_tile(
        a_buf, b_buf, as_buf, bs_buf, last_slot, acc_tl, acc_bl, acc_tr, acc_br, dot_a, dot_b, scale_a_layout,
        scale_b_layout, BLOCK_M, BLOCK_N, bk_packed, bk_scale, subtile_m, subtile_n, subtile_k_packed,
        subtile_scale_k, preshuffle_factor, scale_kwidth)

    _store_quadrants(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc_tl, acc_bl, acc_tr, acc_br, wmma,
                     BLOCK_M, BLOCK_N)


def _build_mxfp4_bk512_cluster_layouts(
        block_m, block_n, block_k, num_warps, cga_layout, cta_m, cta_n):
    """Build partitioned MXFP4 operand layouts for the requested cluster."""
    bk_packed = block_k // 2
    slice_m = block_m // cta_m
    slice_n = block_n // cta_n

    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[bk_packed, 16]], [block_m, bk_packed], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[bk_packed, 16]], [block_n, bk_packed], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, block_n, local_a, local_b, num_warps, [32, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=slice_m, slice_n=slice_n)

    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases, local_wmma.reg_bases,
        local_wmma.instr_shape, cga_layout)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]]) for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[bk_packed, 16]], [block_m, bk_packed], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[bk_packed, 16]], [block_n, bk_packed], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        slice_m, slice_n, padded_a, padded_b, num_warps, [32, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=slice_m, slice_n=slice_n)

    bk_scale = block_k // 32
    preshuffle_factor = 128
    shared_scale_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_m // preshuffle_factor, bk_scale * preshuffle_factor],
        [1, 0], cga_a)
    shared_scale_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_n // preshuffle_factor, bk_scale * preshuffle_factor],
        [1, 0], cga_b)
    return shared_a, shared_b, shared_scale_a, shared_scale_b, wmma


# This BK512 variant ports the leading-four pipeline used by the parameterized
# MXFP8 kernel to packed E2M1 operands. Each CTA computes 256x256x512 within a
# parameterized cluster, using two physical LDS partitions per operand, two fused TDM
# requests per slot, four logical K=128 scaled-WMMA steps, and a padded BF16
# cluster-aware TDM-store epilogue.
@gluon.jit
def _issue_mxfp4_bk512_leading4_refill(
        a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, tile_idx, slot,
        BK_PACKED: gl.constexpr, BK_SCALE_PRESHUFFLED: gl.constexpr):
    """Issue one BK512 data/scale refill from the leading four warps."""
    packed_k = tile_idx * BK_PACKED
    a_load = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, packed_k])
    b_load = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, packed_k])
    tdm.async_load_fused([
        (a_load, a_buf.index(slot), 0b00000011),
        (b_load, b_buf.index(slot), 0b00001100),
    ])

    scale_k = tile_idx * BK_SCALE_PRESHUFFLED
    as_load = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, scale_k])
    bs_load = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, scale_k])
    tdm.async_load_fused([
        (as_load, as_buf.index(slot), 0b00000011),
        (bs_load, bs_buf.index(slot), 0b00001100),
    ])


@gluon.jit
def _consume_mxfp4_bk512_k_step(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, START_K_PACKED: gl.constexpr,
        START_SCALE_K: gl.constexpr, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BK_SCALE: gl.constexpr,
        SUBTILE_K_PACKED: gl.constexpr, SUBTILE_SCALE_K: gl.constexpr,
        PRESHUFFLE_FACTOR: gl.constexpr, SCALE_KWIDTH: gl.constexpr,
        OVERLAP_CLUSTER: gl.constexpr):
    """Consume one logical K=128 slice of a staged BK512 tile."""
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            START_K_PACKED, SUBTILE_K_PACKED, 1).load(layout=DOT_A)
        as_ = _load_mxfp4_slice_mnk_scale(
            as_buf, slot, 0, START_SCALE_K, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE,
            BLOCK_M, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
        b = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            START_K_PACKED, SUBTILE_K_PACKED, 1).permute([1, 0]).load(layout=DOT_B)
        bs = _load_mxfp4_slice_mnk_scale(
            bs_buf, slot, 0, START_SCALE_K, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE,
            BLOCK_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
        if OVERLAP_CLUSTER:
            gl.amd.gfx1250.cluster.arrive()
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(a, as_, "e2m1", b, bs, "e2m1", acc)
        if OVERLAP_CLUSTER:
            gl.amd.gfx1250.cluster.wait()
    return acc


@gluon.jit
def _consume_mxfp4_bk512_k256_step(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, START_K_PACKED: gl.constexpr,
        START_SCALE_K: gl.constexpr, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BK_SCALE: gl.constexpr,
        SUBTILE_K_PACKED: gl.constexpr, SUBTILE_SCALE_K: gl.constexpr,
        PRESHUFFLE_FACTOR: gl.constexpr, SCALE_KWIDTH: gl.constexpr,
        OVERLAP_CLUSTER: gl.constexpr):
    """Load K256 in one warp-pipeline region and consume it in the next."""
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            START_K_PACKED, SUBTILE_K_PACKED, 1).load(layout=DOT_A)
        as0 = _load_mxfp4_slice_mnk_scale(
            as_buf, slot, 0, START_SCALE_K, SCALE_A_LAYOUT, BLOCK_M, BK_SCALE,
            BLOCK_M, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)
        b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            START_K_PACKED, SUBTILE_K_PACKED, 1).permute([1, 0]).load(layout=DOT_B)
        bs0 = _load_mxfp4_slice_mnk_scale(
            bs_buf, slot, 0, START_SCALE_K, SCALE_B_LAYOUT, BLOCK_N, BK_SCALE,
            BLOCK_N, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH)

        a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            START_K_PACKED + SUBTILE_K_PACKED, SUBTILE_K_PACKED, 1).load(layout=DOT_A)
        as1 = _load_mxfp4_slice_mnk_scale(
            as_buf, slot, 0, START_SCALE_K + SUBTILE_SCALE_K,
            SCALE_A_LAYOUT, BLOCK_M, BK_SCALE, BLOCK_M, SUBTILE_SCALE_K,
            PRESHUFFLE_FACTOR, SCALE_KWIDTH)
        b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            START_K_PACKED + SUBTILE_K_PACKED, SUBTILE_K_PACKED, 1
        ).permute([1, 0]).load(layout=DOT_B)
        bs1 = _load_mxfp4_slice_mnk_scale(
            bs_buf, slot, 0, START_SCALE_K + SUBTILE_SCALE_K,
            SCALE_B_LAYOUT, BLOCK_N, BK_SCALE, BLOCK_N, SUBTILE_SCALE_K,
            PRESHUFFLE_FACTOR, SCALE_KWIDTH)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(a0, as0, "e2m1", b0, bs0, "e2m1", acc)
        if OVERLAP_CLUSTER:
            gl.amd.gfx1250.cluster.arrive()
        acc = gl.amd.gfx1250.wmma_scaled(a1, as1, "e2m1", b1, bs1, "e2m1", acc)
        if OVERLAP_CLUSTER:
            gl.amd.gfx1250.cluster.wait()
    return acc


@gluon.jit
def _consume_mxfp4_bk512_tile(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BK_SCALE: gl.constexpr,
        SUBTILE_K_PACKED: gl.constexpr, SUBTILE_SCALE_K: gl.constexpr,
        PRESHUFFLE_FACTOR: gl.constexpr, SCALE_KWIDTH: gl.constexpr,
        K256_STAGES: gl.constexpr, OVERLAP_CLUSTER: gl.constexpr):
    """Consume BK512 as four whole-CTA scaled-WMMA K=128 steps."""
    if K256_STAGES:
        acc = _consume_mxfp4_bk512_k256_step(
            a_buf, b_buf, as_buf, bs_buf, slot, acc, 0, 0, DOT_A, DOT_B,
            SCALE_A_LAYOUT, SCALE_B_LAYOUT, BLOCK_M, BLOCK_N, BK_SCALE,
            SUBTILE_K_PACKED, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH,
            False)
        return _consume_mxfp4_bk512_k256_step(
            a_buf, b_buf, as_buf, bs_buf, slot, acc, 2 * SUBTILE_K_PACKED,
            2 * SUBTILE_SCALE_K, DOT_A, DOT_B, SCALE_A_LAYOUT, SCALE_B_LAYOUT,
            BLOCK_M, BLOCK_N, BK_SCALE, SUBTILE_K_PACKED, SUBTILE_SCALE_K,
            PRESHUFFLE_FACTOR, SCALE_KWIDTH, OVERLAP_CLUSTER)

    acc = _consume_mxfp4_bk512_k_step(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, 0, 0, DOT_A, DOT_B,
        SCALE_A_LAYOUT, SCALE_B_LAYOUT, BLOCK_M, BLOCK_N, BK_SCALE,
        SUBTILE_K_PACKED, SUBTILE_SCALE_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH,
        False)
    acc = _consume_mxfp4_bk512_k_step(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, SUBTILE_K_PACKED,
        SUBTILE_SCALE_K, DOT_A, DOT_B, SCALE_A_LAYOUT, SCALE_B_LAYOUT,
        BLOCK_M, BLOCK_N, BK_SCALE, SUBTILE_K_PACKED, SUBTILE_SCALE_K,
        PRESHUFFLE_FACTOR, SCALE_KWIDTH, False)
    acc = _consume_mxfp4_bk512_k_step(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, 2 * SUBTILE_K_PACKED,
        2 * SUBTILE_SCALE_K, DOT_A, DOT_B, SCALE_A_LAYOUT, SCALE_B_LAYOUT,
        BLOCK_M, BLOCK_N, BK_SCALE, SUBTILE_K_PACKED, SUBTILE_SCALE_K,
        PRESHUFFLE_FACTOR, SCALE_KWIDTH, False)
    return _consume_mxfp4_bk512_k_step(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, 3 * SUBTILE_K_PACKED,
        3 * SUBTILE_SCALE_K, DOT_A, DOT_B, SCALE_A_LAYOUT, SCALE_B_LAYOUT,
        BLOCK_M, BLOCK_N, BK_SCALE, SUBTILE_K_PACKED, SUBTILE_SCALE_K,
        PRESHUFFLE_FACTOR, SCALE_KWIDTH, OVERLAP_CLUSTER)


@gluon.jit
def mxfp4_bk512_warp_pipeline_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, SCALE_BLOCK: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
        GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr, NUM_WARPS: gl.constexpr,
        SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr,
        SHARED_SCALE_A: gl.constexpr, SHARED_SCALE_B: gl.constexpr,
        WMMA_LAYOUT: gl.constexpr, CTA_M: gl.constexpr, CTA_N: gl.constexpr,
        K256_STAGES: gl.constexpr, OVERLAP_CLUSTER: gl.constexpr):
    """Clustered MXFP4 GEMM with a 256x256x512 tile per CTA."""
    gl.static_assert(
        (CTA_M == 2 and CTA_N == 2) or (CTA_M == 2 and CTA_N == 4)
        or (CTA_M == 4 and CTA_N == 2) or (CTA_M == 4 and CTA_N == 4))
    gl.static_assert(BLOCK_M == CTA_M * 256 and BLOCK_N == CTA_N * 256 and BLOCK_K == 512)
    gl.static_assert(SCALE_BLOCK == 32)
    gl.static_assert(NUM_WARPS == 8)
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    pid_m, pid_n = get_xcd_swizzled_pids(
        M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    num_k_steps: gl.constexpr = 4
    bk_packed: gl.constexpr = BLOCK_K // 2
    subtile_k_packed: gl.constexpr = bk_packed // num_k_steps
    bk_scale: gl.constexpr = BLOCK_K // SCALE_BLOCK
    subtile_scale_k: gl.constexpr = bk_scale // num_k_steps
    preshuffle_factor: gl.constexpr = 128
    bk_scale_preshuffled: gl.constexpr = bk_scale * preshuffle_factor
    block_m_preshuffled: gl.constexpr = BLOCK_M // preshuffle_factor
    block_n_preshuffled: gl.constexpr = BLOCK_N // preshuffle_factor
    scale_kwidth: gl.constexpr = 4
    nbuf: gl.constexpr = 2

    wmma_packed: gl.constexpr = gl.amd.AMDWMMALayout(
        3, WMMA_LAYOUT.transposed, WMMA_LAYOUT.warp_bases, WMMA_LAYOUT.reg_bases,
        [32, 16, 64], WMMA_LAYOUT.cga_layout)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma_packed, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma_packed, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [BLOCK_M, subtile_scale_k])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [BLOCK_N, subtile_scale_k])

    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [nbuf, BLOCK_M, bk_packed], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [nbuf, BLOCK_N, bk_packed], SHARED_LAYOUT_B)
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty,
        [nbuf, block_m_preshuffled, bk_scale_preshuffled], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty,
        [nbuf, block_n_preshuffled, bk_scale_preshuffled], SHARED_SCALE_B)

    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K // 2),
        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, bk_packed),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K // 2),
        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, bk_packed),
        layout=SHARED_LAYOUT_B)
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + (pid_m * BLOCK_M) // preshuffle_factor * stride_scale,
        shape=(M // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1),
        block_shape=(block_m_preshuffled, bk_scale_preshuffled),
        layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * BLOCK_N) // preshuffle_factor * stride_scale,
        shape=(N // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1),
        block_shape=(block_n_preshuffled, bk_scale_preshuffled),
        layout=SHARED_SCALE_B)

    for prefetch_idx in gl.static_range(nbuf):
        _issue_mxfp4_bk512_leading4_refill(
            a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
            prefetch_idx, prefetch_idx, bk_packed, bk_scale_preshuffled)

    wait_count: gl.constexpr = 2
    tdm.async_wait(wait_count)
    _cluster_wait(wait_count)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max >= nbuf)

    for tile_idx in range(0, iter_max - nbuf):
        slot = tile_idx % nbuf
        acc = _consume_mxfp4_bk512_tile(
            a_buf, b_buf, as_buf, bs_buf, slot, acc, dot_a, dot_b,
            scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N, bk_scale,
            subtile_k_packed, subtile_scale_k, preshuffle_factor, scale_kwidth,
            K256_STAGES, OVERLAP_CLUSTER)
        if not OVERLAP_CLUSTER:
            _cluster_wait(wait_count)
        with gl.amd.warp_pipeline_stage("tdm", priority=1):
            _issue_mxfp4_bk512_leading4_refill(
                a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
                tile_idx + nbuf, slot, bk_packed, bk_scale_preshuffled)
        tdm.async_wait(wait_count)

    penultimate_slot = (iter_max - 2) % nbuf
    acc = _consume_mxfp4_bk512_tile(
        a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N, bk_scale,
        subtile_k_packed, subtile_scale_k, preshuffle_factor, scale_kwidth,
        K256_STAGES, False)

    tdm.async_wait(0)
    _cluster_wait(0)
    last_slot = (iter_max - 1) % nbuf
    acc = _consume_mxfp4_bk512_tile(
        a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N, bk_scale,
        subtile_k_packed, subtile_scale_k, preshuffle_factor, scale_kwidth,
        K256_STAGES, False)
    _tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc, WMMA_LAYOUT,
        BLOCK_M, BLOCK_N, CTA_N)


@gluon.jit
def fp8_slice_mn_warp_pipeline_kernelC_gfx1250(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk,
                                              stride_bn, stride_cm, stride_cn, DTYPE_A: gl.constexpr,
                                              DTYPE_B: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                                              BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
                                              GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr,
                                              NUM_WARPS: gl.constexpr, NUM_BUFFERS: gl.constexpr,
                                              TDM_WARP_USED_HINT: gl.constexpr):
    gl.static_assert(DTYPE_A != "e2m1" and DTYPE_B != "e2m1",
                     "fp8_slice_mn_warp_pipeline_kernelC_gfx1250 requires FP8 inputs")
    gl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 128)
    gl.static_assert(NUM_WARPS == 8)
    gl.static_assert(NUM_BUFFERS == 3 or NUM_BUFFERS == 4, "FP8 kernelC requires three or four LDS buffers")
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    half_m: gl.constexpr = BLOCK_M // 2
    half_n: gl.constexpr = BLOCK_N // 2
    padded_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 16]], [half_m, BLOCK_K],
                                                                    [1, 0])
    padded_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 16]], [half_n, BLOCK_K],
                                                                    [1, 0])
    layouts: gl.constexpr = gl.amd.gfx1250.make_partitioned_dot_layouts(
        half_m, half_n, padded_a, padded_b, NUM_WARPS, [16, 16, 128], a_transposed=False, b_transposed=True)
    shared_a: gl.constexpr = layouts[0]
    shared_b: gl.constexpr = layouts[1]
    wmma: gl.constexpr = layouts[2]
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma, 16)

    nbuf: gl.constexpr = NUM_BUFFERS
    a_top_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, half_m, BLOCK_K], shared_a)
    a_bot_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, half_m, BLOCK_K], shared_a)
    b_left_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, half_n, BLOCK_K], shared_b)
    b_right_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, half_n, BLOCK_K], shared_b)

    a_base = pid_m * BLOCK_M * stride_am
    b_base = pid_n * BLOCK_N * stride_bn
    a_top_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base, shape=(M, K), strides=(stride_am, stride_ak),
                                            block_shape=(half_m, BLOCK_K), layout=shared_a)
    a_bot_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base + half_m * stride_am, shape=(M, K),
                                            strides=(stride_am, stride_ak), block_shape=(half_m, BLOCK_K),
                                            layout=shared_a)
    b_left_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base, shape=(N, K), strides=(stride_bn, stride_bk),
                                             block_shape=(half_n, BLOCK_K), layout=shared_b)
    b_right_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base + half_n * stride_bn, shape=(N, K),
                                              strides=(stride_bn, stride_bk), block_shape=(half_n, BLOCK_K),
                                              layout=shared_b)

    for i in gl.static_range(NUM_BUFFERS - 1):
        tdm.async_load(b_left_desc, [0, i * BLOCK_K], b_left_buf.index(i),
                       warp_used_hint=TDM_WARP_USED_HINT)
        tdm.async_load(a_top_desc, [0, i * BLOCK_K], a_top_buf.index(i), warp_used_hint=TDM_WARP_USED_HINT)
        tdm.async_load(a_bot_desc, [0, i * BLOCK_K], a_bot_buf.index(i), warp_used_hint=TDM_WARP_USED_HINT)
        tdm.async_load(b_right_desc, [0, i * BLOCK_K], b_right_buf.index(i),
                       warp_used_hint=TDM_WARP_USED_HINT)

    prefetch_wait: gl.constexpr = 4 * (NUM_BUFFERS - 1) - 2
    steady_wait: gl.constexpr = 4 * (NUM_BUFFERS - 1) - 3
    tdm.async_wait(prefetch_wait)
    a_top = a_top_buf.index(0).load(layout=dot_a)
    b_left = b_left_buf.index(0).permute([1, 0]).load(layout=dot_b)

    acc_tl = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    acc_bl = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    acc_tr = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    acc_br = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)

    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max > 3)
    consume_k = 0
    load_k = NUM_BUFFERS - 1
    for _ in range(0, iter_max - (NUM_BUFFERS - 1)):
        read_slot = consume_k % nbuf
        next_slot = (consume_k + 1) % nbuf
        write_slot = load_k % nbuf
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, None, DTYPE_A, b_left, None, DTYPE_B, acc_tl)
        tdm.async_wait(steady_wait)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = a_bot_buf.index(read_slot).load(layout=dot_a)
            tdm.async_load(b_left_desc, [0, load_k * BLOCK_K], b_left_buf.index(write_slot),
                           warp_used_hint=TDM_WARP_USED_HINT)

        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, None, DTYPE_A, b_left, None, DTYPE_B, acc_bl)
        tdm.async_wait(steady_wait)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = b_right_buf.index(read_slot).permute([1, 0]).load(layout=dot_b)
            tdm.async_load(a_top_desc, [0, load_k * BLOCK_K], a_top_buf.index(write_slot),
                           warp_used_hint=TDM_WARP_USED_HINT)

        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, None, DTYPE_A, b_right, None, DTYPE_B, acc_tr)
        tdm.async_wait(steady_wait)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = b_left_buf.index(next_slot).permute([1, 0]).load(layout=dot_b)
            tdm.async_load(a_bot_desc, [0, load_k * BLOCK_K], a_bot_buf.index(write_slot),
                           warp_used_hint=TDM_WARP_USED_HINT)

        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, None, DTYPE_A, b_right, None, DTYPE_B, acc_br)
        tdm.async_wait(steady_wait)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = a_top_buf.index(next_slot).load(layout=dot_a)
            tdm.async_load(b_right_desc, [0, load_k * BLOCK_K], b_right_buf.index(write_slot),
                           warp_used_hint=TDM_WARP_USED_HINT)
        consume_k += 1
        load_k += 1

    for i in gl.static_range(NUM_BUFFERS - 1):
        read_slot = (iter_max - (NUM_BUFFERS - 1 - i)) % nbuf
        acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, None, DTYPE_A, b_left, None, DTYPE_B, acc_tl)
        tdm.async_wait(4 * (NUM_BUFFERS - 1 - i) - 3)
        a_bot = a_bot_buf.index(read_slot).load(layout=dot_a)
        acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, None, DTYPE_A, b_left, None, DTYPE_B, acc_bl)
        tdm.async_wait(4 * (NUM_BUFFERS - 1 - i) - 4)
        b_right = b_right_buf.index(read_slot).permute([1, 0]).load(layout=dot_b)
        acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, None, DTYPE_A, b_right, None, DTYPE_B, acc_tr)
        if i < NUM_BUFFERS - 2:
            next_slot = (iter_max - (NUM_BUFFERS - 1 - i) + 1) % nbuf
            tdm.async_wait(4 * (NUM_BUFFERS - 2 - i) - 1)
            b_left = b_left_buf.index(next_slot).permute([1, 0]).load(layout=dot_b)
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, None, DTYPE_A, b_right, None, DTYPE_B, acc_br)
            tdm.async_wait(4 * (NUM_BUFFERS - 2 - i) - 2)
            a_top = a_top_buf.index(next_slot).load(layout=dot_a)
        else:
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, None, DTYPE_A, b_right, None, DTYPE_B, acc_br)

    _store_quadrants(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc_tl, acc_bl, acc_tr, acc_br, wmma,
                     BLOCK_M, BLOCK_N)


@gluon.jit
def fp8_scaled_slice_mn_warp_pipeline_kernelC_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
        stride_cn, stride_scale, DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr, SCALE_BLOCK: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
        GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr, NUM_WARPS: gl.constexpr, NUM_BUFFERS: gl.constexpr,
        TDM_WARP_USED_HINT: gl.constexpr, SCALE_PRESHUFFLE: gl.constexpr, ASYNC_COPY_SCALE: gl.constexpr):
    gl.static_assert(DTYPE_A != "e2m1" and DTYPE_B != "e2m1",
                     "fp8_scaled_slice_mn_warp_pipeline_kernelC_gfx1250 requires FP8 inputs")
    gl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 128)
    gl.static_assert(BLOCK_K % SCALE_BLOCK == 0)
    gl.static_assert(NUM_WARPS == 8)
    gl.static_assert(NUM_BUFFERS == 3 or NUM_BUFFERS == 4,
                     "Scaled FP8 kernelC requires three or four LDS buffers")
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    half_m: gl.constexpr = BLOCK_M // 2
    half_n: gl.constexpr = BLOCK_N // 2
    bk_scale: gl.constexpr = BLOCK_K // SCALE_BLOCK
    preshuffle_factor: gl.constexpr = 128 if SCALE_PRESHUFFLE else 1
    bk_scale_preshuffled: gl.constexpr = bk_scale * preshuffle_factor
    half_m_preshuffled: gl.constexpr = half_m // preshuffle_factor
    half_n_preshuffled: gl.constexpr = half_n // preshuffle_factor
    scale_kwidth: gl.constexpr = 4 if bk_scale >= 4 else bk_scale

    padded_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 16]], [half_m, BLOCK_K],
                                                                    [1, 0])
    padded_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 16]], [half_n, BLOCK_K],
                                                                    [1, 0])
    shared_scale: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [half_m_preshuffled, bk_scale_preshuffled], [1, 0])
    layouts: gl.constexpr = gl.amd.gfx1250.make_partitioned_dot_layouts(
        half_m, half_n, padded_a, padded_b, NUM_WARPS, [16, 16, 128], a_transposed=False, b_transposed=True)
    shared_a: gl.constexpr = layouts[0]
    shared_b: gl.constexpr = layouts[1]
    wmma: gl.constexpr = layouts[2]
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_a, [half_m, bk_scale])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_b, [half_n, bk_scale])

    nbuf: gl.constexpr = NUM_BUFFERS
    a_top_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, half_m, BLOCK_K], shared_a)
    a_bot_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, half_m, BLOCK_K], shared_a)
    b_left_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, half_n, BLOCK_K], shared_b)
    b_right_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, half_n, BLOCK_K], shared_b)
    as_top_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty,
                                           [nbuf, half_m_preshuffled, bk_scale_preshuffled], shared_scale)
    as_bot_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty,
                                           [nbuf, half_m_preshuffled, bk_scale_preshuffled], shared_scale)
    bs_left_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty,
                                            [nbuf, half_n_preshuffled, bk_scale_preshuffled], shared_scale)
    bs_right_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty,
                                             [nbuf, half_n_preshuffled, bk_scale_preshuffled], shared_scale)

    a_base = pid_m * BLOCK_M * stride_am
    b_base = pid_n * BLOCK_N * stride_bn
    a_top_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base, shape=(M, K), strides=(stride_am, stride_ak),
                                            block_shape=(half_m, BLOCK_K), layout=shared_a)
    a_bot_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base + half_m * stride_am, shape=(M, K),
                                            strides=(stride_am, stride_ak), block_shape=(half_m, BLOCK_K),
                                            layout=shared_a)
    b_left_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base, shape=(N, K), strides=(stride_bn, stride_bk),
                                             block_shape=(half_n, BLOCK_K), layout=shared_b)
    b_right_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base + half_n * stride_bn, shape=(N, K),
                                              strides=(stride_bn, stride_bk), block_shape=(half_n, BLOCK_K),
                                              layout=shared_b)

    as_top_base = (pid_m * BLOCK_M) // preshuffle_factor * stride_scale
    as_bot_base = (pid_m * BLOCK_M + half_m) // preshuffle_factor * stride_scale
    bs_left_base = (pid_n * BLOCK_N) // preshuffle_factor * stride_scale
    bs_right_base = (pid_n * BLOCK_N + half_n) // preshuffle_factor * stride_scale
    as_top_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + as_top_base, shape=(M // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(half_m_preshuffled, bk_scale_preshuffled), layout=shared_scale)
    as_bot_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + as_bot_base, shape=(M // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(half_m_preshuffled, bk_scale_preshuffled), layout=shared_scale)
    bs_left_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + bs_left_base, shape=(N // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(half_n_preshuffled, bk_scale_preshuffled), layout=shared_scale)
    bs_right_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + bs_right_base, shape=(N // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(half_n_preshuffled, bk_scale_preshuffled), layout=shared_scale)

    scale_copy_layout: gl.constexpr = get_scale_blocked_layout(NUM_WARPS)
    scale_m = gl.arange(0, half_m_preshuffled, layout=gl.SliceLayout(1, scale_copy_layout))
    scale_n = gl.arange(0, half_n_preshuffled, layout=gl.SliceLayout(1, scale_copy_layout))
    scale_k = gl.arange(0, bk_scale_preshuffled, layout=gl.SliceLayout(0, scale_copy_layout))
    as_top_ptrs = a_scale_ptr + as_top_base + scale_m[:, None] * stride_scale + scale_k[None, :]
    as_bot_ptrs = a_scale_ptr + as_bot_base + scale_m[:, None] * stride_scale + scale_k[None, :]
    bs_left_ptrs = b_scale_ptr + bs_left_base + scale_n[:, None] * stride_scale + scale_k[None, :]
    bs_right_ptrs = b_scale_ptr + bs_right_base + scale_n[:, None] * stride_scale + scale_k[None, :]

    for i in gl.static_range(NUM_BUFFERS - 1):
        tdm.async_load(b_left_desc, [0, i * BLOCK_K], b_left_buf.index(i), warp_used_hint=TDM_WARP_USED_HINT)
        _issue_mxfp_scale_load(bs_left_desc, bs_left_ptrs, i, bs_left_buf, i, bk_scale_preshuffled,
                               ASYNC_COPY_SCALE)
        tdm.async_load(a_top_desc, [0, i * BLOCK_K], a_top_buf.index(i), warp_used_hint=TDM_WARP_USED_HINT)
        _issue_mxfp_scale_load(as_top_desc, as_top_ptrs, i, as_top_buf, i, bk_scale_preshuffled, ASYNC_COPY_SCALE)
        tdm.async_load(a_bot_desc, [0, i * BLOCK_K], a_bot_buf.index(i), warp_used_hint=TDM_WARP_USED_HINT)
        _issue_mxfp_scale_load(as_bot_desc, as_bot_ptrs, i, as_bot_buf, i, bk_scale_preshuffled, ASYNC_COPY_SCALE)
        tdm.async_load(b_right_desc, [0, i * BLOCK_K], b_right_buf.index(i), warp_used_hint=TDM_WARP_USED_HINT)
        _issue_mxfp_scale_load(bs_right_desc, bs_right_ptrs, i, bs_right_buf, i, bk_scale_preshuffled,
                               ASYNC_COPY_SCALE)

    wait_unit: gl.constexpr = 8
    prefetch_wait: gl.constexpr = (NUM_BUFFERS - 2) * wait_unit - 2
    steady_wait: gl.constexpr = (NUM_BUFFERS - 2) * wait_unit - 3
    _wait_mxfp_scale_pipeline(prefetch_wait, ASYNC_COPY_SCALE)
    a_top = a_top_buf.index(0).load(layout=dot_a)
    as_top = _load_mxfp_scale(as_top_buf, 0, scale_a_layout, half_m, bk_scale, SCALE_PRESHUFFLE,
                              preshuffle_factor, scale_kwidth)
    b_left = b_left_buf.index(0).permute([1, 0]).load(layout=dot_b)
    bs_left = _load_mxfp_scale(bs_left_buf, 0, scale_b_layout, half_n, bk_scale, SCALE_PRESHUFFLE,
                               preshuffle_factor, scale_kwidth)

    acc_tl = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    acc_bl = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    acc_tr = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    acc_br = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max > 3)
    consume_k = 0
    load_k = NUM_BUFFERS - 1

    for _ in range(0, iter_max - (NUM_BUFFERS - 1)):
        read_slot = consume_k % nbuf
        next_slot = (consume_k + 1) % nbuf
        write_slot = load_k % nbuf
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_left, bs_left, DTYPE_B, acc_tl)
        _wait_fp8_scaled_warp_pipeline(steady_wait, ASYNC_COPY_SCALE)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = a_bot_buf.index(read_slot).load(layout=dot_a)
            as_bot = _load_mxfp_scale(as_bot_buf, read_slot, scale_a_layout, half_m, bk_scale, SCALE_PRESHUFFLE,
                                      preshuffle_factor, scale_kwidth)
            tdm.async_load(b_left_desc, [0, load_k * BLOCK_K], b_left_buf.index(write_slot),
                           warp_used_hint=TDM_WARP_USED_HINT)
            _issue_mxfp_scale_load(bs_left_desc, bs_left_ptrs, load_k, bs_left_buf, write_slot,
                                   bk_scale_preshuffled, ASYNC_COPY_SCALE)

        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_left, bs_left, DTYPE_B, acc_bl)
        _wait_fp8_scaled_warp_pipeline(steady_wait, ASYNC_COPY_SCALE)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = b_right_buf.index(read_slot).permute([1, 0]).load(layout=dot_b)
            bs_right = _load_mxfp_scale(bs_right_buf, read_slot, scale_b_layout, half_n, bk_scale,
                                        SCALE_PRESHUFFLE, preshuffle_factor, scale_kwidth)
            tdm.async_load(a_top_desc, [0, load_k * BLOCK_K], a_top_buf.index(write_slot),
                           warp_used_hint=TDM_WARP_USED_HINT)
            _issue_mxfp_scale_load(as_top_desc, as_top_ptrs, load_k, as_top_buf, write_slot,
                                   bk_scale_preshuffled, ASYNC_COPY_SCALE)

        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_right, bs_right, DTYPE_B, acc_tr)
        _wait_fp8_scaled_warp_pipeline(steady_wait, ASYNC_COPY_SCALE)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = b_left_buf.index(next_slot).permute([1, 0]).load(layout=dot_b)
            bs_left = _load_mxfp_scale(bs_left_buf, next_slot, scale_b_layout, half_n, bk_scale, SCALE_PRESHUFFLE,
                                       preshuffle_factor, scale_kwidth)
            tdm.async_load(a_bot_desc, [0, load_k * BLOCK_K], a_bot_buf.index(write_slot),
                           warp_used_hint=TDM_WARP_USED_HINT)
            _issue_mxfp_scale_load(as_bot_desc, as_bot_ptrs, load_k, as_bot_buf, write_slot,
                                   bk_scale_preshuffled, ASYNC_COPY_SCALE)

        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_right, bs_right, DTYPE_B, acc_br)
        _wait_fp8_scaled_warp_pipeline(steady_wait, ASYNC_COPY_SCALE)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = a_top_buf.index(next_slot).load(layout=dot_a)
            as_top = _load_mxfp_scale(as_top_buf, next_slot, scale_a_layout, half_m, bk_scale, SCALE_PRESHUFFLE,
                                      preshuffle_factor, scale_kwidth)
            tdm.async_load(b_right_desc, [0, load_k * BLOCK_K], b_right_buf.index(write_slot),
                           warp_used_hint=TDM_WARP_USED_HINT)
            _issue_mxfp_scale_load(bs_right_desc, bs_right_ptrs, load_k, bs_right_buf, write_slot,
                                   bk_scale_preshuffled, ASYNC_COPY_SCALE)
        consume_k += 1
        load_k += 1

    for i in gl.static_range(NUM_BUFFERS - 1):
        read_slot = (iter_max - (NUM_BUFFERS - 1 - i)) % nbuf
        acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_left, bs_left, DTYPE_B, acc_tl)
        if i < NUM_BUFFERS - 2:
            _wait_mxfp_scale_pipeline((NUM_BUFFERS - 2 - i) * wait_unit - 3, ASYNC_COPY_SCALE)
        else:
            _wait_mxfp_scale_pipeline(1, ASYNC_COPY_SCALE)
        a_bot = a_bot_buf.index(read_slot).load(layout=dot_a)
        as_bot = _load_mxfp_scale(as_bot_buf, read_slot, scale_a_layout, half_m, bk_scale, SCALE_PRESHUFFLE,
                                  preshuffle_factor, scale_kwidth)
        acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_left, bs_left, DTYPE_B, acc_bl)
        if i < NUM_BUFFERS - 2:
            _wait_mxfp_scale_pipeline((NUM_BUFFERS - 2 - i) * wait_unit - 4, ASYNC_COPY_SCALE)
        else:
            _wait_mxfp_scale_pipeline(0, ASYNC_COPY_SCALE)
        b_right = b_right_buf.index(read_slot).permute([1, 0]).load(layout=dot_b)
        bs_right = _load_mxfp_scale(bs_right_buf, read_slot, scale_b_layout, half_n, bk_scale, SCALE_PRESHUFFLE,
                                    preshuffle_factor, scale_kwidth)
        acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, DTYPE_A, b_right, bs_right, DTYPE_B, acc_tr)
        if i < NUM_BUFFERS - 2:
            next_slot = (iter_max - (NUM_BUFFERS - 1 - i) + 1) % nbuf
            _wait_mxfp_scale_pipeline((NUM_BUFFERS - 2 - i) * wait_unit - 5, ASYNC_COPY_SCALE)
            b_left = b_left_buf.index(next_slot).permute([1, 0]).load(layout=dot_b)
            bs_left = _load_mxfp_scale(bs_left_buf, next_slot, scale_b_layout, half_n, bk_scale, SCALE_PRESHUFFLE,
                                       preshuffle_factor, scale_kwidth)
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_right, bs_right, DTYPE_B, acc_br)
            _wait_mxfp_scale_pipeline((NUM_BUFFERS - 2 - i) * wait_unit - 6, ASYNC_COPY_SCALE)
            a_top = a_top_buf.index(next_slot).load(layout=dot_a)
            as_top = _load_mxfp_scale(as_top_buf, next_slot, scale_a_layout, half_m, bk_scale, SCALE_PRESHUFFLE,
                                      preshuffle_factor, scale_kwidth)
        else:
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, DTYPE_A, b_right, bs_right, DTYPE_B, acc_br)

    _store_quadrants(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc_tl, acc_bl, acc_tr, acc_br, wmma,
                     BLOCK_M, BLOCK_N)


@gluon.jit
def mxfp4_slice_mn_warp_pipeline_kernelC_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
        stride_cn, stride_scale, SCALE_BLOCK: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr, GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr,
        NUM_WARPS: gl.constexpr, NUM_BUFFERS: gl.constexpr, TDM_WARP_USED_HINT: gl.constexpr):
    gl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 128)
    gl.static_assert(SCALE_BLOCK == 16 or SCALE_BLOCK == 32)
    gl.static_assert(NUM_WARPS == 8)
    gl.static_assert(NUM_BUFFERS == 3 or NUM_BUFFERS == 4,
                     "MXFP4 kernelC requires three or four LDS buffers")
    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    half_m: gl.constexpr = BLOCK_M // 2
    half_n: gl.constexpr = BLOCK_N // 2
    bk_packed: gl.constexpr = BLOCK_K // 2
    bk_scale: gl.constexpr = BLOCK_K // SCALE_BLOCK
    preshuffle_factor: gl.constexpr = 128
    bk_scale_preshuffled: gl.constexpr = bk_scale * preshuffle_factor
    half_m_preshuffled: gl.constexpr = half_m // preshuffle_factor
    half_n_preshuffled: gl.constexpr = half_n // preshuffle_factor
    scale_kwidth: gl.constexpr = 4 if bk_scale >= 4 else bk_scale

    padded_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[bk_packed if bk_packed >= 256 else 256, 16]], [half_m, bk_packed], [1, 0])
    padded_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[bk_packed if bk_packed >= 256 else 256, 16]], [half_n, bk_packed], [1, 0])
    shared_scale: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [half_m_preshuffled, bk_scale_preshuffled], [1, 0])
    layouts: gl.constexpr = gl.amd.gfx1250.make_partitioned_dot_layouts(
        half_m, half_n, padded_a, padded_b, NUM_WARPS, [32, 16, 128], a_transposed=False, b_transposed=True)
    shared_a: gl.constexpr = layouts[0]
    shared_b: gl.constexpr = layouts[1]
    wmma: gl.constexpr = layouts[2]
    wmma_packed: gl.constexpr = gl.amd.AMDWMMALayout(3, True, wmma.warp_bases, wmma.reg_bases, [32, 16, 64])
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma_packed, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma_packed, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_a, [half_m, bk_scale])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_b, [half_n, bk_scale])

    nbuf: gl.constexpr = NUM_BUFFERS
    a_top_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, half_m, bk_packed], shared_a)
    a_bot_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [nbuf, half_m, bk_packed], shared_a)
    b_left_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, half_n, bk_packed], shared_b)
    b_right_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [nbuf, half_n, bk_packed], shared_b)
    as_top_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty,
                                           [nbuf, half_m_preshuffled, bk_scale_preshuffled], shared_scale)
    as_bot_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty,
                                           [nbuf, half_m_preshuffled, bk_scale_preshuffled], shared_scale)
    bs_left_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty,
                                            [nbuf, half_n_preshuffled, bk_scale_preshuffled], shared_scale)
    bs_right_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty,
                                             [nbuf, half_n_preshuffled, bk_scale_preshuffled], shared_scale)

    a_base = pid_m * BLOCK_M * stride_am
    b_base = pid_n * BLOCK_N * stride_bn
    a_top_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base, shape=(M, K // 2),
                                            strides=(stride_am, stride_ak), block_shape=(half_m, bk_packed),
                                            layout=shared_a)
    a_bot_desc = tdm.make_tensor_descriptor(base=a_ptr + a_base + half_m * stride_am, shape=(M, K // 2),
                                            strides=(stride_am, stride_ak), block_shape=(half_m, bk_packed),
                                            layout=shared_a)
    b_left_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base, shape=(N, K // 2),
                                             strides=(stride_bn, stride_bk), block_shape=(half_n, bk_packed),
                                             layout=shared_b)
    b_right_desc = tdm.make_tensor_descriptor(base=b_ptr + b_base + half_n * stride_bn, shape=(N, K // 2),
                                              strides=(stride_bn, stride_bk), block_shape=(half_n, bk_packed),
                                              layout=shared_b)

    as_top_base = (pid_m * BLOCK_M) // preshuffle_factor * stride_scale
    as_bot_base = (pid_m * BLOCK_M + half_m) // preshuffle_factor * stride_scale
    bs_left_base = (pid_n * BLOCK_N) // preshuffle_factor * stride_scale
    bs_right_base = (pid_n * BLOCK_N + half_n) // preshuffle_factor * stride_scale
    as_top_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + as_top_base, shape=(M // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(half_m_preshuffled, bk_scale_preshuffled), layout=shared_scale)
    as_bot_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + as_bot_base, shape=(M // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(half_m_preshuffled, bk_scale_preshuffled), layout=shared_scale)
    bs_left_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + bs_left_base, shape=(N // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(half_n_preshuffled, bk_scale_preshuffled), layout=shared_scale)
    bs_right_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + bs_right_base, shape=(N // preshuffle_factor, K // SCALE_BLOCK * preshuffle_factor),
        strides=(stride_scale, 1), block_shape=(half_n_preshuffled, bk_scale_preshuffled), layout=shared_scale)

    for i in gl.static_range(NUM_BUFFERS - 1):
        _issue_mxfp4_fused_scale_load(as_top_desc, as_bot_desc, bs_left_desc, bs_right_desc, i, as_top_buf, as_bot_buf,
                                      bs_left_buf, bs_right_buf, i, bk_scale_preshuffled)
        tdm.async_load(b_left_desc, [0, i * bk_packed], b_left_buf.index(i),
                       warp_used_hint=TDM_WARP_USED_HINT)
        tdm.async_load(a_top_desc, [0, i * bk_packed], a_top_buf.index(i), warp_used_hint=TDM_WARP_USED_HINT)
        tdm.async_load(a_bot_desc, [0, i * bk_packed], a_bot_buf.index(i), warp_used_hint=TDM_WARP_USED_HINT)
        tdm.async_load(b_right_desc, [0, i * bk_packed], b_right_buf.index(i),
                       warp_used_hint=TDM_WARP_USED_HINT)

    wait_unit: gl.constexpr = 5
    prefetch_wait: gl.constexpr = (NUM_BUFFERS - 2) * wait_unit - 2
    steady_wait: gl.constexpr = (NUM_BUFFERS - 2) * wait_unit - 3
    tdm.async_wait(prefetch_wait)
    a_top = a_top_buf.index(0).load(layout=dot_a)
    as_top = _load_mxfp_scale(as_top_buf, 0, scale_a_layout, half_m, bk_scale, True, preshuffle_factor,
                              scale_kwidth)
    b_left = b_left_buf.index(0).permute([1, 0]).load(layout=dot_b)
    bs_left = _load_mxfp_scale(bs_left_buf, 0, scale_b_layout, half_n, bk_scale, True, preshuffle_factor,
                               scale_kwidth)

    acc_tl = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    acc_bl = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    acc_tr = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    acc_br = gl.zeros((half_m, half_n), dtype=gl.float32, layout=wmma)
    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max > 3)
    consume_k = 0
    load_k = NUM_BUFFERS - 1

    for _ in range(0, iter_max - (NUM_BUFFERS - 1)):
        read_slot = consume_k % nbuf
        next_slot = (consume_k + 1) % nbuf
        write_slot = load_k % nbuf
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, "e2m1", b_left, bs_left, "e2m1", acc_tl)
        tdm.async_wait(steady_wait)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = a_bot_buf.index(read_slot).load(layout=dot_a)
            as_bot = _load_mxfp_scale(as_bot_buf, read_slot, scale_a_layout, half_m, bk_scale, True,
                                      preshuffle_factor, scale_kwidth)
            _issue_mxfp4_fused_scale_load(as_top_desc, as_bot_desc, bs_left_desc, bs_right_desc, load_k, as_top_buf,
                                          as_bot_buf, bs_left_buf, bs_right_buf, write_slot, bk_scale_preshuffled)
            tdm.async_load(b_left_desc, [0, load_k * bk_packed], b_left_buf.index(write_slot),
                           warp_used_hint=TDM_WARP_USED_HINT)

        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, "e2m1", b_left, bs_left, "e2m1", acc_bl)
        tdm.async_wait(steady_wait)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = b_right_buf.index(read_slot).permute([1, 0]).load(layout=dot_b)
            bs_right = _load_mxfp_scale(bs_right_buf, read_slot, scale_b_layout, half_n, bk_scale, True,
                                        preshuffle_factor, scale_kwidth)
            tdm.async_load(a_top_desc, [0, load_k * bk_packed], a_top_buf.index(write_slot),
                           warp_used_hint=TDM_WARP_USED_HINT)

        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, "e2m1", b_right, bs_right, "e2m1", acc_tr)
        tdm.async_wait(steady_wait)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = b_left_buf.index(next_slot).permute([1, 0]).load(layout=dot_b)
            bs_left = _load_mxfp_scale(bs_left_buf, next_slot, scale_b_layout, half_n, bk_scale, True,
                                       preshuffle_factor, scale_kwidth)
            tdm.async_load(a_bot_desc, [0, load_k * bk_packed], a_bot_buf.index(write_slot),
                           warp_used_hint=TDM_WARP_USED_HINT)

        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, "e2m1", b_right, bs_right, "e2m1", acc_br)
        tdm.async_wait(steady_wait)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = a_top_buf.index(next_slot).load(layout=dot_a)
            as_top = _load_mxfp_scale(as_top_buf, next_slot, scale_a_layout, half_m, bk_scale, True,
                                      preshuffle_factor, scale_kwidth)
            tdm.async_load(b_right_desc, [0, load_k * bk_packed], b_right_buf.index(write_slot),
                           warp_used_hint=TDM_WARP_USED_HINT)
        consume_k += 1
        load_k += 1

    for i in gl.static_range(NUM_BUFFERS - 1):
        read_slot = (iter_max - (NUM_BUFFERS - 1 - i)) % nbuf
        acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, as_top, "e2m1", b_left, bs_left, "e2m1", acc_tl)
        if i < NUM_BUFFERS - 2:
            tdm.async_wait((NUM_BUFFERS - 2 - i) * wait_unit - 3)
        else:
            tdm.async_wait(1)
        a_bot = a_bot_buf.index(read_slot).load(layout=dot_a)
        as_bot = _load_mxfp_scale(as_bot_buf, read_slot, scale_a_layout, half_m, bk_scale, True, preshuffle_factor,
                                  scale_kwidth)
        acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, "e2m1", b_left, bs_left, "e2m1", acc_bl)
        if i < NUM_BUFFERS - 2:
            tdm.async_wait((NUM_BUFFERS - 2 - i) * wait_unit - 4)
        else:
            tdm.async_wait(0)
        b_right = b_right_buf.index(read_slot).permute([1, 0]).load(layout=dot_b)
        bs_right = _load_mxfp_scale(bs_right_buf, read_slot, scale_b_layout, half_n, bk_scale, True,
                                    preshuffle_factor, scale_kwidth)
        acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, as_top, "e2m1", b_right, bs_right, "e2m1", acc_tr)
        if i < NUM_BUFFERS - 2:
            next_slot = (iter_max - (NUM_BUFFERS - 1 - i) + 1) % nbuf
            if (NUM_BUFFERS - 2 - i) * wait_unit - 5 > 0:
                tdm.async_wait((NUM_BUFFERS - 2 - i) * wait_unit - 5)
            else:
                tdm.async_wait(0)
            b_left = b_left_buf.index(next_slot).permute([1, 0]).load(layout=dot_b)
            bs_left = _load_mxfp_scale(bs_left_buf, next_slot, scale_b_layout, half_n, bk_scale, True,
                                       preshuffle_factor, scale_kwidth)
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, "e2m1", b_right, bs_right, "e2m1", acc_br)
            if (NUM_BUFFERS - 2 - i) * wait_unit - 6 > 0:
                tdm.async_wait((NUM_BUFFERS - 2 - i) * wait_unit - 6)
            a_top = a_top_buf.index(next_slot).load(layout=dot_a)
            as_top = _load_mxfp_scale(as_top_buf, next_slot, scale_a_layout, half_m, bk_scale, True,
                                      preshuffle_factor, scale_kwidth)
        else:
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, as_bot, "e2m1", b_right, bs_right, "e2m1", acc_br)

    _store_quadrants(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc_tl, acc_bl, acc_tr, acc_br, wmma,
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


def _run_direct_benchmark(launch, M, N, K, warmup, iters):
    if iters <= 0:
        raise ValueError("--direct-iters must be positive")

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        launch()
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0

    per_iter_s = elapsed_s / iters
    tflops = (2 * M * N * K) / per_iter_s / 1e12

    print("benchmark mode   : direct synchronized CPU wall")
    print(f"warmup iters     : {warmup}")
    print(f"total iters      : {iters}")
    print()
    print(f"total elapsed    : {elapsed_s:.6f} s")
    print(f"per-iter         : {per_iter_s * 1e6:.2f} us")
    print(f"TFLOPS           : {tflops:.3f}")


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
    expected_tile = (512, 512, 64) if args.f16_cluster_kernel_c else (256, 256, 64)
    if (args.BM, args.BN, args.BK) != expected_tile:
        kernel_name = "cluster KernelC" if args.f16_cluster_kernel_c else "sliceMN"
        raise ValueError(
            f"The fp16 {kernel_name} kernel expects -BM {expected_tile[0]} -BN {expected_tile[1]} "
            f"-BK {expected_tile[2]}")
    if args.num_warps != 8:
        raise ValueError("The fp16 inter-wave sliceMN port expects --num-warps 8")
    if args.f16_cluster_kernel_c and args.num_buffers not in (3, 4):
        raise ValueError("F16 cluster KernelC requires --num-buffers 3 or 4")
    if not args.f16_cluster_kernel_c and args.num_buffers < 2:
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
    cluster_layouts = None
    if args.f16_cluster_kernel_c:
        cga_layout_c = make_cga_layout([2, 2], [2, 2], [0, 1])
        cluster_layouts = _build_f16_cluster_layouts(args.BM, args.BN, args.BK, args.num_warps, cga_layout_c)

    def launch():
        if args.f16_cluster_kernel_c:
            shared_a, shared_b, wmma = cluster_layouts
            return f16_cluster_kernelC_gfx1250[grid](
                a, b, c, args.M, args.N, args.K, a.stride(0), a.stride(1), b.stride(1), b.stride(0), c.stride(0),
                c.stride(1), BLOCK_M=args.BM, BLOCK_N=args.BN, BLOCK_K=args.BK, GRID_MN=grid[0],
                NUM_XCDS=args.num_xcds, GROUP_SIZE_M=args.group_size_m, NUM_BUFFERS=args.num_buffers,
                NUM_WARPS=args.num_warps, SHARED_LAYOUT_A=shared_a, SHARED_LAYOUT_B=shared_b, WMMA_LAYOUT=wmma,
                num_warps=args.num_warps, num_ctas=4, waves_per_eu=args.num_warps // 4)
        return f16_slice_mn_warp_pipeline_kernel_gfx1250[grid](
            a, b, c, args.M, args.N, args.K, a.stride(0), a.stride(1), b.stride(1), b.stride(0), c.stride(0),
            c.stride(1), BLOCK_M=args.BM, BLOCK_N=args.BN, BLOCK_K=args.BK, GRID_MN=grid[0], NUM_XCDS=args.num_xcds,
            GROUP_SIZE_M=args.group_size_m, WARP_BASES=((0, 1), (1, 0), (2, 0)), num_warps=args.num_warps,
            NUM_BUFFERS=args.num_buffers, RESOLVE_PARTITION_CONFLICTS=args.resolve_partition_conflicts,
            NUM_WARPS=args.num_warps, MAX_ITER=triton.cdiv(args.K, args.BK),
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


def _make_bf16_kernelc_3pf_inputs(args):
    output_dtype = torch.bfloat16 if args.bf16_output else torch.float32
    if args.input_mode == "random":
        torch.manual_seed(42)
        a = torch.randn((args.M, args.K), dtype=torch.bfloat16)
        b = torch.randn((args.K, args.N), dtype=torch.bfloat16)
        if args.transpose_b:
            b = b.T.contiguous()
        c = torch.zeros((args.M, args.N), dtype=output_dtype)
        return a.cuda(), b.cuda(), c.cuda()

    device = torch.device("cuda")
    a = torch.empty((args.M, args.K), dtype=torch.bfloat16, device=device)
    b_shape = (args.N, args.K) if args.transpose_b else (args.K, args.N)
    b = torch.empty(b_shape, dtype=torch.bfloat16, device=device)
    chunk = 8 * 1024 * 1024
    for output, cosine in ((a, False), (b, True)):
        flat = output.view(-1)
        for begin in range(0, flat.numel(), chunk):
            end = min(begin + chunk, flat.numel())
            angle = torch.arange(begin, end, dtype=torch.float64, device=device)
            value = torch.cos(angle) if cosine else torch.sin(angle)
            flat[begin:end].copy_(value.float())
    c = torch.zeros((args.M, args.N), dtype=output_dtype, device=device)
    return a, b, c


def _make_bf16_case(args):
    if args.dtype_b != "bfloat16":
        raise ValueError("The BF16 path requires --dtype-a bfloat16 --dtype-b bfloat16")
    if args.scale_preshuffled or args.async_copy_scale:
        raise ValueError("--scale-preshuffled and --async-copy-scale do not apply to BF16")
    if not args.transpose_b:
        raise ValueError("The BF16 SliceMNK kernel expects --transpose-b so K is contiguous in B")
    is_cluster_4x4 = args.bf16_cluster_4x4_tdm or args.bf16_cluster_4x4_identity_tdm
    expected_mn = (1024, 1024) if is_cluster_4x4 else (256, 256)
    if (args.BM, args.BN) != expected_mn:
        raise ValueError(
            f"The selected BF16 kernel requires -BM {expected_mn[0]} -BN {expected_mn[1]}")
    if args.num_warps != 8:
        raise ValueError("The BF16 SliceMNK kernel requires --num-warps 8")
    is_kernelc = args.bf16_kernelc_3pf or args.bf16_kernelc_2pf or is_cluster_4x4
    if (args.bf16_kernelc_3pf or is_cluster_4x4) and (args.BK, args.num_buffers) != (64, 3):
        raise ValueError("The restored BF16 KernelC requires -BK 64 and --num-buffers 3")
    if args.bf16_kernelc_2pf and (args.BK, args.num_buffers) != (128, 2):
        raise ValueError("The double-buffered BF16 KernelC requires -BK 128 and --num-buffers 2")
    if args.bf16_cluster_4x4_tdm and (args.group_size_m != 4 or args.num_xcds != 8):
        raise ValueError("The isolated 4x4 result requires --group-size-m 4 and --num-xcds 8")
    if not is_kernelc and (args.BK, args.num_buffers) not in ((128, 2), (64, 3)):
        raise ValueError("The BF16 SliceMNK kernel supports BK128 with 2 buffers or BK64 with 3 buffers")
    if args.M % args.BM or args.N % args.BN or args.K % args.BK:
        raise ValueError("The BF16 SliceMNK kernel requires M, N, and K divisible by their block sizes")
    if triton.cdiv(args.K, args.BK) < args.num_buffers:
        raise ValueError("The BF16 SliceMNK kernel requires at least NUM_BUFFERS K tiles")

    if is_kernelc:
        a, b, c = _make_bf16_kernelc_3pf_inputs(args)
    else:
        torch.manual_seed(args.seed)
        a = torch.randn((args.M, args.K), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((args.K, args.N), dtype=torch.bfloat16)
        if args.transpose_b:
            b = b.T.contiguous()
        b = b.cuda()
        output_dtype = torch.bfloat16 if args.bf16_output else torch.float32
        c = torch.zeros((args.M, args.N), dtype=output_dtype, device="cuda")
    grid = (triton.cdiv(args.M, args.BM) * triton.cdiv(args.N, args.BN), 1)
    identity_cluster_grid = (triton.cdiv(args.M, args.BM), triton.cdiv(args.N, args.BN))
    cluster_layouts = None
    if is_cluster_4x4:
        cga_layout_c = make_cga_layout([4, 4], [4, 4], [0, 1])
        cluster_layouts = _build_f16_cluster_layouts(args.BM, args.BN, args.BK, args.num_warps, cga_layout_c)

    def launch():
        if args.bf16_cluster_4x4_identity_tdm:
            shared_a, shared_b, wmma = cluster_layouts
            return bf16_cluster_4x4_identity_tdm_kernel_gfx1250[identity_cluster_grid](
                a, b, c, args.M, args.N, args.K, a.stride(0), a.stride(1), b.stride(1), b.stride(0), c.stride(0),
                c.stride(1), BLOCK_M=args.BM, BLOCK_N=args.BN, BLOCK_K=args.BK, NUM_WARPS=args.num_warps,
                SHARED_LAYOUT_A=shared_a, SHARED_LAYOUT_B=shared_b, WMMA_LAYOUT=wmma, num_warps=args.num_warps,
                waves_per_eu=args.num_warps // 4, num_ctas=16)
        if args.bf16_cluster_4x4_tdm:
            shared_a, shared_b, wmma = cluster_layouts
            return bf16_cluster_4x4_tdm_kernel_gfx1250[grid](
                a, b, c, args.M, args.N, args.K, a.stride(0), a.stride(1), b.stride(1), b.stride(0), c.stride(0),
                c.stride(1), BLOCK_M=args.BM, BLOCK_N=args.BN, BLOCK_K=args.BK, GRID_MN=grid[0],
                NUM_XCDS=args.num_xcds, GROUP_SIZE_M=args.group_size_m, NUM_WARPS=args.num_warps,
                SHARED_LAYOUT_A=shared_a, SHARED_LAYOUT_B=shared_b, WMMA_LAYOUT=wmma, num_warps=args.num_warps,
                waves_per_eu=args.num_warps // 4, num_ctas=16)
        if args.bf16_kernelc_3pf:
            return bf16_kernelc_3pf_gfx1250[grid](
                a, b, c, args.M, args.N, args.K, a.stride(0), a.stride(1), b.stride(1), b.stride(0), c.stride(0),
                c.stride(1), BLOCK_M=args.BM, BLOCK_N=args.BN, BLOCK_K=args.BK, GRID_MN=grid[0],
                NUM_XCDS=args.num_xcds, GROUP_SIZE_M=args.group_size_m, NUM_WARPS=args.num_warps,
                num_warps=args.num_warps, waves_per_eu=args.num_warps // 4)
        if args.bf16_kernelc_2pf:
            return bf16_kernelc_2pf_gfx1250[grid](
                a, b, c, args.M, args.N, args.K, a.stride(0), a.stride(1), b.stride(1), b.stride(0), c.stride(0),
                c.stride(1), BLOCK_M=args.BM, BLOCK_N=args.BN, BLOCK_K=args.BK, GRID_MN=grid[0],
                NUM_XCDS=args.num_xcds, GROUP_SIZE_M=args.group_size_m, NUM_WARPS=args.num_warps,
                num_warps=args.num_warps, waves_per_eu=args.num_warps // 4)
        return bf16_slice_mnk_warp_pipeline_kernel_gfx1250[grid](
            a, b, c, args.M, args.N, args.K, a.stride(0), a.stride(1), b.stride(1), b.stride(0), c.stride(0),
            c.stride(1), BLOCK_M=args.BM, BLOCK_N=args.BN, BLOCK_K=args.BK, GROUP_SIZE_M=args.group_size_m,
            GRID_MN=grid[0], NUM_XCDS=args.num_xcds, NUM_WARPS=args.num_warps, num_warps=args.num_warps,
            NUM_BUFFERS=args.num_buffers, waves_per_eu=args.num_warps // 4)

    def check():
        c.zero_()
        launch()
        torch.cuda.synchronize()
        ref_b = b.cpu().T.to(torch.float32) if args.transpose_b else b.cpu().to(torch.float32)
        ref = a.cpu().to(torch.float32) @ ref_b
        output = c.cpu()
        if args.bf16_output:
            ref = ref.to(torch.bfloat16)
            torch.testing.assert_close(output, ref, rtol=1e-2, atol=2.5e-1)
        else:
            torch.testing.assert_close(output, ref, rtol=1e-3, atol=1e-2)
        print("result verified", flush=True)

    return launch, check


def _make_fp8_case(args):
    if not args.dtype_a.startswith("float8") or not args.dtype_b.startswith("float8"):
        raise ValueError("The plain FP8 path requires FP8 inputs for both operands")
    if args.scale_preshuffled or args.async_copy_scale or args.with_a_scale:
        raise ValueError("Scale options require --mxfp")
    if not args.transpose_b:
        raise ValueError("The tutorial FP8 kernel expects --transpose-b so K is contiguous in B")
    if (args.BM, args.BN, args.BK) != (256, 256, 128):
        raise ValueError("The tutorial FP8 path requires -BM 256 -BN 256 -BK 128")
    if args.num_warps != 8:
        raise ValueError("The dedicated plain-FP8 path requires --num-warps 8")
    if args.kernel_c and args.num_buffers not in (3, 4):
        raise ValueError("FP8 kernelC requires --num-buffers 3 or 4")
    if not args.kernel_c and args.num_buffers != 2:
        raise ValueError("The tutorial FP8 path uses fixed double buffering; pass --num-buffers 2")
    if args.M % args.BM or args.N % args.BN or args.K % args.BK:
        raise ValueError("The tutorial FP8 path requires M, N, and K to be divisible by their block sizes")
    if triton.cdiv(args.K, args.BK) <= 3:
        raise ValueError("K/BLOCK_K must be greater than 3 for the 2x-unrolled tutorial pipeline")

    torch.manual_seed(args.seed)
    a = init_data(args.dtype_a, args.M, args.K)
    b = init_data(args.dtype_b, args.K, args.N)
    c_ref = None
    if args.check:
        c_ref = (a.to(torch.float32) @ b.to(torch.float32)).to(torch.float16)

    a_d = a.contiguous().cuda()
    b_d = b.T.contiguous().cuda()
    c_d = torch.empty((args.M, args.N), dtype=torch.float16, device="cuda")
    grid = (triton.cdiv(args.M, args.BM) * triton.cdiv(args.N, args.BN), 1)

    def launch():
        kernel_args = (
            a_d, b_d, c_d, args.M, args.N, args.K, a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1), MXFP_DTYPE_TO_KERNEL[args.dtype_a], MXFP_DTYPE_TO_KERNEL[args.dtype_b],
            args.BM, args.BN, args.BK, args.group_size_m)
        launch_options = {
            "GRID_MN": grid[0],
            "NUM_XCDS": args.num_xcds,
            "NUM_WARPS": args.num_warps,
            "num_warps": args.num_warps,
            "llvm_fn_attrs": (("amdgpu-agpr-alloc", "0,0"), ),
            "waves_per_eu": args.num_warps // 4,
        }
        if args.kernel_c:
            return fp8_slice_mn_warp_pipeline_kernelC_gfx1250[grid](
                *kernel_args, NUM_BUFFERS=args.num_buffers, TDM_WARP_USED_HINT=0b00001111, **launch_options)
        return fp8_slice_mn_warp_pipeline_kernel_gfx1250[grid](*kernel_args, **launch_options)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(c_d.cpu(), c_ref, rtol=1e-3, atol=1e-3)
        print("result verified", flush=True)

    return launch, check


def _pack_fp8_scale(scale, scale_kwidth):
    non_k, k_scale = scale.shape
    preshuffle_factor = 128
    num_chunk_m = non_k // preshuffle_factor
    num_chunk_k = k_scale // scale_kwidth
    scale = scale.view(num_chunk_m, 4, preshuffle_factor // 4, num_chunk_k, scale_kwidth)
    scale = scale.permute(0, 3, 2, 1, 4).contiguous()
    return scale.view(non_k // preshuffle_factor, k_scale * preshuffle_factor)


def _pack_fp8_scale_hipblaslt(scale, scale_kwidth):
    non_k, k_scale = scale.shape
    # gfx1250 hipBLASLt layout: [K-scale tile, non-K coordinate, four scales].
    scale = scale.view(non_k, k_scale // scale_kwidth, scale_kwidth)
    return scale.permute(1, 0, 2).contiguous().view(k_scale // scale_kwidth, non_k * scale_kwidth)


def _make_fp8_scaled_case(args):
    if not args.dtype_a.startswith("float8") or not args.dtype_b.startswith("float8"):
        raise ValueError("--with-scale requires FP8 inputs for both operands")
    if args.with_a_scale:
        raise ValueError("--with-scale always scales both operands; do not pass --with-a-scale")
    if not args.transpose_b:
        raise ValueError("The scaled FP8 kernels expect --transpose-b so K is contiguous in B")
    cluster_selectors = (args.fp8_cluster, args.fp8_cluster_bf16_style, args.fp8_scaled_cluster_bk128)
    if sum(cluster_selectors) > 1:
        raise ValueError("Scaled FP8 cluster selectors are mutually exclusive")
    if any(cluster_selectors) and args.kernel_c:
        raise ValueError("The scaled FP8 cluster selectors and --kernelC are mutually exclusive")
    expected_bk = 128 if args.kernel_c else 256
    if args.fp8_scaled_cluster_bk128:
        expected_tile = (512, 256, 128)
    elif args.fp8_cluster_bf16_style:
        cluster_m, cluster_n = args.fp8_cluster_shape
        expected_tile = (cluster_m * 256, cluster_n * 256, 256)
    elif args.fp8_cluster:
        expected_tile = (1024, 1024, 256)
    else:
        expected_tile = (256, 256, expected_bk)
    if (args.BM, args.BN, args.BK) != expected_tile:
        if args.fp8_scaled_cluster_bk128:
            kernel_name = "BK128 cluster"
        else:
            kernel_name = ("BF16-style cluster" if args.fp8_cluster_bf16_style else
                           ("cluster tutorial" if args.fp8_cluster else ("kernelC" if args.kernel_c else "tutorial")))
        raise ValueError(
            f"The scaled FP8 {kernel_name} kernel requires -BM {expected_tile[0]} -BN {expected_tile[1]} "
            f"-BK {expected_tile[2]}")
    if args.num_warps != 8:
        raise ValueError("The scaled FP8 kernels require --num-warps 8")
    if (args.kernel_c or args.fp8_scaled_cluster_bk128) and args.num_buffers not in (3, 4):
        raise ValueError("Scaled FP8 kernelC and BK128 cluster kernels require --num-buffers 3 or 4")
    if (args.kernel_c or args.fp8_scaled_cluster_bk128) and args.l2_prefetch_distance:
        raise ValueError("--l2-prefetch-distance applies only to the BK256 scaled FP8 tutorial kernels")
    if not args.kernel_c and not args.fp8_scaled_cluster_bk128 and args.num_buffers != 2:
        raise ValueError("The scaled tutorial FP8 kernel requires --num-buffers 2")
    if args.M % args.BM or args.N % args.BN or args.K % args.BK:
        raise ValueError("The scaled FP8 kernels require M, N, and K to be divisible by their block sizes")
    if args.kernel_c and triton.cdiv(args.K, args.BK) <= 3:
        raise ValueError("Scaled FP8 kernelC requires K/BLOCK_K to be greater than 3")
    if args.fp8_scaled_cluster_bk128 and triton.cdiv(args.K, args.BK) < args.num_buffers:
        raise ValueError("Scaled FP8 BK128 cluster kernel requires at least NUM_BUFFERS K tiles")
    if not args.kernel_c and not args.fp8_scaled_cluster_bk128 and triton.cdiv(args.K, args.BK) < 2:
        raise ValueError("The scaled tutorial FP8 kernel requires at least two K tiles")
    if args.scale_block not in (16, 32):
        raise ValueError("The scaled WMMA kernels support --scale-block 16 or 32")
    if args.BK % args.scale_block:
        raise ValueError("BLOCK_K must be divisible by --scale-block")
    if not args.kernel_c and not args.scale_preshuffled:
        raise ValueError("The BK256 scaled tutorial FP8 kernel requires --scale-preshuffled")
    if not args.kernel_c and args.async_copy_scale:
        raise ValueError("The BK256 scaled tutorial FP8 kernel stages scales with TDM; do not pass --async-copy-scale")

    scale_k = triton.cdiv(args.K, args.scale_block)
    block_scale_k = args.BK // args.scale_block
    scale_kwidth = 4 if block_scale_k >= 4 else block_scale_k
    if args.scale_preshuffled:
        if args.fp8_scale_layout == "triton" and (args.M % 128 or args.N % 128):
            raise ValueError("--scale-preshuffled requires M and N to be divisible by 128")
        if scale_k % scale_kwidth:
            raise ValueError("--scale-preshuffled requires K/scale_block to be divisible by its scale K-width")

    torch.manual_seed(args.seed)
    if args.input_mode == "trig":
        dtype_a = torch.float8_e4m3fn if args.dtype_a == "float8_e4m3" else torch.float8_e5m2
        dtype_b = torch.float8_e4m3fn if args.dtype_b == "float8_e4m3" else torch.float8_e5m2
        a = torch.empty((args.M, args.K), dtype=dtype_a)
        b = torch.empty((args.K, args.N), dtype=dtype_b)
        chunk = 8 * 1024 * 1024
        for output, cosine in ((a, False), (b, True)):
            flat = output.view(-1)
            for begin in range(0, flat.numel(), chunk):
                end = min(begin + chunk, flat.numel())
                angle = torch.arange(begin, end, dtype=torch.float64)
                value = torch.cos(angle) if cosine else torch.sin(angle)
                flat[begin:end].copy_(value.float())
    else:
        a = init_data(args.dtype_a, args.M, args.K)
        b = init_data(args.dtype_b, args.K, args.N)
    a_scale_obj = MXScaleTensor(size=(args.M, scale_k)).random(low=1.0, high=32.0)
    b_scale_obj = MXScaleTensor(size=(args.N, scale_k)).random(low=1.0, high=32.0)
    a_scale = a_scale_obj.data
    b_scale = b_scale_obj.data

    c_ref = None
    if args.check:
        output_dtype = torch.bfloat16 if args.fp8_bf16_output else torch.float16
        c_ref = torch_gemm_mxfp(a, b, a_scale_obj, b_scale_obj, args.scale_block, args.M, args.N,
                                args.K).to(output_dtype)

    if args.scale_preshuffled:
        if args.fp8_scale_layout == "hipblaslt":
            a_scale = _pack_fp8_scale_hipblaslt(a_scale, scale_kwidth)
            b_scale = _pack_fp8_scale_hipblaslt(b_scale, scale_kwidth)
        else:
            a_scale = _pack_fp8_scale(a_scale, scale_kwidth)
            b_scale = _pack_fp8_scale(b_scale, scale_kwidth)

    cluster_layouts = None
    if any(cluster_selectors):
        if args.fp8_scaled_cluster_bk128:
            cta_m, cta_n = 2, 1
        elif args.fp8_cluster_bf16_style:
            cta_m, cta_n = args.fp8_cluster_shape
        else:
            cta_m, cta_n = 4, 4
        cga_layout_c = make_cga_layout([cta_m, cta_n], [cta_m, cta_n], [0, 1])
        cluster_layouts = _build_fp8_scaled_cluster_layouts(args.BM, args.BN, args.BK, args.scale_block,
                                                            args.num_warps, cga_layout_c,
                                                            CTA_M=cta_m, CTA_N=cta_n,
                                                            HIPBLASLT_SCALE_LAYOUT=(
                                                                args.fp8_scale_layout == "hipblaslt"),
                                                            USE_PARTITIONED=args.fp8_cluster_bf16_style)

    a_d = a.contiguous().cuda()
    b_d = b.T.contiguous().cuda()
    a_scale_d = a_scale.cuda()
    b_scale_d = b_scale.cuda()
    output_dtype = torch.bfloat16 if args.fp8_bf16_output else torch.float16
    c_d = torch.empty((args.M, args.N), dtype=output_dtype, device="cuda")
    grid = (triton.cdiv(args.M, args.BM) * triton.cdiv(args.N, args.BN), 1)
    swizzle_block_m, swizzle_block_n = args.swizzle_block or (0, 0)
    if args.z_order and args.swizzle_block:
        raise ValueError("--z-order and --swizzle-block are mutually exclusive")
    if args.z_order:
        tiles_m = triton.cdiv(args.M, args.BM)
        tiles_n = triton.cdiv(args.N, args.BN)
        if args.kernel_c:
            raise ValueError("--z-order is currently supported only by the scaled FP8 tutorial kernel")
        if tiles_m != tiles_n or tiles_m & (tiles_m - 1):
            raise ValueError("--z-order requires a square power-of-two CTA grid")
    if args.swizzle_block:
        tiles_m = triton.cdiv(args.M, args.BM)
        tiles_n = triton.cdiv(args.N, args.BN)
        if args.kernel_c:
            raise ValueError("--swizzle-block is currently supported only by the scaled FP8 tutorial kernel")
        if tiles_m % swizzle_block_m or tiles_n % swizzle_block_n:
            raise ValueError("--swizzle-block dimensions must divide the CTA grid")

    def launch():
        kernel_args = (
            a_d, b_d, c_d, a_scale_d, b_scale_d, args.M, args.N, args.K, a_d.stride(0), a_d.stride(1),
            b_d.stride(1), b_d.stride(0), c_d.stride(0), c_d.stride(1), b_scale_d.stride(0),
            MXFP_DTYPE_TO_KERNEL[args.dtype_a], MXFP_DTYPE_TO_KERNEL[args.dtype_b], args.scale_block, args.BM,
            args.BN, args.BK, args.group_size_m)
        launch_options = {
            "GRID_MN": grid[0],
            "NUM_XCDS": args.num_xcds,
            "NUM_WARPS": args.num_warps,
            "SCALE_PRESHUFFLE": args.scale_preshuffled,
            "ASYNC_COPY_SCALE": args.async_copy_scale,
            "num_warps": args.num_warps,
            "llvm_fn_attrs": (("amdgpu-agpr-alloc", "0,0"), ),
            "waves_per_eu": args.num_warps // 4,
        }
        if args.kernel_c:
            return fp8_scaled_slice_mn_warp_pipeline_kernelC_gfx1250[grid](
                *kernel_args, NUM_BUFFERS=args.num_buffers, TDM_WARP_USED_HINT=0b00001111, **launch_options)
        if args.fp8_scaled_cluster_bk128:
            shared_a, shared_b, shared_scale_a, shared_scale_b, wmma = cluster_layouts
            return fp8_scaled_cluster_bk128_kernel_gfx1250[
                grid](*kernel_args, NUM_BUFFERS=args.num_buffers, SHARED_LAYOUT_A=shared_a,
                      SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_scale_a, SHARED_SCALE_B=shared_scale_b,
                      WMMA_LAYOUT=wmma, Z_ORDER=args.z_order, SWIZZLE_BLOCK_M=swizzle_block_m,
                      SWIZZLE_BLOCK_N=swizzle_block_n, num_ctas=2, **launch_options)
        if args.fp8_cluster_bf16_style:
            cta_m, cta_n = args.fp8_cluster_shape
            shared_a, shared_b, shared_scale_a, shared_scale_b, wmma = cluster_layouts
            return fp8_scaled_cluster_bf16_style_kernel_gfx1250[
                grid](*kernel_args, SHARED_LAYOUT_A=shared_a, SHARED_LAYOUT_B=shared_b,
                      SHARED_SCALE_A=shared_scale_a, SHARED_SCALE_B=shared_scale_b, WMMA_LAYOUT=wmma,
                      HIPBLASLT_SCALE_LAYOUT=args.fp8_scale_layout == "hipblaslt",
                      AB_SEPARATE_DATA=args.fp8_tdm_schedule == "ab-scales",
                      LEADING_FOUR_TDM=args.fp8_tdm_schedule == "leading4",
                      CTA_M=cta_m, CTA_N=cta_n, num_ctas=cta_m * cta_n,
                      **launch_options)
        if args.fp8_cluster:
            shared_a, shared_b, shared_scale_a, shared_scale_b, wmma = cluster_layouts
            return fp8_scaled_slice_mn_cluster_warp_pipeline_kernel_gfx1250[
                grid](*kernel_args, L2_PREFETCH_DISTANCE=args.l2_prefetch_distance, SHARED_LAYOUT_A=shared_a,
                      SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_scale_a, SHARED_SCALE_B=shared_scale_b,
                      WMMA_LAYOUT=wmma, Z_ORDER=args.z_order, SWIZZLE_BLOCK_M=swizzle_block_m,
                      SWIZZLE_BLOCK_N=swizzle_block_n, num_ctas=16, **launch_options)
        return fp8_scaled_slice_mn_warp_pipeline_kernel_gfx1250[
            grid](*kernel_args, L2_PREFETCH_DISTANCE=args.l2_prefetch_distance, Z_ORDER=args.z_order,
                  SWIZZLE_BLOCK_M=swizzle_block_m, SWIZZLE_BLOCK_N=swizzle_block_n, **launch_options)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        if args.fp8_bf16_output:
            torch.testing.assert_close(c_d.cpu(), c_ref, rtol=1e-2, atol=5e-1)
        else:
            torch.testing.assert_close(c_d.cpu(), c_ref, rtol=1e-3, atol=1e-3)
        print("result verified", flush=True)

    return launch, check


def _make_mxfp4_pipeline_case(args):
    is_kernel_c = args.mxfp4_kernel_c
    is_bk512 = args.mxfp4_bk512
    cluster_m, cluster_n = args.mxfp4_cluster_shape
    if not args.transpose_b:
        raise ValueError("The dedicated MXFP4 kernels require --transpose-b so packed K is contiguous in B")
    if is_kernel_c and (args.BM, args.BN, args.BK) != (256, 256, 128):
        raise ValueError("The dedicated MXFP4 KernelC kernel requires -BM 256 -BN 256 -BK 128")
    expected_bk512_tile = (cluster_m * 256, cluster_n * 256, 512)
    if is_bk512 and (args.BM, args.BN, args.BK) != expected_bk512_tile:
        raise ValueError(
            f"The {cluster_m}x{cluster_n} MXFP4 BK512 kernel requires "
            f"-BM {expected_bk512_tile[0]} -BN {expected_bk512_tile[1]} -BK 512")
    if not is_kernel_c and not is_bk512 and (args.BM, args.BN, args.BK) != (256, 256, 256):
        raise ValueError("The MXFP4 tutorial experiment requires -BM 256 -BN 256 -BK 256")
    if args.num_warps != 8:
        raise ValueError("The dedicated MXFP4 kernels require --num-warps 8")
    if is_kernel_c and args.num_buffers not in (3, 4):
        raise ValueError("MXFP4 kernelC requires --num-buffers 3 or 4")
    if not is_kernel_c and args.num_buffers != 2:
        raise ValueError("The two-buffer MXFP4 kernels require --num-buffers 2")
    if is_bk512 and args.scale_block != 32:
        raise ValueError("The dedicated MXFP4 BK512 kernel requires --scale-block 32")
    if args.M % args.BM or args.N % args.BN or args.K % args.BK:
        raise ValueError("The dedicated MXFP4 kernels require M, N, and K to be divisible by their block sizes")
    num_k_tiles = triton.cdiv(args.K, args.BK)
    if is_kernel_c and num_k_tiles <= 3:
        raise ValueError("MXFP4 KernelC requires K/BLOCK_K to be greater than 3")
    if not is_kernel_c and num_k_tiles < 2:
        raise ValueError("The two-buffer MXFP4 kernels require K/BLOCK_K to be at least 2")
    if args.scale_block not in (16, 32):
        raise ValueError("The MXFP4 kernels support --scale-block 16 or 32")

    scale_k = args.K // args.scale_block
    block_scale_k = args.BK // args.scale_block
    scale_kwidth = 4 if block_scale_k >= 4 else block_scale_k
    if args.M % 128 or args.N % 128 or scale_k % scale_kwidth:
        raise ValueError("MXFP4 preshuffled scales require M/N divisible by 128 and a compatible scale K-width")

    torch.manual_seed(args.seed)
    if args.input_mode == "trig":
        def make_trig_input(shape, cosine):
            result = MXFP4Tensor(size=shape)
            encoded = torch.empty(shape[0] * shape[1], dtype=torch.uint8)
            chunk = 8 * 1024 * 1024
            for begin in range(0, encoded.numel(), chunk):
                end = min(begin + chunk, encoded.numel())
                angle = torch.arange(begin, end, dtype=torch.float64)
                values = torch.cos(angle) if cosine else torch.sin(angle)
                encoded[begin:end].copy_(MXFP4Tensor(data=values.float()).data)
            result.data = encoded.view(shape)
            return result

        a = make_trig_input((args.M, args.K), False)
        b = make_trig_input((args.K, args.N), True)
    else:
        a = init_data("float4", args.M, args.K)
        b = init_data("float4", args.K, args.N)
    a_scale_obj = MXScaleTensor(size=(args.M, scale_k)).random(low=1.0, high=32.0)
    b_scale_obj = MXScaleTensor(size=(args.N, scale_k)).random(low=1.0, high=32.0)

    c_ref = None
    if args.check:
        output_dtype = torch.bfloat16 if is_bk512 else torch.float16
        c_ref = torch_gemm_mxfp(
            a, b, a_scale_obj, b_scale_obj, args.scale_block,
            args.M, args.N, args.K).to(output_dtype)

    a_scale = _pack_fp8_scale(a_scale_obj.data, scale_kwidth)
    b_scale = _pack_fp8_scale(b_scale_obj.data, scale_kwidth)
    a = a.to_packed_tensor(dim=1)
    b = b.to_packed_tensor(dim=0)
    a_d = a.data.contiguous().cuda()
    b_d = b.data.T.contiguous().cuda()
    a_scale_d = a_scale.cuda()
    b_scale_d = b_scale.cuda()
    output_dtype = torch.bfloat16 if is_bk512 else torch.float16
    c_d = torch.empty((args.M, args.N), dtype=output_dtype, device="cuda")
    grid = (triton.cdiv(args.M, args.BM) * triton.cdiv(args.N, args.BN), 1)
    bk512_layouts = None
    if is_bk512:
        cga_layout = make_cga_layout(
            [cluster_m, cluster_n], [cluster_m, cluster_n], [0, 1])
        bk512_layouts = _build_mxfp4_bk512_cluster_layouts(
            args.BM, args.BN, args.BK, args.num_warps, cga_layout,
            cluster_m, cluster_n)

    def launch():
        kernel_args = (
            a_d, b_d, c_d, a_scale_d, b_scale_d, args.M, args.N, args.K, a_d.stride(0), a_d.stride(1),
            b_d.stride(1), b_d.stride(0), c_d.stride(0), c_d.stride(1), b_scale_d.stride(0), args.scale_block,
            args.BM, args.BN, args.BK, args.group_size_m)
        launch_options = {
            "GRID_MN": grid[0],
            "NUM_XCDS": args.num_xcds,
            "NUM_WARPS": args.num_warps,
            "num_warps": args.num_warps,
            "llvm_fn_attrs": (("amdgpu-agpr-alloc", "0,0"), ),
            "waves_per_eu": args.num_warps // 4,
        }
        if is_kernel_c:
            return mxfp4_slice_mn_warp_pipeline_kernelC_gfx1250[grid](
                *kernel_args, NUM_BUFFERS=args.num_buffers, TDM_WARP_USED_HINT=0b00001111, **launch_options)
        if is_bk512:
            shared_a, shared_b, shared_scale_a, shared_scale_b, wmma = bk512_layouts
            return mxfp4_bk512_warp_pipeline_gfx1250[grid](
                *kernel_args, SHARED_LAYOUT_A=shared_a, SHARED_LAYOUT_B=shared_b,
                SHARED_SCALE_A=shared_scale_a, SHARED_SCALE_B=shared_scale_b,
                WMMA_LAYOUT=wmma, CTA_M=cluster_m, CTA_N=cluster_n,
                K256_STAGES=args.mxfp4_k256_stages,
                OVERLAP_CLUSTER=args.mxfp4_overlap_cluster,
                num_ctas=cluster_m * cluster_n, **launch_options)
        return mxfp4_slice_mn_warp_pipeline_tutorial_gfx1250[grid](*kernel_args, **launch_options)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        if is_bk512:
            torch.testing.assert_close(c_d.cpu(), c_ref.cpu(), rtol=1e-2, atol=5e-1)
        else:
            torch.testing.assert_close(c_d.cpu(), c_ref.cpu(), rtol=1e-3, atol=1e-3)
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
    use_warp_pipeline = args.dtype_a.startswith("float8") and args.dtype_b.startswith("float8")

    def launch():
        return mxfp_slice_mn_warp_pipeline_kernel_gfx1250[grid](
            a_d, b_d, c_d, a_scale_d, b_scale_d, args.M, args.N, args.K, stride_am, stride_ak, stride_bk, stride_bn,
            stride_cm, stride_cn, stride_scale, MXFP_DTYPE_TO_KERNEL[args.dtype_a], MXFP_DTYPE_TO_KERNEL[args.dtype_b],
            args.scale_block, args.BM, args.BN, args.BK, args.group_size_m, GRID_MN=grid[0], NUM_XCDS=args.num_xcds,
            TRANSPOSE_B=args.transpose_b, WITH_A_SCALE=args.with_a_scale, NUM_WARPS=args.num_warps,
            NUM_BUFFERS=args.num_buffers, RESOLVE_PARTITION_CONFLICTS=args.resolve_partition_conflicts,
            SCALE_PRESHUFFLE=args.scale_preshuffled, ASYNC_COPY_SCALE=args.async_copy_scale,
            USE_WARP_PIPELINE=use_warp_pipeline,
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
    parser.add_argument("-BK", type=int, default=None,
                        help="BLOCK_K (defaults to 256 for MXFP4/scaled FP8 tutorials, 128 for plain FP8/KernelC, "
                        "otherwise 64)")
    parser.add_argument("--dtype-a", default="float16",
                        choices=["float16", "bfloat16", "float8_e4m3", "float8_e5m2", "float4"])
    parser.add_argument("--dtype-b", default=None,
                        choices=["float16", "bfloat16", "float8_e4m3", "float8_e5m2", "float4"])
    parser.add_argument("--num-warps", type=int, default=8, choices=[4, 8])
    parser.add_argument("--num-buffers", type=int, default=None, choices=[2, 3, 4],
                        help="LDS buffer count (defaults to 3 for MXFP4 KernelC, otherwise 2)")
    parser.add_argument("--group-size-m", type=int, default=4, choices=[1, 2, 4, 8])
    parser.add_argument("--num-xcds", type=int, default=8)
    parser.add_argument("--z-order", action="store_true",
                        help="Use Morton/Z-order CTA traversal for square power-of-two scaled-FP8 grids")
    parser.add_argument("--swizzle-block", type=int, nargs=2, metavar=("M", "N"),
                        help="Traverse the scaled-FP8 CTA grid in MxN blocks")
    parser.add_argument("--transpose-b", action="store_true", default=True)
    parser.add_argument("--no-transpose-b", action="store_false", dest="transpose_b")
    parser.add_argument("--scale-block", type=int, default=32, help="MXFP scale block")
    parser.add_argument("--mxfp", action="store_true", help="Use block-scaled MXFP for FP8 inputs")
    parser.add_argument("--with-scale", action="store_true",
                        help="Use separate plain-FP8 kernels with both A and B E8M0 scales")
    parser.add_argument("--with-a-scale", action="store_true", help="Use A scales on the MXFP path")
    parser.add_argument("--scale-preshuffled", "--scale_preshuffled", dest="scale_preshuffled", action="store_true",
                        help="Use preshuffled E8M0 scale tensors")
    parser.add_argument("--fp8-scale-layout", choices=["triton", "hipblaslt"], default="triton",
                        help="Global scale permutation for --fp8-cluster-bf16-style")
    parser.add_argument("--async-copy-scale", "--async_copy_scale", dest="async_copy_scale", action="store_true",
                        help="Stage E8M0 scale tensors with async copy instead of TDM")
    parser.add_argument("--l2-prefetch-distance", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="Scaled FP8 tutorial A/B-prefetch distance in BK tiles (0 disables)")
    parser.add_argument("--f16-cluster-kernelC", dest="f16_cluster_kernel_c", action="store_true",
                        help="Use the four-CTA, 512x512 F16 KernelC kernel")
    parser.add_argument("--bf16-kernelc-3pf", dest="bf16_kernelc_3pf", action="store_true",
                        help="Use the restored 256x256x64 BF16 KernelC with its TDM-store epilogue")
    parser.add_argument("--bf16-kernelc-2pf", dest="bf16_kernelc_2pf", action="store_true",
                        help="Use the double-buffered 256x256x128 BF16 KernelC")
    parser.add_argument("--bf16-cluster-4x4-tdm", dest="bf16_cluster_4x4_tdm", action="store_true",
                        help="Use the standalone 16-CTA BF16 KernelC with its CGA-aware TDM-store epilogue")
    parser.add_argument("--bf16-cluster-4x4-identity-tdm", dest="bf16_cluster_4x4_identity_tdm",
                        action="store_true",
                        help="Use a separate 16-CTA BF16 KernelC with direct 2D cluster coordinates")
    parser.add_argument("--bf16-output", action="store_true",
                        help="Store BF16 instead of FP32 from BF16 kernels")
    parser.add_argument("--input-mode", choices=["random", "trig"], default="random",
                        help="Input initialization for BF16, scaled-FP8, and dedicated MXFP4 kernels")
    parser.add_argument("--fp8-cluster", action="store_true",
                        help="Use the 16-CTA, 1024x1024 scaled-FP8 cluster tutorial kernel")
    parser.add_argument("--fp8-cluster-bf16-style", action="store_true",
                        help="Use the parameterized square-cluster BK256 scaled-FP8 kernel with partitioned A/B LDS")
    parser.add_argument("--fp8-cluster-shape", type=int, nargs=2, metavar=("CTA_M", "CTA_N"),
                        default=(4, 4),
                        help="Square cluster shape for --fp8-cluster-bf16-style: 1 1, 2 2, or 4 4")
    parser.add_argument("--fp8-tdm-schedule", choices=["leading4", "fused-pairs", "ab-scales"], default=None,
                        help="TDM schedule for --fp8-cluster-bf16-style (default: two fused loads on warps 0..3)")
    parser.add_argument("--fp8-scaled-cluster-bk128", action="store_true",
                        help="Use the 2-CTA, 512x256 scaled-FP8 BK128 kernel")
    parser.add_argument("--fp8-bf16-output", action="store_true",
                        help="Store BF16 instead of FP16 from scaled-FP8 kernels")
    parser.add_argument("--kernelC", dest="kernel_c", action="store_true",
                        help="Use the plain-FP8 KernelC variant with four-warp TDM load issue")
    mxfp4_group = parser.add_mutually_exclusive_group()
    mxfp4_group.add_argument("--mxfp4-tutorial", action="store_true",
                             help="Use the fixed two-buffer fused-scale MXFP4 tutorial kernel")
    mxfp4_group.add_argument("--mxfp4-kernelC", dest="mxfp4_kernel_c", action="store_true",
                             help="Use the three/four-buffer fused-scale MXFP4 KernelC kernel")
    mxfp4_group.add_argument("--mxfp4-bk512", action="store_true",
                             help="Use the clustered, eight-warp-per-CTA 256x256x512 MXFP4 kernel with BF16 output")
    parser.add_argument("--mxfp4-cluster-shape", type=int, nargs=2, metavar=("CTA_M", "CTA_N"),
                        default=(4, 4),
                        help="Cluster shape for --mxfp4-bk512: 2 2, 2 4, 4 2, or 4 4")
    parser.add_argument("--mxfp4-k256-stages", action="store_true",
                        help="Load and consume two K128 fragments in each BK512 warp-pipeline pair")
    parser.add_argument("--mxfp4-overlap-cluster", action="store_true",
                        help="Overlap cluster synchronization with the final BK512 scaled WMMA")
    parser.add_argument("--resolve-partition-conflicts", action="store_true",
                        help="Use partition-aware gfx1250 WMMA/shared layouts for FP16/MXFP; plain FP8 always uses them")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true", help="Check output against torch")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark the selected kernel")
    parser.add_argument("--benchmark-mode", choices=["graph", "direct"], default="graph",
                        help="Use CUDA graph replay or direct synchronized CPU-wall timing")
    parser.add_argument("--warmup", type=int, default=10, help="Benchmark warmup iterations")
    parser.add_argument("--direct-iters", type=int, default=1000,
                        help="Timed iterations for --benchmark-mode direct")
    parser.add_argument("--probe-iters", type=int, default=20, help="Iterations for CUDA event timing probe")
    parser.add_argument("--graph-ms", type=float, default=100.0, help="Target CUDA graph body duration in ms")
    parser.add_argument("--n-replays", type=int, default=20, help="Number of CUDA graph replays to time")
    parser.add_argument("--iters-per-graph", type=int, default=None, help="Override graph body iteration count")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    dedicated_mxfp4 = args.mxfp4_tutorial or args.mxfp4_kernel_c or args.mxfp4_bk512
    if dedicated_mxfp4:
        if (args.mxfp or args.with_scale or args.with_a_scale or args.f16_cluster_kernel_c or args.fp8_cluster
                or args.fp8_cluster_bf16_style or args.bf16_kernelc_3pf or args.bf16_kernelc_2pf
                or args.bf16_cluster_4x4_tdm
                or args.bf16_cluster_4x4_identity_tdm or args.fp8_scaled_cluster_bk128 or args.kernel_c
                or args.async_copy_scale):
            raise ValueError(
                "Dedicated MXFP4 selectors cannot be combined with --mxfp, --with-scale, --with-a-scale, "
                "--f16-cluster-kernelC, --bf16-kernelc-3pf, --bf16-kernelc-2pf, --bf16-cluster-4x4-tdm, "
                "--bf16-cluster-4x4-identity-tdm, --fp8-cluster, --fp8-cluster-bf16-style, "
                "--fp8-scaled-cluster-bk128, --kernelC, or --async-copy-scale")
        args.dtype_a = "float4"
        args.dtype_b = "float4"
        args.scale_preshuffled = True
    bf16_kernelc_selectors = (
        args.bf16_kernelc_3pf,
        args.bf16_kernelc_2pf,
        args.bf16_cluster_4x4_tdm,
        args.bf16_cluster_4x4_identity_tdm,
    )
    if any(bf16_kernelc_selectors):
        if (args.f16_cluster_kernel_c or args.fp8_cluster or args.fp8_cluster_bf16_style
                or args.fp8_scaled_cluster_bk128 or args.kernel_c):
            raise ValueError("BF16 KernelC selectors cannot be combined with another kernel selector")
        if sum(bf16_kernelc_selectors) > 1:
            raise ValueError("BF16 KernelC selectors are mutually exclusive")
        args.dtype_a = "bfloat16"
        args.dtype_b = "bfloat16"
    if args.num_buffers is None:
        args.num_buffers = 3 if (args.mxfp4_kernel_c or args.f16_cluster_kernel_c or args.bf16_kernelc_3pf
                                 or args.bf16_cluster_4x4_tdm or args.bf16_cluster_4x4_identity_tdm
                                 or args.fp8_scaled_cluster_bk128) else 2
    if args.dtype_b is None:
        args.dtype_b = args.dtype_a
    plain_fp8 = not args.mxfp and args.dtype_a.startswith("float8") and args.dtype_b.startswith("float8")
    if args.with_scale and not plain_fp8:
        raise ValueError("--with-scale requires plain FP8 inputs without --mxfp")
    if args.with_scale and args.with_a_scale:
        raise ValueError("--with-scale and --with-a-scale are mutually exclusive")
    if args.kernel_c and not plain_fp8:
        raise ValueError("--kernelC is supported only by the dedicated plain-FP8 path")
    if args.f16_cluster_kernel_c and args.dtype_a != "float16":
        raise ValueError("--f16-cluster-kernelC requires --dtype-a float16")
    if args.fp8_cluster and not (plain_fp8 and args.with_scale):
        raise ValueError("--fp8-cluster requires the plain-FP8 --with-scale path")
    if args.fp8_cluster_bf16_style and not (plain_fp8 and args.with_scale):
        raise ValueError("--fp8-cluster-bf16-style requires the plain-FP8 --with-scale path")
    if tuple(args.fp8_cluster_shape) not in ((1, 1), (2, 2), (4, 4)):
        raise ValueError("--fp8-cluster-shape must be 1 1, 2 2, or 4 4")
    if tuple(args.fp8_cluster_shape) != (4, 4) and not args.fp8_cluster_bf16_style:
        raise ValueError("--fp8-cluster-shape applies only to --fp8-cluster-bf16-style")
    valid_mxfp4_cluster_shapes = ((2, 2), (2, 4), (4, 2), (4, 4))
    if tuple(args.mxfp4_cluster_shape) not in valid_mxfp4_cluster_shapes:
        raise ValueError("--mxfp4-cluster-shape must be 2 2, 2 4, 4 2, or 4 4")
    if tuple(args.mxfp4_cluster_shape) != (4, 4) and not args.mxfp4_bk512:
        raise ValueError("--mxfp4-cluster-shape applies only to --mxfp4-bk512")
    if args.mxfp4_k256_stages and not args.mxfp4_bk512:
        raise ValueError("--mxfp4-k256-stages applies only to --mxfp4-bk512")
    if args.mxfp4_overlap_cluster and not args.mxfp4_bk512:
        raise ValueError("--mxfp4-overlap-cluster applies only to --mxfp4-bk512")
    if args.fp8_tdm_schedule is None:
        args.fp8_tdm_schedule = "leading4" if args.fp8_cluster_bf16_style else "fused-pairs"
    elif not args.fp8_cluster_bf16_style:
        raise ValueError("--fp8-tdm-schedule applies only to --fp8-cluster-bf16-style")
    if args.fp8_scaled_cluster_bk128 and not (plain_fp8 and args.with_scale):
        raise ValueError("--fp8-scaled-cluster-bk128 requires the plain-FP8 --with-scale path")
    if args.fp8_bf16_output and not (plain_fp8 and args.with_scale):
        raise ValueError("--fp8-bf16-output requires the plain-FP8 --with-scale path")
    if args.fp8_scale_layout == "hipblaslt" and not args.fp8_cluster_bf16_style:
        raise ValueError("--fp8-scale-layout hipblaslt requires --fp8-cluster-bf16-style")
    partition_conflict_avoidance = plain_fp8 or dedicated_mxfp4 or args.resolve_partition_conflicts
    if args.BK is None:
        if args.bf16_kernelc_2pf or args.fp8_scaled_cluster_bk128:
            args.BK = 128
        elif any(bf16_kernelc_selectors):
            args.BK = 64
        elif args.mxfp4_bk512:
            args.BK = 512
        elif args.mxfp4_tutorial or (plain_fp8 and args.with_scale and not args.kernel_c):
            args.BK = 256
        elif plain_fp8 or args.mxfp4_kernel_c or args.dtype_a == "bfloat16":
            args.BK = 128
        else:
            args.BK = 64
    if not args.check and not args.benchmark:
        args.check = True

    print(
        f"({args.M=}, {args.N=}, {args.K=}), ({args.BM=}, {args.BN=}, {args.BK=}), "
        f"{args.dtype_a=}, {args.dtype_b=}, {args.num_warps=}, {args.num_buffers=}, "
        f"{args.transpose_b=}, {args.mxfp=}, {args.with_scale=}, {args.scale_preshuffled=}, "
        f"{args.async_copy_scale=}, {args.l2_prefetch_distance=}, "
        f"{args.f16_cluster_kernel_c=}, {args.bf16_kernelc_3pf=}, {args.bf16_kernelc_2pf=}, "
        f"{args.bf16_cluster_4x4_tdm=}, "
        f"{args.bf16_cluster_4x4_identity_tdm=}, {args.bf16_output=}, "
        f"{args.input_mode=}, "
        f"{args.fp8_cluster=}, {args.fp8_cluster_bf16_style=}, {args.fp8_cluster_shape=}, "
        f"{args.fp8_tdm_schedule=}, "
        f"{args.fp8_scaled_cluster_bk128=}, "
        f"{args.fp8_bf16_output=}, {args.fp8_scale_layout=}, "
        f"{args.kernel_c=}, {args.mxfp4_tutorial=}, {args.mxfp4_kernel_c=}, {args.mxfp4_bk512=}, "
        f"{args.mxfp4_cluster_shape=}, {args.mxfp4_k256_stages=}, "
        f"{args.mxfp4_overlap_cluster=}, "
        f"{partition_conflict_avoidance=}, sliceMN=True"
    )

    if dedicated_mxfp4:
        launch, check = _make_mxfp4_pipeline_case(args)
    elif args.dtype_a == "float16":
        if args.mxfp:
            raise ValueError("--mxfp does not apply to float16")
        launch, check = _make_f16_case(args)
    elif args.dtype_a == "bfloat16":
        if args.mxfp:
            raise ValueError("--mxfp does not apply to bfloat16")
        launch, check = _make_bf16_case(args)
    elif plain_fp8:
        launch, check = _make_fp8_scaled_case(args) if args.with_scale else _make_fp8_case(args)
    else:
        launch, check = _make_mxfp_case(args)

    if args.check:
        check()
    if args.benchmark:
        if args.benchmark_mode == "direct":
            _run_direct_benchmark(launch, args.M, args.N, args.K, args.warmup, args.direct_iters)
        else:
            _run_benchmark(launch, args.M, args.N, args.K, args.warmup, args.probe_iters, args.graph_ms,
                           args.n_replays, args.iters_per_graph)
