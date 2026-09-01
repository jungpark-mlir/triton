#!/usr/bin/env python3
"""Standalone 2x2-CTA MXFP8 GEMM for AMD GFX1250.

The kernel computes E4M3 x E4M3 with E8M0 block-32 scales and writes BF16
or FP32.
Its fixed aggregate tile is 512x512x256 (256x256 output elements per CTA).

Run this file with a Triton checkout that includes the GFX1250 Gluon TDM and
scaled-WMMA support, for example:

  PYTHONPATH=~/mnt/main/triton/python python mxfp8_2x2_cluster.py --check
"""

import argparse
import time

import torch
import triton
from triton._C.libtriton.gluon_ir import make_cga_layout
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm
from triton.tools.mxfp import MXScaleTensor


BLOCK_M = 512
BLOCK_N = 512
BLOCK_K = 256
SCALE_BLOCK = 32
NUM_WARPS = 8
NUM_CTAS = 4
GROUP_SIZE_M = 4
NUM_XCDS = 8
TARGET_WGPS = 512

JIT_BLOCK_M = gl.constexpr(BLOCK_M)
JIT_BLOCK_N = gl.constexpr(BLOCK_N)
JIT_BLOCK_K = gl.constexpr(BLOCK_K)
JIT_SCALE_BLOCK = gl.constexpr(SCALE_BLOCK)
JIT_NUM_CTAS = gl.constexpr(NUM_CTAS)
JIT_GROUP_SIZE_M = gl.constexpr(GROUP_SIZE_M)
JIT_NUM_XCDS = gl.constexpr(NUM_XCDS)


@gluon.jit
def get_xcd_swizzled_pid(
        GRID_MN: gl.constexpr, XCD_SWIZZLE_CHUNK: gl.constexpr,
        XCD_SWIZZLE_PERMUTE: gl.constexpr,
        XCD_SWIZZLE_OFFSET: gl.constexpr):
    pid = gl.program_id(axis=0)
    pids_per_xcd = (GRID_MN + JIT_NUM_XCDS - 1) // JIT_NUM_XCDS
    tall_xcds = GRID_MN % JIT_NUM_XCDS
    tall_xcds = JIT_NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % JIT_NUM_XCDS
    local_pid = pid // JIT_NUM_XCDS
    if (GRID_MN % JIT_NUM_XCDS == 0
            and pids_per_xcd % XCD_SWIZZLE_CHUNK == 0):
        # Assign contiguous N-tile chunks to each XCD. A chunk of one leaves
        # logical PIDs interleaved across XCDs; pids_per_xcd reproduces the
        # conventional contiguous-per-XCD mapping.
        chunk_group = local_pid // XCD_SWIZZLE_CHUNK
        chunk_offset = local_pid % XCD_SWIZZLE_CHUNK
        tile_xcd = xcd
        if XCD_SWIZZLE_PERMUTE == "rotate":
            tile_xcd = (xcd + chunk_group) % JIT_NUM_XCDS
        elif XCD_SWIZZLE_PERMUTE == "xor":
            tile_xcd = xcd ^ (chunk_group % JIT_NUM_XCDS)
        elif XCD_SWIZZLE_PERMUTE == "serpentine":
            tile_xcd = (
                JIT_NUM_XCDS - 1 - xcd if chunk_group % 2 else xcd)
        pid = ((chunk_group * JIT_NUM_XCDS + tile_xcd)
               * XCD_SWIZZLE_CHUNK + chunk_offset)
    else:
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = (tall_xcds * pids_per_xcd
                   + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid)
    return (pid + XCD_SWIZZLE_OFFSET) % GRID_MN


@gluon.jit
def get_grouped_tile_coords(tile_idx, M, N):
    num_pid_m = gl.cdiv(M, JIT_BLOCK_M)
    num_pid_n = gl.cdiv(N, JIT_BLOCK_N)
    num_pid_in_group = JIT_GROUP_SIZE_M * num_pid_n
    group_id = tile_idx // num_pid_in_group
    first_pid_m = group_id * JIT_GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, JIT_GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_idx % num_pid_in_group) % group_size_m)
    pid_n = (tile_idx % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@gluon.jit
def cluster_wait():
    gl.amd.gfx1250.cluster.arrive()
    gl.amd.gfx1250.cluster.wait()


@gluon.jit
def load_scale(scale_buffer, slot, start_k: gl.constexpr, layout: gl.constexpr):
    # Undo the host preshuffle into the logical [nonK, K/32] scale tile.
    scale_slice = scale_buffer.index(slot).reshape((4, 2, 32, 4, 4)).permute(
        (0, 3, 2, 1, 4)).reshape((JIT_BLOCK_M, JIT_BLOCK_K // JIT_SCALE_BLOCK))
    return scale_slice.slice(0, JIT_BLOCK_M, 0).slice(
        start_k, JIT_BLOCK_K // JIT_SCALE_BLOCK // 2, 1).load(layout=layout)


@gluon.jit
def issue_fused_scales(as_desc, bs_desc, as_buf, bs_buf, tile_idx, slot):
    scale_k = tile_idx * (JIT_BLOCK_K // JIT_SCALE_BLOCK * 128)
    as_load_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, scale_k])
    bs_load_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, scale_k])
    tdm.async_load_fused([
        (as_load_desc, as_buf.index(slot), 0b00001111),
        (bs_load_desc, bs_buf.index(slot), 0b11110000),
    ])


@gluon.jit
def issue_separate_data(a_desc, b_desc, a_buf, b_buf, tile_idx, slot):
    tile_k = tile_idx * JIT_BLOCK_K
    a_load_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, tile_k])
    b_load_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, tile_k])
    tdm.async_load(a_load_desc, dest=a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(b_load_desc, dest=b_buf.index(slot), warp_used_hint=0b11110000)


@gluon.jit
def prefetch_next_output_tile(
        a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am,
        stride_ak, stride_bk, stride_bn, stride_scale, pid_m, pid_n, pred,
        shared_layout_a: gl.constexpr, shared_layout_b: gl.constexpr,
        shared_scale_a: gl.constexpr, shared_scale_b: gl.constexpr,
        NEXT_PREFETCH: gl.constexpr):
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * JIT_BLOCK_M * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(JIT_BLOCK_M, JIT_BLOCK_K),
        layout=shared_layout_a)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * JIT_BLOCK_N * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(JIT_BLOCK_N, JIT_BLOCK_K),
        layout=shared_layout_b)
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + (pid_m * JIT_BLOCK_M // 128) * stride_scale,
        shape=(M // 128, K // JIT_SCALE_BLOCK * 128), strides=(stride_scale, 1),
        block_shape=(4, 1024), layout=shared_scale_a)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * JIT_BLOCK_N // 128) * stride_scale,
        shape=(N // 128, K // JIT_SCALE_BLOCK * 128), strides=(stride_scale, 1),
        block_shape=(4, 1024), layout=shared_scale_b)

    for tile_idx in gl.static_range(NEXT_PREFETCH):
        tile_k = tile_idx * JIT_BLOCK_K
        scale_k = tile_idx * (JIT_BLOCK_K // JIT_SCALE_BLOCK * 128)
        tdm.prefetch(a_desc, [0, tile_k], pred=pred)
        tdm.prefetch(b_desc, [0, tile_k], pred=pred)
        tdm.prefetch(as_desc, [0, scale_k], pred=pred)
        tdm.prefetch(bs_desc, [0, scale_k], pred=pred)


@gluon.jit
def consume_epilogue_tile(a_buf, b_buf, as_buf, bs_buf, slot, acc,
                          dot_a: gl.constexpr, dot_b: gl.constexpr,
                          scale_a_layout: gl.constexpr,
                          scale_b_layout: gl.constexpr):
    # Keep the two resident waves phase-shifted while draining the prefetched
    # K-tail instead of falling back to a single, fully serial consumer.
    with gl.amd.warp_pipeline_stage("stage0_epilogue", priority=0):
        a0 = a_buf.index(slot).slice(0, JIT_BLOCK_M, 0).slice(
            0, JIT_BLOCK_K // 2, 1).load(layout=dot_a)
        as0 = load_scale(as_buf, slot, 0, scale_a_layout)
        b0 = b_buf.index(slot).slice(0, JIT_BLOCK_N, 0).slice(
            0, JIT_BLOCK_K // 2, 1).permute([1, 0]).load(layout=dot_b)
        bs0 = load_scale(bs_buf, slot, 0, scale_b_layout)
    with gl.amd.warp_pipeline_stage("stage1_epilogue", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0, bs0, "e4m3", acc)

    with gl.amd.warp_pipeline_stage("stage0_epilogue", priority=0):
        a1 = a_buf.index(slot).slice(0, JIT_BLOCK_M, 0).slice(
            JIT_BLOCK_K // 2, JIT_BLOCK_K // 2, 1).load(layout=dot_a)
        as1 = load_scale(
            as_buf, slot, JIT_BLOCK_K // JIT_SCALE_BLOCK // 2, scale_a_layout)
        b1 = b_buf.index(slot).slice(0, JIT_BLOCK_N, 0).slice(
            JIT_BLOCK_K // 2, JIT_BLOCK_K // 2, 1).permute([1, 0]).load(layout=dot_b)
        bs1 = load_scale(
            bs_buf, slot, JIT_BLOCK_K // JIT_SCALE_BLOCK // 2, scale_b_layout)
    with gl.amd.warp_pipeline_stage("stage1_epilogue", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1, bs1, "e4m3", acc)
    return acc


@gluon.jit
def consume_plain_tile(a_buf, b_buf, as_buf, bs_buf, slot, acc,
                       dot_a: gl.constexpr, dot_b: gl.constexpr,
                       scale_a_layout: gl.constexpr,
                       scale_b_layout: gl.constexpr):
    a0 = a_buf.index(slot).slice(
        0, JIT_BLOCK_M, 0).slice(0, JIT_BLOCK_K // 2, 1).load(layout=dot_a)
    as0 = load_scale(as_buf, slot, 0, scale_a_layout)
    b0 = b_buf.index(slot).slice(0, JIT_BLOCK_N, 0).slice(
        0, JIT_BLOCK_K // 2, 1).permute([1, 0]).load(layout=dot_b)
    bs0 = load_scale(bs_buf, slot, 0, scale_b_layout)
    acc = gl.amd.gfx1250.wmma_scaled(
        a0, as0, "e4m3", b0, bs0, "e4m3", acc)

    a1 = a_buf.index(slot).slice(0, JIT_BLOCK_M, 0).slice(
        JIT_BLOCK_K // 2, JIT_BLOCK_K // 2, 1).load(layout=dot_a)
    as1 = load_scale(
        as_buf, slot, JIT_BLOCK_K // JIT_SCALE_BLOCK // 2, scale_a_layout)
    b1 = b_buf.index(slot).slice(0, JIT_BLOCK_N, 0).slice(
        JIT_BLOCK_K // 2, JIT_BLOCK_K // 2, 1).permute([1, 0]).load(layout=dot_b)
    bs1 = load_scale(
        bs_buf, slot, JIT_BLOCK_K // JIT_SCALE_BLOCK // 2, scale_b_layout)
    return gl.amd.gfx1250.wmma_scaled(
        a1, as1, "e4m3", b1, bs1, "e4m3", acc)


@gluon.jit
def consume_and_refill(a_buf, b_buf, as_buf, bs_buf, slot, a_desc, b_desc,
                       as_desc, bs_desc, refill_idx, acc, dot_a: gl.constexpr,
                       dot_b: gl.constexpr, scale_a_layout: gl.constexpr,
                       scale_b_layout: gl.constexpr):
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a0 = a_buf.index(slot).slice(0, JIT_BLOCK_M, 0).slice(0, JIT_BLOCK_K // 2, 1).load(layout=dot_a)
        as0 = load_scale(as_buf, slot, 0, scale_a_layout)
        b0 = b_buf.index(slot).slice(0, JIT_BLOCK_N, 0).slice(
            0, JIT_BLOCK_K // 2, 1).permute([1, 0]).load(layout=dot_b)
        bs0 = load_scale(bs_buf, slot, 0, scale_b_layout)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(a0, as0, "e4m3", b0, bs0, "e4m3", acc)

    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a1 = a_buf.index(slot).slice(0, JIT_BLOCK_M, 0).slice(
            JIT_BLOCK_K // 2, JIT_BLOCK_K // 2, 1).load(layout=dot_a)
        as1 = load_scale(
            as_buf, slot, JIT_BLOCK_K // JIT_SCALE_BLOCK // 2, scale_a_layout)
        b1 = b_buf.index(slot).slice(0, JIT_BLOCK_N, 0).slice(
            JIT_BLOCK_K // 2, JIT_BLOCK_K // 2, 1).permute([1, 0]).load(layout=dot_b)
        bs1 = load_scale(
            bs_buf, slot, JIT_BLOCK_K // JIT_SCALE_BLOCK // 2, scale_b_layout)
        gl.amd.gfx1250.cluster.arrive()
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(a1, as1, "e4m3", b1, bs1, "e4m3", acc)
        gl.amd.gfx1250.cluster.wait()
        # Validated request order: A data, B data, then fused A/B scales.
        issue_separate_data(a_desc, b_desc, a_buf, b_buf, refill_idx, slot)
        issue_fused_scales(as_desc, bs_desc, as_buf, bs_buf, refill_idx, slot)
    return acc


@gluon.jit
def store_output(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
                 shared, shared_layout: gl.constexpr):
    shared.store(acc.to(c_ptr.type.element_ty))
    desc = tdm.make_tensor_descriptor(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        block_shape=(JIT_BLOCK_M, JIT_BLOCK_N), layout=shared_layout)
    tdm.async_store(desc, [pid_m * JIT_BLOCK_M, pid_n * JIT_BLOCK_N], shared)
    tdm.async_wait(2)


@gluon.jit
def store_output_direct(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N,
                        acc, wmma_layout: gl.constexpr,
                        BUFFER_STORE: gl.constexpr):
    offs_m = pid_m * JIT_BLOCK_M + gl.arange(
        0, JIT_BLOCK_M, layout=gl.SliceLayout(1, wmma_layout))
    offs_n = pid_n * JIT_BLOCK_N + gl.arange(
        0, JIT_BLOCK_N, layout=gl.SliceLayout(0, wmma_layout))
    offsets = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    output = acc.to(c_ptr.type.element_ty)
    if BUFFER_STORE:
        gl.amd.gfx1250.buffer_store(output, c_ptr, offsets, mask=mask)
    else:
        gl.store(c_ptr + offsets, output, mask=mask)


@gluon.jit
def store_output_quarters(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N,
                          acc, shared, shared_layout: gl.constexpr):
    quarter: gl.constexpr = 128
    acc4 = acc.reshape((2, 256, 2, 256)).permute((0, 2, 1, 3))
    desc = tdm.make_tensor_descriptor(
        base=c_ptr, shape=(M // 256, N // 256, 256, 256),
        strides=(256 * stride_cm, 256 * stride_cn, stride_cm, stride_cn),
        block_shape=(2, 2, quarter, quarter), layout=shared_layout)

    for qm in gl.static_range(2):
        for qn in gl.static_range(2):
            # Drain the previous quarter (or the preceding output tile's final
            # quarter) only when this dedicated buffer is about to be reused.
            tdm.async_wait(0)
            output = gl.amd.slice(
                acc4, [2, 2, quarter, quarter],
                [0, 0, qm * quarter, qn * quarter])
            shared.store(output.to(c_ptr.type.element_ty))
            tdm.async_store(
                desc,
                [2 * pid_m, 2 * pid_n, qm * quarter, qn * quarter],
                shared)
    # Deliberately leave the fourth store outstanding. The next output tile can
    # start using its disjoint operand buffers before this buffer is reused.


@gluon.jit
def compute_output_tile(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, pid_m, pid_n, next_pid_m, next_pid_n, has_next,
        quarter_c_buf,
        dot_a: gl.constexpr, dot_b: gl.constexpr,
        scale_a_layout: gl.constexpr, scale_b_layout: gl.constexpr,
        shared_layout_a: gl.constexpr, shared_layout_b: gl.constexpr,
        shared_scale_a: gl.constexpr, shared_scale_b: gl.constexpr,
        quarter_c_layout: gl.constexpr,
        wmma_layout: gl.constexpr, TAIL_MODE: gl.constexpr,
        OUTPUT_STORE: gl.constexpr, NEXT_PREFETCH: gl.constexpr):
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, JIT_BLOCK_M, JIT_BLOCK_K], shared_layout_a)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, JIT_BLOCK_N, JIT_BLOCK_K], shared_layout_b)
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty, [2, 4, 1024], shared_scale_a)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty, [2, 4, 1024], shared_scale_b)

    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * JIT_BLOCK_M * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(JIT_BLOCK_M, JIT_BLOCK_K),
        layout=shared_layout_a)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * JIT_BLOCK_N * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(JIT_BLOCK_N, JIT_BLOCK_K),
        layout=shared_layout_b)
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + (pid_m * JIT_BLOCK_M // 128) * stride_scale,
        shape=(M // 128, K // JIT_SCALE_BLOCK * 128), strides=(stride_scale, 1),
        block_shape=(4, 1024), layout=shared_scale_a)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * JIT_BLOCK_N // 128) * stride_scale,
        shape=(N // 128, K // JIT_SCALE_BLOCK * 128), strides=(stride_scale, 1),
        block_shape=(4, 1024), layout=shared_scale_b)

    for prefetch_idx in gl.static_range(2):
        issue_separate_data(a_desc, b_desc, a_buf, b_buf, prefetch_idx, prefetch_idx)
        issue_fused_scales(as_desc, bs_desc, as_buf, bs_buf, prefetch_idx, prefetch_idx)

    tdm.async_wait(2)
    cluster_wait()
    acc = gl.zeros((JIT_BLOCK_M, JIT_BLOCK_N), dtype=gl.float32, layout=wmma_layout)
    iter_max = gl.cdiv(K, JIT_BLOCK_K)
    gl.assume(iter_max >= 2)
    for tile_idx in range(0, iter_max - 2):
        slot = tile_idx % 2
        acc = consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, slot, a_desc, b_desc, as_desc,
            bs_desc, tile_idx + 2, acc, dot_a, dot_b, scale_a_layout,
            scale_b_layout)
        tdm.async_wait(2)

    if TAIL_MODE == "nested":
        # Keep each tail step in an inner loop so the warp-pipeline pass does
        # not select the enclosing persistent output-tile loop.
        for tail_idx in range(iter_max - 2, iter_max - 1):
            slot = tail_idx % 2
            acc = consume_epilogue_tile(
                a_buf, b_buf, as_buf, bs_buf, slot, acc, dot_a, dot_b,
                scale_a_layout, scale_b_layout)

        tdm.async_wait(0)
        cluster_wait()
        for tail_idx in range(iter_max - 1, iter_max):
            slot = tail_idx % 2
            acc = consume_epilogue_tile(
                a_buf, b_buf, as_buf, bs_buf, slot, acc, dot_a, dot_b,
                scale_a_layout, scale_b_layout)
    elif TAIL_MODE == "plain":
        penultimate_slot = (iter_max - 2) % 2
        acc = consume_plain_tile(
            a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, dot_a, dot_b,
            scale_a_layout, scale_b_layout)

        tdm.async_wait(0)
        cluster_wait()
        last_slot = (iter_max - 1) % 2
        acc = consume_plain_tile(
            a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b,
            scale_a_layout, scale_b_layout)
    else:
        penultimate_slot = (iter_max - 2) % 2
        acc = consume_epilogue_tile(
            a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, dot_a, dot_b,
            scale_a_layout, scale_b_layout)

        tdm.async_wait(0)
        cluster_wait()
        last_slot = (iter_max - 1) % 2
        acc = consume_epilogue_tile(
            a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b,
            scale_a_layout, scale_b_layout)

    # TDM output staging aliases the operand phase in LDS and must synchronize
    # before changing that storage. Register stores can issue first; the
    # persistent-loop barrier below still protects the next tile's LDS reuse.
    if OUTPUT_STORE != "direct" and OUTPUT_STORE != "buffer":
        cluster_wait()

    if NEXT_PREFETCH:
        prefetch_next_output_tile(
            a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am,
            stride_ak, stride_bk, stride_bn, stride_scale, next_pid_m,
            next_pid_n, has_next, shared_layout_a, shared_layout_b,
            shared_scale_a, shared_scale_b, NEXT_PREFETCH)

    if OUTPUT_STORE == "direct" or OUTPUT_STORE == "buffer":
        store_output_direct(
            c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
            wmma_layout, BUFFER_STORE=OUTPUT_STORE == "buffer")
    elif OUTPUT_STORE == "tdm-quarter":
        store_output_quarters(
            c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
            quarter_c_buf, quarter_c_layout)
    else:
        # Allocate the output staging tile after the operand buffers' last use
        # so shared-memory liveness can overlay the two phases.
        shared_layout_c: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[JIT_BLOCK_N // 2, 8]], [JIT_BLOCK_M, JIT_BLOCK_N], [1, 0],
            wmma_layout.cga_layout)
        c_buf = gl.allocate_shared_memory(
            c_ptr.type.element_ty, [JIT_BLOCK_M, JIT_BLOCK_N], shared_layout_c)
        store_output(
            c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc, c_buf,
            shared_layout_c)


@gluon.jit
def mxfp8_2x2_cluster_kernel(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr, OUTPUT_ITERS: gl.constexpr,
        SINGLE_TILE_PER_PROGRAM: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
        OUTPUT_STORE: gl.constexpr, PERSISTENT_TAIL: gl.constexpr,
        NEXT_PREFETCH: gl.constexpr, XCD_SWIZZLE_CHUNK: gl.constexpr,
        XCD_SWIZZLE_PERMUTE: gl.constexpr,
        XCD_SWIZZLE_OFFSET: gl.constexpr):
    gl.static_assert(gl.num_ctas() == JIT_NUM_CTAS)
    persistent_pid = get_xcd_swizzled_pid(
        GRID_MN, XCD_SWIZZLE_CHUNK, XCD_SWIZZLE_PERMUTE,
        XCD_SWIZZLE_OFFSET)
    total_tiles = gl.cdiv(M, JIT_BLOCK_M) * gl.cdiv(N, JIT_BLOCK_N)

    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_a, [JIT_BLOCK_M, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_b, [JIT_BLOCK_N, 4])
    quarter_c_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[128, 8]], [2, 2, 128, 128], [3, 2, 1, 0],
        ((1, 0, 0, 0), (0, 1, 0, 0)))
    if OUTPUT_STORE == "tdm-quarter":
        quarter_c_buf = gl.allocate_shared_memory(
            c_ptr.type.element_ty, [2, 2, 128, 128], quarter_c_layout)
    else:
        # Compile-time dead placeholder for the non-quarter epilogues.
        quarter_c_buf = c_ptr

    if SINGLE_TILE_PER_PROGRAM:
        # The target shape has no more tiles than resident WGP programs. Avoid
        # persistent-loop control while retaining the same bounded scheduler.
        output_tile = persistent_pid
        pid_m, pid_n = get_grouped_tile_coords(output_tile, M, N)
        compute_output_tile(
            a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
            stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
            stride_cn, stride_scale, pid_m, pid_n, pid_m, pid_n, False,
            quarter_c_buf, dot_a, dot_b, scale_a_layout,
            scale_b_layout,
            SHARED_LAYOUT_A,
            SHARED_LAYOUT_B,
            SHARED_SCALE_A, SHARED_SCALE_B, quarter_c_layout, WMMA_LAYOUT,
            TAIL_MODE="explicit", OUTPUT_STORE=OUTPUT_STORE,
            NEXT_PREFETCH=NEXT_PREFETCH)
    else:
        # OUTPUT_ITERS is a host-computed constexpr. The loop-unroll pass
        # removes this outer loop before warp pipelining the K-tail loops.
        for output_iteration in range(OUTPUT_ITERS):
            output_tile = persistent_pid + output_iteration * GRID_MN
            if output_tile < total_tiles:
                pid_m, pid_n = get_grouped_tile_coords(output_tile, M, N)
                next_output_tile = min(
                    output_tile + GRID_MN, total_tiles - 1)
                next_pid_m, next_pid_n = get_grouped_tile_coords(
                    next_output_tile, M, N)
                has_next = output_tile + GRID_MN < total_tiles
                compute_output_tile(
                    a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
                    stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                    stride_cn, stride_scale, pid_m, pid_n, next_pid_m,
                    next_pid_n, has_next, quarter_c_buf,
                    dot_a, dot_b, scale_a_layout, scale_b_layout, SHARED_LAYOUT_A,
                    SHARED_LAYOUT_B, SHARED_SCALE_A, SHARED_SCALE_B,
                    quarter_c_layout, WMMA_LAYOUT,
                    TAIL_MODE=PERSISTENT_TAIL, OUTPUT_STORE=OUTPUT_STORE,
                    NEXT_PREFETCH=NEXT_PREFETCH)

                if output_tile + GRID_MN < total_tiles:
                    cluster_wait()

    if OUTPUT_STORE == "tdm-quarter":
        tdm.async_wait(0)


def build_layouts(output_store):
    cga_layout = make_cga_layout([2, 2], [2, 2], [0, 1])
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 16]], [BLOCK_M, BLOCK_K], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 16]], [BLOCK_N, BLOCK_K], [1, 0])
    slice_mn = 128 if output_store == "tdm-quarter" else 256
    local_layouts = gl.amd.gfx1250.make_partitioned_dot_layouts(
        BLOCK_M, BLOCK_N, local_a, local_b, NUM_WARPS, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=slice_mn,
        slice_n=slice_mn)
    local_wmma = local_layouts[2]
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga_layout)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]]) for basis in dot_b.cga_layout)

    shared_a = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 16]], [BLOCK_M, BLOCK_K], [1, 0], cga_a)
    shared_b = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 16]], [BLOCK_N, BLOCK_K], [1, 0], cga_b)
    shared_as = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [4, 1024], [1, 0], cga_a)
    shared_bs = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [4, 1024], [1, 0], cga_b)
    return shared_a, shared_b, shared_as, shared_bs, wmma


def pack_scale(scale):
    non_k, k_scale = scale.shape
    scale = scale.view(non_k // 128, 4, 32, k_scale // 4, 4)
    scale = scale.permute(0, 3, 2, 1, 4).contiguous()
    return scale.view(non_k // 128, k_scale * 128)


def init_fp8(shape, mode, cosine=False):
    if mode == "random":
        return torch.randint(20, 40, shape, dtype=torch.uint8).view(torch.float8_e4m3fn)
    output = torch.empty(shape, dtype=torch.float8_e4m3fn)
    flat = output.view(-1)
    chunk = 8 * 1024 * 1024
    for begin in range(0, flat.numel(), chunk):
        end = min(begin + chunk, flat.numel())
        angle = torch.arange(begin, end, dtype=torch.float64)
        value = torch.cos(angle) if cosine else torch.sin(angle)
        flat[begin:end].copy_(value.float())
    return output


def reference_gemm(a, b, a_scale, b_scale, output_dtype):
    a_scale_f32 = a_scale.to(torch.float32).repeat_interleave(SCALE_BLOCK, dim=1)
    b_scale_f32 = b_scale.to(torch.float32).repeat_interleave(SCALE_BLOCK, dim=1).T.contiguous()
    return torch.matmul(
        a.to(torch.float32) * a_scale_f32,
        b.to(torch.float32) * b_scale_f32,
    ).to(output_dtype)


def make_case(
        M, N, K, mode, seed, need_reference, persistent_wgps, output_store,
        output_dtype, persistent_tail, next_tile_prefetch, xcd_swizzle_chunk,
        xcd_swizzle_permute, xcd_swizzle_offset):
    if M % BLOCK_M or N % BLOCK_N or K % BLOCK_K:
        raise ValueError("M, N, and K must be divisible by 512, 512, and 256")
    if K < 2 * BLOCK_K:
        raise ValueError("K must contain at least two BK256 tiles")

    torch.manual_seed(seed)
    a = init_fp8((M, K), mode)
    b = init_fp8((K, N), mode, cosine=True)
    scale_k = K // SCALE_BLOCK
    a_scale_obj = MXScaleTensor(size=(M, scale_k)).random(low=1.0, high=32.0)
    b_scale_obj = MXScaleTensor(size=(N, scale_k)).random(low=1.0, high=32.0)
    torch_output_dtype = (
        torch.bfloat16 if output_dtype == "bf16" else torch.float32)
    reference = (
        reference_gemm(
            a, b, a_scale_obj, b_scale_obj, torch_output_dtype)
        if need_reference else None)

    a_d = a.contiguous().cuda()
    b_d = b.T.contiguous().cuda()
    a_scale_d = pack_scale(a_scale_obj.data).cuda()
    b_scale_d = pack_scale(b_scale_obj.data).cuda()
    c_d = torch.empty((M, N), dtype=torch_output_dtype, device="cuda")
    layouts = build_layouts(output_store)
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    persistent_programs = max(1, persistent_wgps // NUM_CTAS)
    grid = (min(total_tiles, persistent_programs), 1)
    output_iters = triton.cdiv(total_tiles, grid[0])

    def launch(
            offset=xcd_swizzle_offset, chunk=xcd_swizzle_chunk,
            permute=xcd_swizzle_permute):
        shared_a, shared_b, shared_as, shared_bs, wmma = layouts
        return mxfp8_2x2_cluster_kernel[grid](
            a_d, b_d, c_d, a_scale_d, b_scale_d, M, N, K,
            a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1), b_scale_d.stride(0),
            GRID_MN=grid[0], OUTPUT_ITERS=output_iters,
            SINGLE_TILE_PER_PROGRAM=total_tiles <= persistent_programs,
            SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
            SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
            OUTPUT_STORE=output_store, PERSISTENT_TAIL=persistent_tail,
            NEXT_PREFETCH=next_tile_prefetch,
            XCD_SWIZZLE_CHUNK=chunk,
            XCD_SWIZZLE_PERMUTE=permute,
            XCD_SWIZZLE_OFFSET=offset,
            num_ctas=NUM_CTAS, num_warps=NUM_WARPS,
            llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),), waves_per_eu=2)

    return launch, c_d, reference


def check(launch, output, reference):
    output.zero_()
    launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu(), reference, rtol=1e-2, atol=5e-1)
    print("result verified", flush=True)


def event_probe(fn, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def capture_graph(fn, iterations):
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
    return graph


def benchmark(launch, M, N, K, warmup, probe_iters, graph_ms, replays):
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    probe_ms = event_probe(launch, probe_iters)
    iterations = max(1, int(graph_ms / max(probe_ms, 1e-6)))
    graph = capture_graph(launch, iterations)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    total_iterations = iterations * replays
    per_iter = elapsed / total_iterations
    tflops = 2 * M * N * K / per_iter / 1e12

    print(f"probe per-iter   : {probe_ms * 1e3:.2f} us")
    print(f"iters per graph  : {iterations}")
    print(f"replays          : {replays}")
    print(f"per-iter         : {per_iter * 1e6:.2f} us")
    print(f"TFLOPS           : {tflops:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-M", type=int, default=512)
    parser.add_argument("-N", type=int, default=512)
    parser.add_argument("-K", type=int, default=1536)
    parser.add_argument("--input-mode", choices=["random", "trig"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--probe-iters", type=int, default=20)
    parser.add_argument("--graph-ms", type=float, default=100.0)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument(
        "--persistent-wgps", type=int, default=TARGET_WGPS,
        help="physical WGP budget (a 2x2 clustered program consumes four)")
    parser.add_argument(
        "--output-store",
        choices=["tdm", "tdm-quarter", "direct", "buffer"],
        default="tdm",
        help="output epilogue implementation")
    parser.add_argument(
        "--output-dtype", choices=["bf16", "fp32"], default="bf16",
        help="output element type; FP32 is supported by direct register stores")
    parser.add_argument(
        "--persistent-tail", choices=["nested", "plain", "explicit"],
        default="plain", help="K-tail lowering inside the persistent loop")
    parser.add_argument(
        "--next-tile-prefetch", type=int, choices=[0, 1, 2, 4, 6], default=0,
        help="number of next-output K tiles to prefetch into L2")
    parser.add_argument(
        "--xcd-swizzle-chunk", type=int, choices=[1, 2, 4, 8, 16], default=2,
        help="contiguous logical N tiles assigned to each XCD at a time")
    parser.add_argument(
        "--xcd-swizzle-permute",
        choices=["identity", "rotate", "xor", "serpentine"],
        default="identity",
        help="per-cohort permutation of logical N chunks across XCDs")
    parser.add_argument(
        "--xcd-swizzle-offset", type=int, default=0,
        help="cyclic logical tile offset after XCD assignment")
    args = parser.parse_args()
    if not args.check and not args.benchmark:
        args.check = True

    if args.persistent_wgps < NUM_CTAS:
        parser.error(f"--persistent-wgps must be at least {NUM_CTAS}")
    if (args.output_dtype == "fp32"
            and args.output_store not in ("direct", "buffer")):
        parser.error(
            "--output-dtype fp32 requires --output-store direct or buffer")

    launch, output, reference = make_case(
        args.M, args.N, args.K, args.input_mode, args.seed, args.check,
        args.persistent_wgps, args.output_store, args.output_dtype,
        args.persistent_tail, args.next_tile_prefetch,
        args.xcd_swizzle_chunk, args.xcd_swizzle_permute,
        args.xcd_swizzle_offset)
    if args.check:
        check(launch, output, reference)
    if args.benchmark:
        benchmark(
            launch, args.M, args.N, args.K, args.warmup, args.probe_iters,
            args.graph_ms, args.replays)


if __name__ == "__main__":
    main()
