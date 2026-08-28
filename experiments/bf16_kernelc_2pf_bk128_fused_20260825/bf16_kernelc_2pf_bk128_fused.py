import argparse
import time

import torch
import triton
from triton._C.libtriton.gluon_ir import make_cga_layout
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm


BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
NUM_WARPS = 8
NUM_BUFFERS = 2
CLUSTER_NUM_BUFFERS = 2
CLUSTER_CTAS_M = 4
CLUSTER_CTAS_N = 4


@gluon.jit
def get_xcd_swizzled_pids(M, N, GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr,
                          GROUP_SIZE_M: gl.constexpr, BLOCK_M: gl.constexpr,
                          BLOCK_N: gl.constexpr):
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

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@gluon.jit
def issue_loads(a_desc, b_desc, a_dst, b_dst, BLOCK_K: gl.constexpr):
    tdm.async_load(a_desc, [0, 0], a_dst, warp_used_hint=0b00001111)
    tdm.async_load(b_desc, [0, 0], b_dst, warp_used_hint=0b00001111)
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K])
    return a_desc, b_desc


@gluon.jit
def consume_slot(a_buf, b_buf, slot, acc, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
                 BLOCK_K: gl.constexpr, WAIT_FOR_TDM: gl.constexpr,
                 WAIT_FOR_CLUSTER: gl.constexpr):
    half_k: gl.constexpr = BLOCK_K // 2
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        if WAIT_FOR_TDM:
            tdm.async_wait(0)
        if WAIT_FOR_CLUSTER:
            gl.amd.gfx1250.cluster.arrive()
            gl.amd.gfx1250.cluster.wait()
        a = a_buf.index(slot).slice(0, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(0, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)

    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(half_k, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(half_k, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    return acc


@gluon.jit
def consume_and_refill(a_buf, b_buf, slot, a_desc, b_desc, acc,
                       DOT_A: gl.constexpr, DOT_B: gl.constexpr, BLOCK_K: gl.constexpr):
    half_k: gl.constexpr = BLOCK_K // 2
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(0, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(0, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)

    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(half_k, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(half_k, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        # The stage boundary completes LDS reads before TDM overwrites the
        # double-buffered slot. WMMA consumes only the register operands.
        a_desc, b_desc = issue_loads(
            a_desc, b_desc, a_buf.index(slot), b_buf.index(slot), BLOCK_K)
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    return a_desc, b_desc, acc


@gluon.jit
def cluster_consume_and_refill(a_buf, b_buf, slot, a_desc, b_desc, acc,
                               DOT_A: gl.constexpr, DOT_B: gl.constexpr,
                               BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                               BLOCK_K: gl.constexpr):
    half_k: gl.constexpr = BLOCK_K // 2
    wmma_k: gl.constexpr = half_k // 2
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(0, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(0, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)

    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(half_k, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(half_k, half_k, 1).permute([1, 0]).load(layout=DOT_B)
        gl.amd.gfx1250.cluster.arrive()
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        a0 = gl.amd.slice(a, [BLOCK_M, wmma_k], [0, 0])
        b0 = gl.amd.slice(b, [wmma_k, BLOCK_N], [0, 0])
        a1 = gl.amd.slice(a, [BLOCK_M, wmma_k], [0, wmma_k])
        b1 = gl.amd.slice(b, [wmma_k, BLOCK_N], [wmma_k, 0])
        acc = gl.amd.gfx1250.wmma(a0, b0, acc)
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc = issue_loads(
            a_desc, b_desc, a_buf.index(slot), b_buf.index(slot), BLOCK_K)
        acc = gl.amd.gfx1250.wmma(a1, b1, acc)
    return a_desc, b_desc, acc


@gluon.jit
def cluster_consume_and_refill_bk32(a_buf, b_buf, slot, a_desc, b_desc, acc,
                                    DOT_A: gl.constexpr, DOT_B: gl.constexpr,
                                    BLOCK_K: gl.constexpr):
    quarter_k: gl.constexpr = BLOCK_K // 4
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(0, quarter_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(0, quarter_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)

    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(quarter_k, quarter_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(quarter_k, quarter_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)

    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(2 * quarter_k, quarter_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(2 * quarter_k, quarter_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)

    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(3 * quarter_k, quarter_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(3 * quarter_k, quarter_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        # Keep the overwrite-safety ordering used by the BK64 schedule.
        gl.amd.gfx1250.cluster.arrive()
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc = issue_loads(
            a_desc, b_desc, a_buf.index(slot), b_buf.index(slot), BLOCK_K)
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    return a_desc, b_desc, acc


@gluon.jit
def cluster_consume_before_refill(a_buf, b_buf, slot, acc,
                                  DOT_A: gl.constexpr, DOT_B: gl.constexpr, BLOCK_K: gl.constexpr):
    half_k: gl.constexpr = BLOCK_K // 2
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        # The first refill pair completed before slot 0 was consumed. Drain
        # the second pair while slot-0 WMMA runs, then rendezvous so every
        # CTA's remote slot-1 writes are visible before reading LDS.
        tdm.async_wait(0)
        gl.amd.gfx1250.cluster.arrive()
        gl.amd.gfx1250.cluster.wait()
        a = a_buf.index(slot).slice(0, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(0, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)

    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(half_k, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(half_k, half_k, 1).permute([1, 0]).load(layout=DOT_B)
        gl.amd.gfx1250.cluster.arrive()
    return a, b, acc


@gluon.jit
def cluster_refill_pair_and_wmma(a_buf, b_buf, a_desc, b_desc, a, b, acc,
                                 BLOCK_K: gl.constexpr):
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc = issue_loads(
            a_desc, b_desc, a_buf.index(0), b_buf.index(0), BLOCK_K)
        a_desc, b_desc = issue_loads(
            a_desc, b_desc, a_buf.index(1), b_buf.index(1), BLOCK_K)
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    return a_desc, b_desc, acc


@gluon.jit
def bf16_kernelc_2pf_bk128_fused_gfx1250(
        a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
        stride_cm, stride_cn, GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr,
        GROUP_SIZE_M: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        BLOCK_K: gl.constexpr, NUM_WARPS: gl.constexpr, NUM_BUFFERS: gl.constexpr):
    gl.static_assert(a_ptr.type.element_ty.is_bf16() and b_ptr.type.element_ty.is_bf16())
    pid_m, pid_n = get_xcd_swizzled_pids(
        M, N, GRID_MN, NUM_XCDS, GROUP_SIZE_M, BLOCK_M, BLOCK_N)

    padded_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0])
    padded_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 8]], [BLOCK_N, BLOCK_K], [1, 0])
    layouts: gl.constexpr = gl.amd.gfx1250.make_partitioned_dot_layouts(
        BLOCK_M, BLOCK_N, padded_a, padded_b, NUM_WARPS, [16, 16, 32],
        a_transposed=False, b_transposed=True, transposed=True)
    shared_a: gl.constexpr = layouts[0]
    shared_b: gl.constexpr = layouts[1]
    wmma: gl.constexpr = layouts[2]
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma, 8)

    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, BLOCK_K), layout=shared_a)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, BLOCK_K), layout=shared_b)
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], shared_a)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [NUM_BUFFERS, BLOCK_N, BLOCK_K], shared_b)

    a_desc, b_desc = issue_loads(a_desc, b_desc, a_buf.index(0), b_buf.index(0), BLOCK_K)
    a_desc, b_desc = issue_loads(a_desc, b_desc, a_buf.index(1), b_buf.index(1), BLOCK_K)
    tdm.async_wait(2)

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=wmma)
    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max >= NUM_BUFFERS)

    for _ in range(0, (iter_max - NUM_BUFFERS) // NUM_BUFFERS):
        a_desc, b_desc, acc = consume_and_refill(
            a_buf, b_buf, 0, a_desc, b_desc, acc, dot_a, dot_b, BLOCK_K)
        tdm.async_wait(2)
        a_desc, b_desc, acc = consume_and_refill(
            a_buf, b_buf, 1, a_desc, b_desc, acc, dot_a, dot_b, BLOCK_K)
        tdm.async_wait(2)

    phase = ((iter_max - NUM_BUFFERS) // NUM_BUFFERS) * NUM_BUFFERS
    for _ in range(0, (iter_max - NUM_BUFFERS) % NUM_BUFFERS):
        slot = phase % NUM_BUFFERS
        a_desc, b_desc, acc = consume_and_refill(
            a_buf, b_buf, slot, a_desc, b_desc, acc, dot_a, dot_b, BLOCK_K)
        tdm.async_wait(2)
        phase += 1

    tdm.async_wait(0)
    for _ in gl.static_range(NUM_BUFFERS):
        acc = consume_slot(
            a_buf, b_buf, phase % NUM_BUFFERS, acc, dot_a, dot_b, BLOCK_K,
            False, False)
        phase += 1

    c_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N, 4]], [BLOCK_M, BLOCK_N], [1, 0])
    c_shared = gl.allocate_shared_memory(c_ptr.type.element_ty, [BLOCK_M, BLOCK_N], c_layout)
    c_shared.store(acc.to(c_ptr.type.element_ty))
    c_desc = tdm.make_tensor_descriptor(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        block_shape=(BLOCK_M, BLOCK_N), layout=c_layout)
    tdm.async_store(c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], c_shared)
    tdm.async_wait(0)


@gluon.jit
def bf16_kernelc_2pf_bk128_fused_cluster_4x4_gfx1250(
        a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
        stride_cm, stride_cn, GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr,
        GROUP_SIZE_M: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        BLOCK_K: gl.constexpr, NUM_WARPS: gl.constexpr, NUM_BUFFERS: gl.constexpr,
        SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr,
        WMMA_LAYOUT: gl.constexpr, SAME_SCHEDULE: gl.constexpr,
        SLICE_BK32: gl.constexpr):
    gl.static_assert(a_ptr.type.element_ty.is_bf16() and b_ptr.type.element_ty.is_bf16())
    gl.static_assert(BLOCK_M % 256 == 0 and BLOCK_N % 256 == 0 and BLOCK_K == 128)
    gl.static_assert(NUM_WARPS == 8 and NUM_BUFFERS == 2)
    gl.static_assert(
        gl.num_ctas() == (BLOCK_M // 256) * (BLOCK_N // 256),
        "cluster dimensions must match the logical KernelC tile")
    pid_m, pid_n = get_xcd_swizzled_pids(
        M, N, GRID_MN, NUM_XCDS, GROUP_SIZE_M, BLOCK_M, BLOCK_N)

    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 8)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, BLOCK_K),
        layout=SHARED_LAYOUT_B)
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [NUM_BUFFERS, BLOCK_N, BLOCK_K], SHARED_LAYOUT_B)

    a_desc, b_desc = issue_loads(a_desc, b_desc, a_buf.index(0), b_buf.index(0), BLOCK_K)
    a_desc, b_desc = issue_loads(a_desc, b_desc, a_buf.index(1), b_buf.index(1), BLOCK_K)
    tdm.async_wait(2)

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max >= NUM_BUFFERS)

    if SAME_SCHEDULE:
        for _ in range(0, (iter_max - NUM_BUFFERS) // NUM_BUFFERS):
            if SLICE_BK32:
                a_desc, b_desc, acc = cluster_consume_and_refill_bk32(
                    a_buf, b_buf, 0, a_desc, b_desc, acc,
                    dot_a, dot_b, BLOCK_K)
            else:
                a_desc, b_desc, acc = cluster_consume_and_refill(
                    a_buf, b_buf, 0, a_desc, b_desc, acc,
                    dot_a, dot_b, BLOCK_M, BLOCK_N, BLOCK_K)
            tdm.async_wait(2)
            if SLICE_BK32:
                a_desc, b_desc, acc = cluster_consume_and_refill_bk32(
                    a_buf, b_buf, 1, a_desc, b_desc, acc,
                    dot_a, dot_b, BLOCK_K)
            else:
                a_desc, b_desc, acc = cluster_consume_and_refill(
                    a_buf, b_buf, 1, a_desc, b_desc, acc,
                    dot_a, dot_b, BLOCK_M, BLOCK_N, BLOCK_K)
            tdm.async_wait(2)

        # Keep the tail in a loop so the warp-pipeline pass also transforms
        # the epilogue. The preceding wait(2) makes the penultimate slot ready
        # while the final slot is still in flight.
        for tail_idx in range(iter_max - NUM_BUFFERS, iter_max - 1):
            slot = tail_idx % NUM_BUFFERS
            acc = consume_slot(
                a_buf, b_buf, slot, acc, dot_a, dot_b, BLOCK_K, False, False)

            tdm.async_wait(0)
            slot = (tail_idx + 1) % NUM_BUFFERS
            acc = consume_slot(
                a_buf, b_buf, slot, acc, dot_a, dot_b, BLOCK_K, False, False)
    else:
        for _ in range(0, (iter_max - NUM_BUFFERS) // NUM_BUFFERS):
            acc = consume_slot(
                a_buf, b_buf, 0, acc, dot_a, dot_b, BLOCK_K, False, True)
            a, b, acc = cluster_consume_before_refill(
                a_buf, b_buf, 1, acc, dot_a, dot_b, BLOCK_K)
            a_desc, b_desc, acc = cluster_refill_pair_and_wmma(
                a_buf, b_buf, a_desc, b_desc, a, b, acc, BLOCK_K)
            tdm.async_wait(2)

        acc = consume_slot(
            a_buf, b_buf, 0, acc, dot_a, dot_b, BLOCK_K, False, True)
        acc = consume_slot(
            a_buf, b_buf, 1, acc, dot_a, dot_b, BLOCK_K, True, True)

    c_layout: gl.constexpr = gl.SwizzledSharedLayout(
        1, 1, 1, [1, 0], WMMA_LAYOUT.cga_layout)
    c_shared = gl.allocate_shared_memory(
        c_ptr.type.element_ty, [BLOCK_M, BLOCK_N], c_layout)
    c_shared.store(acc.to(c_ptr.type.element_ty))
    c_desc = tdm.make_tensor_descriptor(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        block_shape=(BLOCK_M, BLOCK_N), layout=c_layout)
    tdm.async_store(c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], c_shared)
    tdm.async_wait(0)


def build_cluster_layouts(block_m, block_n, block_k, num_warps,
                          cluster_ctas_m, cluster_ctas_n, use_partitioned,
                          partition_groups_a, partition_groups_b):
    cga_layout_c = make_cga_layout(
        [cluster_ctas_m, cluster_ctas_n],
        [cluster_ctas_m, cluster_ctas_n], [0, 1])
    slice_m = block_m // cluster_ctas_m
    slice_n = block_n // cluster_ctas_n
    padded_a_local = gl.PaddedSharedLayout.with_identity_for(
        [[block_k, 8]], [block_m, block_k], [1, 0])
    padded_b_local = gl.PaddedSharedLayout.with_identity_for(
        [[block_k, 8]], [block_n, block_k], [1, 0])
    local_layouts = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, block_n, padded_a_local, padded_b_local, num_warps,
        [16, 16, 32], a_transposed=False, b_transposed=True,
        slice_m=slice_m, slice_n=slice_n, transposed=True)
    local_wmma = local_layouts[2]
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga_layout_c)
    dot_a = gl.DotOperandLayout(0, wmma, 8)
    dot_b = gl.DotOperandLayout(1, wmma, 8)
    cga_layout_a = dot_a.cga_layout
    cga_layout_b = tuple(tuple([basis[1], basis[0]]) for basis in dot_b.cga_layout)
    shared_a = gl.PaddedSharedLayout.with_identity_for(
        [[block_k, 8]], [block_m, block_k], [1, 0], cga_layout_a)
    shared_b = gl.PaddedSharedLayout.with_identity_for(
        [[block_k, 8]], [block_n, block_k], [1, 0], cga_layout_b)
    if use_partitioned:
        # Partitioning is CTA-local. Each operand's logical piece count is two
        # physical partitions times its group count, independent of cluster
        # shape.
        partitioned_a, partitioned_b, _ = \
            gl.amd.gfx1250.make_partitioned_dot_layouts(
                partition_groups_a * slice_m, partition_groups_b * slice_n,
                shared_a, shared_b, num_warps,
                [16, 16, 32], a_transposed=False, b_transposed=True,
                slice_m=slice_m, slice_n=slice_n, transposed=True)
        shared_a = partitioned_a
        shared_b = partitioned_b
    return shared_a, shared_b, wmma


def make_inputs(args):
    if args.input_mode == "random":
        torch.manual_seed(42)
        a = torch.randn((args.M, args.K), dtype=torch.bfloat16)
        b = torch.randn((args.K, args.N), dtype=torch.bfloat16).T.contiguous()
        c = torch.zeros((args.M, args.N), dtype=torch.float32)
        return a.cuda(), b.cuda(), c.cuda()

    device = torch.device("cuda")
    a = torch.empty((args.M, args.K), dtype=torch.bfloat16, device=device)
    b = torch.empty((args.N, args.K), dtype=torch.bfloat16, device=device)
    chunk = 8 * 1024 * 1024
    for output, cosine in ((a, False), (b, True)):
        flat = output.view(-1)
        for begin in range(0, flat.numel(), chunk):
            end = min(begin + chunk, flat.numel())
            angle = torch.arange(begin, end, dtype=torch.float64, device=device)
            value = torch.cos(angle) if cosine else torch.sin(angle)
            flat[begin:end].copy_(value.float())
    c = torch.zeros((args.M, args.N), dtype=torch.float32, device=device)
    return a, b, c


def make_launch(args, a, b, c):
    is_cluster = args.cluster_4x4 or args.cluster_same_schedule
    block_m = BLOCK_M * args.cluster_ctas_m if is_cluster else BLOCK_M
    block_n = BLOCK_N * args.cluster_ctas_n if is_cluster else BLOCK_N
    grid = (triton.cdiv(args.M, block_m) * triton.cdiv(args.N, block_n), 1)
    cluster_layouts = None
    if is_cluster:
        partition_groups_a = (
            args.cluster_partition_groups_a or args.cluster_partition_groups)
        partition_groups_b = (
            args.cluster_partition_groups_b or args.cluster_partition_groups)
        cluster_layouts = build_cluster_layouts(
            block_m, block_n, BLOCK_K, NUM_WARPS,
            args.cluster_ctas_m, args.cluster_ctas_n,
            args.cluster_partitioned, partition_groups_a, partition_groups_b)

    def launch():
        if is_cluster:
            shared_a, shared_b, wmma = cluster_layouts
            return bf16_kernelc_2pf_bk128_fused_cluster_4x4_gfx1250[grid](
                a, b, c, args.M, args.N, args.K,
                a.stride(0), a.stride(1), b.stride(1), b.stride(0),
                c.stride(0), c.stride(1), GRID_MN=grid[0],
                NUM_XCDS=args.num_xcds, GROUP_SIZE_M=args.group_size_m,
                BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=BLOCK_K,
                NUM_WARPS=NUM_WARPS, NUM_BUFFERS=CLUSTER_NUM_BUFFERS,
                SHARED_LAYOUT_A=shared_a, SHARED_LAYOUT_B=shared_b,
                WMMA_LAYOUT=wmma, SAME_SCHEDULE=args.cluster_same_schedule,
                SLICE_BK32=args.cluster_bk32,
                num_warps=NUM_WARPS,
                waves_per_eu=NUM_WARPS // 4,
                num_ctas=args.cluster_ctas_m * args.cluster_ctas_n)
        return bf16_kernelc_2pf_bk128_fused_gfx1250[grid](
            a, b, c, args.M, args.N, args.K,
            a.stride(0), a.stride(1), b.stride(1), b.stride(0),
            c.stride(0), c.stride(1), GRID_MN=grid[0],
            NUM_XCDS=args.num_xcds, GROUP_SIZE_M=args.group_size_m,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            NUM_WARPS=NUM_WARPS, NUM_BUFFERS=NUM_BUFFERS,
            num_warps=NUM_WARPS, waves_per_eu=NUM_WARPS // 4)

    return launch


def check_result(launch, a, b, c):
    c.zero_()
    launch()
    torch.cuda.synchronize()
    reference = a.cpu().float() @ b.cpu().T.float()
    torch.testing.assert_close(c.cpu(), reference, rtol=1e-3, atol=1e-2)
    print("result verified", flush=True)


def event_probe(launch, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        launch()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def capture_graph(launch, iters):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(iters):
                launch()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return graph


def benchmark(launch, args):
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    probe_ms = event_probe(launch, args.probe_iters)
    graph = capture_graph(launch, args.iters_per_graph)
    total_iters = args.iters_per_graph * args.n_replays

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.n_replays):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    per_iter = elapsed / total_iters

    print(f"probe per-iter   : {probe_ms * 1e3:.2f} us")
    print(f"iters per graph  : {args.iters_per_graph}")
    print(f"replays          : {args.n_replays}")
    print(f"total iters      : {total_iters}")
    print()
    print(f"total elapsed    : {elapsed:.6f} s")
    print(f"per-iter         : {per_iter * 1e6:.2f} us")
    print(f"TFLOPS           : {(2 * args.M * args.N * args.K) / per_iter / 1e12:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Isolated fused-TDM BF16 KernelC for gfx1250")
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-K", type=int, default=65536)
    parser.add_argument("--group-size-m", type=int, default=4)
    parser.add_argument("--num-xcds", type=int, default=8)
    parser.add_argument("--input-mode", choices=["random", "trig"], default="trig")
    parser.add_argument("--cluster-4x4", action="store_true",
                        help="distribute each 1024x1024 logical tile across a 4x4 CTA cluster")
    parser.add_argument("--cluster-same-schedule", action="store_true",
                        help="use the standalone one-slot refill schedule with a 4x4 CTA cluster")
    parser.add_argument("--cluster-bk32", action="store_true",
                        help="split each clustered BK128 slot into four BK32 WMMA stages")
    parser.add_argument("--cluster-ctas-m", type=int, choices=[2, 4], default=4)
    parser.add_argument("--cluster-ctas-n", type=int, choices=[2, 4], default=4)
    parser.add_argument("--cluster-partitioned", action="store_true",
                        help="partition both operands into CTA-local logical pieces")
    parser.add_argument("--cluster-partition-groups", type=int, choices=[1, 2], default=1,
                        help="groups per physical partition (default: 1)")
    parser.add_argument("--cluster-partition-groups-a", type=int, choices=[1, 2],
                        help="override partition groups for A")
    parser.add_argument("--cluster-partition-groups-b", type=int, choices=[1, 2],
                        help="override partition groups for B")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--probe-iters", type=int, default=10)
    parser.add_argument("--iters-per-graph", type=int, default=10)
    parser.add_argument("--n-replays", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cluster_4x4 and args.cluster_same_schedule:
        raise ValueError("select only one cluster schedule")
    is_cluster = args.cluster_4x4 or args.cluster_same_schedule
    if args.cluster_partitioned and not is_cluster:
        raise ValueError("--cluster-partitioned requires a cluster schedule")
    if args.cluster_bk32 and not args.cluster_same_schedule:
        raise ValueError("--cluster-bk32 requires --cluster-same-schedule")
    block_m = BLOCK_M * args.cluster_ctas_m if is_cluster else BLOCK_M
    block_n = BLOCK_N * args.cluster_ctas_n if is_cluster else BLOCK_N
    if args.M % block_m or args.N % block_n or args.K % BLOCK_K:
        raise ValueError(
            f"M, N, and K must be divisible by {block_m}, {block_n}, and {BLOCK_K}")
    if triton.cdiv(args.K, BLOCK_K) < NUM_BUFFERS:
        raise ValueError("K must contain at least two BK128 tiles")
    if is_cluster and triton.cdiv(args.K, BLOCK_K) % NUM_BUFFERS:
        raise ValueError("the 4x4 cluster schedule requires an even number of BK128 tiles")
    if not args.check and not args.benchmark:
        raise ValueError("select --check and/or --benchmark")

    print(args, flush=True)
    a, b, c = make_inputs(args)
    launch = make_launch(args, a, b, c)
    if args.check:
        check_result(launch, a, b, c)
    if args.benchmark:
        benchmark(launch, args)


if __name__ == "__main__":
    main()
