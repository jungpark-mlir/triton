import time
import torch

import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

try:
    import pytest
except ModuleNotFoundError:

    class _PytestMarkStub:

        @staticmethod
        def parametrize(*_args, **_kwargs):

            def decorator(fn):
                return fn

            return decorator

    class _PytestStub:
        mark = _PytestMarkStub()

        @staticmethod
        def skip(reason):
            raise RuntimeError(reason)

    pytest = _PytestStub()

# Handle imports for both pytest (module context) and direct execution
try:
    from .gfx1250_utils import static_profile
    from .f16_gemm_common_gfx1250 import (
        create_shared_layouts,
        create_tensor_descriptors,
        issue_loads,
        issue_wmma,
        lds_load,
        issue_wmma_compute,
        TileScheduler,
    )
except ImportError:
    from gfx1250_utils import static_profile
    from f16_gemm_common_gfx1250 import (
        create_shared_layouts,
        create_tensor_descriptors,
        issue_loads,
        issue_wmma,
        lds_load,
        issue_wmma_compute,
        TileScheduler,
    )


@gluon.jit
def get_xcd_swizzled_pids(M, N, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, GRID_MN: ttgl.constexpr,
                          NUM_XCDS: ttgl.constexpr, GROUP_SIZE_M: ttgl.constexpr):
    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    num_pid_n = ttgl.cdiv(N, BLOCK_N)

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
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n


@gluon.jit
def gemm_tdm_pipelined_warp_pipelined_kernel(a_ptr, b_ptr, c_ptr,  #
                                             M, N, K,  #
                                             stride_am, stride_ak,  #
                                             stride_bk, stride_bn,  #
                                             stride_cm, stride_cn,  #
                                             BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                             BLOCK_K: ttgl.constexpr,  #
                                             NUM_BUFFERS: ttgl.constexpr,  #
                                             TRANSPOSE_B: ttgl.constexpr,  #
                                             WARP_BASES: ttgl.constexpr):
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])

    shared_layouts: ttgl.constexpr = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    SHARED_LAYOUT_A: ttgl.constexpr = shared_layouts[0]
    SHARED_LAYOUT_B: ttgl.constexpr = shared_layouts[1]
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    a_desc, b_desc = create_tensor_descriptors(a_ptr, b_ptr, pid_m * BLOCK_M * stride_am, pid_n * BLOCK_N * stride_bn,
                                               stride_am, stride_ak, stride_bn, stride_bk, SHARED_LAYOUT_A,
                                               SHARED_LAYOUT_B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    producer = 0
    consumer = 0
    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

    # Prefetch NUM_BUFFERS - 1 tiles; the main loop produces one tile for
    # each tile it consumes, and the epilogue drains the prefetched tail.
    for _ in ttgl.static_range(NUM_BUFFERS - 1):
        producer = issue_loads(producer, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B)

    # Wait for the first prefetch
    ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)
    for _ in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
        with ttgl.amd.warp_pipeline_stage("stage0", priority=1):
            consumer, a, b = lds_load(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B, NUM_BUFFERS,
                                      TRANSPOSE_B)
        # Wait for the last one, either before the loop or a load below.
        ttgl.amd.gfx1250.tdm.async_wait(0)
        with ttgl.amd.warp_pipeline_stage("stage1", priority=0):
            producer = issue_loads(producer, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                   TRANSPOSE_B)
            accumulator = issue_wmma_compute(a, b, accumulator)

    for i in ttgl.static_range(NUM_BUFFERS - 1):
        with ttgl.amd.warp_pipeline_stage("stage0_epilogue", priority=1):
            consumer, a, b = lds_load(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B, NUM_BUFFERS,
                                      TRANSPOSE_B)
        ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1 - i) * 2)
        with ttgl.amd.warp_pipeline_stage("stage1_epilogue", priority=0):
            accumulator = issue_wmma_compute(a, b, accumulator)

    offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.amd.gfx1250.buffer_store(accumulator, c_ptr, offs_c, mask=mask_c)


# ---------------------------------------------------------------------------
# Partial TDM copy variant: only a subset of warps issue TDM copies.
# Cleared warps get pred=0 (hardware no-op), freeing TDM bandwidth.
# ---------------------------------------------------------------------------


@gluon.jit
def issue_loads_predicated(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K: ttgl.constexpr,
                           NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr,
                           TDM_WARP_USED_HINT: ttgl.constexpr):
    ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_am, producer * BLOCK_K], a_buffer.index(producer % NUM_BUFFERS),
                                    warp_used_hint=TDM_WARP_USED_HINT)
    if not TRANSPOSE_B:
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [producer * BLOCK_K, off_bn], b_buffer.index(producer % NUM_BUFFERS),
                                        warp_used_hint=TDM_WARP_USED_HINT)
    else:
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_bn, producer * BLOCK_K], b_buffer.index(producer % NUM_BUFFERS),
                                        warp_used_hint=TDM_WARP_USED_HINT)
    producer += 1
    return producer


@gluon.jit
def gemm_tdm_predicated_pipelined_warp_pipelined_kernel(a_ptr, b_ptr, c_ptr,  #
                                                        M, N, K,  #
                                                        stride_am, stride_ak,  #
                                                        stride_bk, stride_bn,  #
                                                        stride_cm, stride_cn,  #
                                                        BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                                        BLOCK_K: ttgl.constexpr,  #
                                                        NUM_BUFFERS: ttgl.constexpr,  #
                                                        TRANSPOSE_B: ttgl.constexpr,  #
                                                        WARP_BASES: ttgl.constexpr,  #
                                                        TDM_WARP_USED_HINT: ttgl.constexpr):
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])

    shared_layouts: ttgl.constexpr = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    SHARED_LAYOUT_A: ttgl.constexpr = shared_layouts[0]
    SHARED_LAYOUT_B: ttgl.constexpr = shared_layouts[1]
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    a_desc, b_desc = create_tensor_descriptors(a_ptr, b_ptr, pid_m * BLOCK_M * stride_am, pid_n * BLOCK_N * stride_bn,
                                               stride_am, stride_ak, stride_bn, stride_bk, SHARED_LAYOUT_A,
                                               SHARED_LAYOUT_B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    producer = 0
    consumer = 0
    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

    for _ in ttgl.static_range(2):
        producer = issue_loads_predicated(producer, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                          TRANSPOSE_B, TDM_WARP_USED_HINT)

    ttgl.amd.gfx1250.tdm.async_wait(1 * 2)
    for _ in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
        with ttgl.amd.warp_pipeline_stage("stage0", priority=1):
            consumer, a, b = lds_load(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B, NUM_BUFFERS,
                                      TRANSPOSE_B)
            producer = issue_loads_predicated(producer, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                              TRANSPOSE_B, TDM_WARP_USED_HINT)
        with ttgl.amd.warp_pipeline_stage("stage1", priority=0):
            accumulator = issue_wmma_compute(a, b, accumulator)
        ttgl.amd.gfx1250.tdm.async_wait(2)

    for i in ttgl.static_range(NUM_BUFFERS - 1):
        ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1 - i) * 2)
        consumer, accumulator = issue_wmma(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                           accumulator, (NUM_BUFFERS - 2 - i) * 2, NUM_BUFFERS, TRANSPOSE_B)

    offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.amd.gfx1250.buffer_store(accumulator, c_ptr, offs_c, mask=mask_c)


# ---------------------------------------------------------------------------
# KernelB variant: loads+TDM in stage0, dot-only in stage1 (num_buffers>=3).
#
# Schedule (num_buffers=4):
#          w0      w1               w2             w3
# g0 g1 g2 / l0 g3 / d0    / l1 g4 / d1    / l2 g5 / d2    / l3
# g0 g1 g2         / l0 g3 / d0    / l1 g4 / d1    / l2 g5 / d2    /
#          w0              w1              w2              w3
#
# The write index is derived from `phase` (the read counter) so that the write
# and the read share the same SSA base, letting the membar dynamic-index
# disjointness analysis prove the two slots never overlap within an iteration.
# ---------------------------------------------------------------------------


@gluon.jit
def gemm_tdm_pipelined_warp_pipelined_kernelB(a_ptr, b_ptr, c_ptr,  #
                                              M, N, K,  #
                                              stride_am, stride_ak,  #
                                              stride_bk, stride_bn,  #
                                              stride_cm, stride_cn,  #
                                              BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                              BLOCK_K: ttgl.constexpr,  #
                                              NUM_BUFFERS: ttgl.constexpr,  #
                                              TRANSPOSE_B: ttgl.constexpr,  #
                                              WARP_BASES: ttgl.constexpr,  #
                                              GRID_MN: ttgl.constexpr, NUM_XCDS: ttgl.constexpr,
                                              GROUP_SIZE_M: ttgl.constexpr,  #
                                              USE_TDM_STORE: ttgl.constexpr,  #
                                              PRIO_SWAP: ttgl.constexpr = False):
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 3, "kernelB requires NUM_BUFFERS >= 3")

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])

    shared_layouts: ttgl.constexpr = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    SHARED_LAYOUT_A: ttgl.constexpr = shared_layouts[0]
    SHARED_LAYOUT_B: ttgl.constexpr = shared_layouts[1]
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    a_desc, b_desc = create_tensor_descriptors(a_ptr, b_ptr, pid_m * BLOCK_M * stride_am, pid_n * BLOCK_N * stride_bn,
                                               stride_am, stride_ak, stride_bn, stride_bk, SHARED_LAYOUT_A,
                                               SHARED_LAYOUT_B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    phase = 0
    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

    producer = 0
    for _ in ttgl.static_range(NUM_BUFFERS - 1):
        producer = issue_loads(producer, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B)

    ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
    _p0: ttgl.constexpr = 1 if PRIO_SWAP else 0
    _p1: ttgl.constexpr = 0 if PRIO_SWAP else 1

    for _ in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
        with ttgl.amd.warp_pipeline_stage("stage0", priority=_p0):
            write_phase = phase + (NUM_BUFFERS - 1)
            phase, a, b = lds_load(phase, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B, NUM_BUFFERS,
                                   TRANSPOSE_B)
            issue_loads(write_phase, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B)
        ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
        with ttgl.amd.warp_pipeline_stage("stage1", priority=_p1):
            accumulator = issue_wmma_compute(a, b, accumulator)

    for i in ttgl.static_range(NUM_BUFFERS - 1):
        with ttgl.amd.warp_pipeline_stage("stage0", priority=_p0):
            phase, a, b = lds_load(phase, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B, NUM_BUFFERS,
                                   TRANSPOSE_B)
        ttgl.amd.gfx1250.tdm.async_wait(0)
        with ttgl.amd.warp_pipeline_stage("stage1", priority=_p1):
            accumulator = issue_wmma_compute(a, b, accumulator)

    if USE_TDM_STORE:
        C_SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 4]], [BLOCK_M, BLOCK_N],
                                                                                    [1, 0])
        c_shared = ttgl.allocate_shared_memory(c_ptr.type.element_ty, shape=[BLOCK_M, BLOCK_N], layout=C_SHARED_LAYOUT)
        c_shared.store(accumulator.to(c_ptr.type.element_ty))
        c_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                                             block_shape=(BLOCK_M, BLOCK_N), layout=C_SHARED_LAYOUT)
        ttgl.amd.gfx1250.tdm.async_store(c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], c_shared)
        ttgl.amd.gfx1250.tdm.async_wait(0)
    else:
        offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
        offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
        offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        ttgl.amd.gfx1250.buffer_store(accumulator, c_ptr, offs_c, mask=mask_c)


# ---------------------------------------------------------------------------
# Persistent kernelB variant: fixed resident grid, each workgroup iterates over
# MN tiles assigned by the local TileScheduler.
# ---------------------------------------------------------------------------


@gluon.jit
def persistent_gemm_tdm_pipelined_warp_pipelined_kernelB(a_ptr, b_ptr, c_ptr,  #
                                                         M, N, K,  #
                                                         stride_am, stride_ak,  #
                                                         stride_bk, stride_bn,  #
                                                         stride_cm, stride_cn,  #
                                                         BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                                         BLOCK_K: ttgl.constexpr,  #
                                                         NUM_BUFFERS: ttgl.constexpr,  #
                                                         TRANSPOSE_B: ttgl.constexpr,  #
                                                         WARP_BASES: ttgl.constexpr,  #
                                                         NUM_XCDS: ttgl.constexpr, GROUP_SIZE_M: ttgl.constexpr,  #
                                                         USE_TDM_STORE: ttgl.constexpr,  #
                                                         PRIO_SWAP: ttgl.constexpr = False):
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 3, "persistent kernelB requires NUM_BUFFERS >= 3")

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])

    shared_layouts: ttgl.constexpr = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    SHARED_LAYOUT_A: ttgl.constexpr = shared_layouts[0]
    SHARED_LAYOUT_B: ttgl.constexpr = shared_layouts[1]
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    a_desc, b_desc = create_tensor_descriptors(a_ptr, b_ptr, 0, 0, stride_am, stride_ak, stride_bn, stride_bk,
                                               SHARED_LAYOUT_A, SHARED_LAYOUT_B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                               TRANSPOSE_B)

    scheduler = TileScheduler.initialize(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, STREAMK_TILES=0)
    pid = scheduler.get_pid()
    num_sms = scheduler.get_num_sms()
    pid = scheduler.apply_chiplet_transform_chunked(pid, num_sms, num_xcds=NUM_XCDS, chunk_size=2)

    for tile_idx in range(pid, scheduler.get_num_tiles(), num_sms):
        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_idx, GROUP_SIZE_M=GROUP_SIZE_M)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N

        a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape,
                                               layout=a_desc.layout)
        b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape,
                                               layout=b_desc.layout)
        phase = 0
        accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

        producer = 0
        for _ in ttgl.static_range(NUM_BUFFERS - 1):
            producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                   TRANSPOSE_B)

        ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
        _p0: ttgl.constexpr = 1 if PRIO_SWAP else 0
        _p1: ttgl.constexpr = 0 if PRIO_SWAP else 1

        for _ in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
            with ttgl.amd.warp_pipeline_stage("stage0", priority=_p0):
                write_phase = phase + (NUM_BUFFERS - 1)
                phase, a, b = lds_load(phase, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B, NUM_BUFFERS,
                                       TRANSPOSE_B)
                issue_loads(write_phase, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                            TRANSPOSE_B)
            ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
            with ttgl.amd.warp_pipeline_stage("stage1", priority=_p1):
                accumulator = issue_wmma_compute(a, b, accumulator)

        for _ in range(0, NUM_BUFFERS - 1):
            with ttgl.amd.warp_pipeline_stage("stage0_epilogue", priority=_p0):
                phase, a, b = lds_load(phase, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B, NUM_BUFFERS,
                                       TRANSPOSE_B)
            ttgl.amd.gfx1250.tdm.async_wait(0)
            with ttgl.amd.warp_pipeline_stage("stage1_epilogue", priority=_p1):
                accumulator = issue_wmma_compute(a, b, accumulator)

        a_buffer._keep_alive()
        b_buffer._keep_alive()
        if USE_TDM_STORE:
            C_SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 4]],
                                                                                        [BLOCK_M, BLOCK_N], [1, 0])
            c_shared = ttgl.allocate_shared_memory(c_ptr.type.element_ty, shape=[BLOCK_M, BLOCK_N],
                                                   layout=C_SHARED_LAYOUT)
            c_shared.store(accumulator.to(c_ptr.type.element_ty))
            c_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=c_ptr, shape=(M, N),
                                                                 strides=(stride_cm, stride_cn),
                                                                 block_shape=(BLOCK_M, BLOCK_N), layout=C_SHARED_LAYOUT)
            ttgl.amd.gfx1250.tdm.async_store(c_desc, [off_am, off_bn], c_shared)
            ttgl.amd.gfx1250.tdm.async_wait(0)
            c_shared._keep_alive()
        else:
            offs_cm = off_am + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
            offs_cn = off_bn + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
            offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            ttgl.amd.gfx1250.buffer_store(accumulator, c_ptr, offs_c, mask=mask_c)
        ttgl.amd.gfx1250.tdm.async_wait(0)


# ---------------------------------------------------------------------------
# KernelC variant: kernelB schedule with TDM loads issued by a warp subset.
# The warp_used_hint only changes which warps issue the global-to-LDS copy;
# all warps still consume LDS and participate in WMMA.
# ---------------------------------------------------------------------------


@gluon.jit
def gemm_tdm_pipelined_warp_pipelined_kernelC(a_ptr, b_ptr, c_ptr,  #
                                              M, N, K,  #
                                              stride_am, stride_ak,  #
                                              stride_bk, stride_bn,  #
                                              stride_cm, stride_cn,  #
                                              BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                              BLOCK_K: ttgl.constexpr,  #
                                              NUM_BUFFERS: ttgl.constexpr,  #
                                              TRANSPOSE_B: ttgl.constexpr,  #
                                              WARP_BASES: ttgl.constexpr,  #
                                              GRID_MN: ttgl.constexpr, NUM_XCDS: ttgl.constexpr,
                                              GROUP_SIZE_M: ttgl.constexpr,  #
                                              USE_TDM_STORE: ttgl.constexpr,  #
                                              TDM_WARP_USED_HINT: ttgl.constexpr,  #
                                              PRIO_SWAP: ttgl.constexpr = False):
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 3, "kernelC requires NUM_BUFFERS >= 3")

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])

    shared_layouts: ttgl.constexpr = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    SHARED_LAYOUT_A: ttgl.constexpr = shared_layouts[0]
    SHARED_LAYOUT_B: ttgl.constexpr = shared_layouts[1]
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    pid_m, pid_n = get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    a_desc, b_desc = create_tensor_descriptors(a_ptr, b_ptr, pid_m * BLOCK_M * stride_am, pid_n * BLOCK_N * stride_bn,
                                               stride_am, stride_ak, stride_bn, stride_bk, SHARED_LAYOUT_A,
                                               SHARED_LAYOUT_B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    phase = 0
    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

    producer = 0
    for _ in ttgl.static_range(NUM_BUFFERS - 1):
        producer = issue_loads_predicated(producer, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                          TRANSPOSE_B, TDM_WARP_USED_HINT)

    ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
    _p0: ttgl.constexpr = 1 if PRIO_SWAP else 0
    _p1: ttgl.constexpr = 0 if PRIO_SWAP else 1

    for _ in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
        with ttgl.amd.warp_pipeline_stage("stage0", priority=_p0):
            write_phase = phase + (NUM_BUFFERS - 1)
            phase, a, b = lds_load(phase, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B, NUM_BUFFERS,
                                   TRANSPOSE_B)
            issue_loads_predicated(write_phase, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                   TRANSPOSE_B, TDM_WARP_USED_HINT)

        with ttgl.amd.warp_pipeline_stage("stage1", priority=_p1):
            accumulator = issue_wmma_compute(a, b, accumulator)
        ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)

    for i in ttgl.static_range(NUM_BUFFERS - 1):
        with ttgl.amd.warp_pipeline_stage("stage0", priority=_p0):
            phase, a, b = lds_load(phase, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B, NUM_BUFFERS,
                                   TRANSPOSE_B)
        with ttgl.amd.warp_pipeline_stage("stage1", priority=_p1):
            accumulator = issue_wmma_compute(a, b, accumulator)
        ttgl.amd.gfx1250.tdm.async_wait(0)

    if USE_TDM_STORE:
        C_SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 4]], [BLOCK_M, BLOCK_N],
                                                                                    [1, 0])
        c_shared = ttgl.allocate_shared_memory(c_ptr.type.element_ty, shape=[BLOCK_M, BLOCK_N], layout=C_SHARED_LAYOUT)
        c_shared.store(accumulator.to(c_ptr.type.element_ty))
        c_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                                             block_shape=(BLOCK_M, BLOCK_N), layout=C_SHARED_LAYOUT)
        ttgl.amd.gfx1250.tdm.async_store(c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], c_shared)
        ttgl.amd.gfx1250.tdm.async_wait(0)
    else:
        offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
        offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
        offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        ttgl.amd.gfx1250.buffer_store(accumulator, c_ptr, offs_c, mask=mask_c)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_gemm_inputs(M, N, K, TRANSPOSE_B, INPUT_DTYPE=torch.float16):
    a = torch.randn((M, K), dtype=INPUT_DTYPE)
    b = torch.randn((K, N), dtype=INPUT_DTYPE)
    if TRANSPOSE_B:
        b = b.T.contiguous()
    c = torch.zeros((M, N), dtype=torch.float32)
    return a.cuda(), b.cuda(), c.cuda()


def _launch_gemm(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, a_device, b_device, c_device,
                 USE_KERNEL_B, USE_KERNEL_C=False, USE_PERSISTENT_KERNEL_B=False, GROUP_SIZE_M=4, USE_TDM_STORE=False,
                 NUM_WGPS=256):
    stride_am, stride_ak = a_device.stride(0), a_device.stride(1)
    stride_bk, stride_bn = (b_device.stride(0), b_device.stride(1)) if not TRANSPOSE_B else (b_device.stride(1),
                                                                                             b_device.stride(0))
    stride_cm, stride_cn = c_device.stride(0), c_device.stride(1)

    NUM_WARPS = 8
    WARP_BASES = ((0, 1), (1, 0), (2, 0))
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    if USE_PERSISTENT_KERNEL_B:
        persistent_grid = (min(NUM_WGPS, grid[0]), 1)
        return persistent_gemm_tdm_pipelined_warp_pipelined_kernelB[persistent_grid](
            a_device, b_device, c_device,  #
            M, N, K,  #
            stride_am, stride_ak,  #
            stride_bk, stride_bn,  #
            stride_cm, stride_cn,  #
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
            NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B, WARP_BASES=WARP_BASES,  #
            NUM_XCDS=8, GROUP_SIZE_M=GROUP_SIZE_M,  #
            USE_TDM_STORE=USE_TDM_STORE,  #
            num_warps=NUM_WARPS, waves_per_eu=NUM_WARPS // 4)
    if USE_KERNEL_B:
        GRID_MN = grid[0]
        return gemm_tdm_pipelined_warp_pipelined_kernelB[grid](
            a_device, b_device, c_device,  #
            M, N, K,  #
            stride_am, stride_ak,  #
            stride_bk, stride_bn,  #
            stride_cm, stride_cn,  #
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
            NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B, WARP_BASES=WARP_BASES,  #
            GRID_MN=GRID_MN, NUM_XCDS=8, GROUP_SIZE_M=GROUP_SIZE_M,  #
            USE_TDM_STORE=USE_TDM_STORE,  #
            num_warps=NUM_WARPS, waves_per_eu=NUM_WARPS // 4)
    if USE_KERNEL_C:
        GRID_MN = grid[0]
        TDM_WARP_USED_HINT = 0b00001111
        return gemm_tdm_pipelined_warp_pipelined_kernelC[grid](
            a_device, b_device, c_device,  #
            M, N, K,  #
            stride_am, stride_ak,  #
            stride_bk, stride_bn,  #
            stride_cm, stride_cn,  #
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
            NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B, WARP_BASES=WARP_BASES,  #
            GRID_MN=GRID_MN, NUM_XCDS=8, GROUP_SIZE_M=GROUP_SIZE_M,  #
            USE_TDM_STORE=USE_TDM_STORE, TDM_WARP_USED_HINT=TDM_WARP_USED_HINT,  #
            num_warps=NUM_WARPS, waves_per_eu=NUM_WARPS // 4)
    return gemm_tdm_pipelined_warp_pipelined_kernel[grid](
        a_device, b_device, c_device,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
        NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B, WARP_BASES=WARP_BASES,  #
        num_warps=NUM_WARPS, waves_per_eu=NUM_WARPS // 4)


def _launch_gemm_predicated(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, a_device, b_device, c_device):
    stride_am, stride_ak = a_device.stride(0), a_device.stride(1)
    stride_bk, stride_bn = (b_device.stride(0), b_device.stride(1)) if not TRANSPOSE_B else (b_device.stride(1),
                                                                                             b_device.stride(0))
    stride_cm, stride_cn = c_device.stride(0), c_device.stride(1)

    NUM_WARPS = 8
    WARP_BASES = ((0, 1), (1, 0), (2, 0))
    TDM_WARP_USED_HINT = 0b00001111
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    return gemm_tdm_predicated_pipelined_warp_pipelined_kernel[grid](
        a_device, b_device, c_device,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
        NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B, WARP_BASES=WARP_BASES,  #
        TDM_WARP_USED_HINT=TDM_WARP_USED_HINT, num_warps=NUM_WARPS, waves_per_eu=NUM_WARPS // 4)


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


def _benchmark_gemm(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, USE_KERNEL_B, USE_KERNEL_C,
                    USE_PERSISTENT_KERNEL_B, FOUR_WARP_TDM, GROUP_SIZE_M, USE_TDM_STORE, INPUT_DTYPE, NUM_WGPS, warmup,
                    probe_iters, graph_ms, n_replays, iters_per_graph):
    if triton.cdiv(K, BLOCK_K) < NUM_BUFFERS:
        raise ValueError("K/BLOCK_K must be at least NUM_BUFFERS")
    if sum((USE_KERNEL_B, USE_KERNEL_C, USE_PERSISTENT_KERNEL_B)) > 1:
        raise ValueError("--kernelB, --kernelC, and --persistent-kernelB are mutually exclusive")
    if (USE_KERNEL_C or USE_PERSISTENT_KERNEL_B) and FOUR_WARP_TDM:
        raise ValueError("--kernelC/--persistent-kernelB and --4warp-tdm are mutually exclusive")
    if (USE_KERNEL_B or USE_KERNEL_C or USE_PERSISTENT_KERNEL_B) and NUM_BUFFERS < 3:
        raise ValueError("kernelB/kernelC/persistent-kernelB require NUM_BUFFERS >= 3")
    if USE_TDM_STORE and (not (USE_KERNEL_B or USE_KERNEL_C or USE_PERSISTENT_KERNEL_B) or FOUR_WARP_TDM):
        raise ValueError("--tdmstore is currently wired only for --kernelB/--kernelC/--persistent-kernelB")
    if NUM_WGPS <= 0:
        raise ValueError("--num-wgps must be positive")

    torch.manual_seed(42)
    a_device, b_device, c_device = _make_gemm_inputs(M, N, K, TRANSPOSE_B, INPUT_DTYPE)

    def launch():
        if FOUR_WARP_TDM:
            return _launch_gemm_predicated(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, a_device,
                                           b_device, c_device)
        return _launch_gemm(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, a_device, b_device, c_device,
                            USE_KERNEL_B, USE_KERNEL_C, USE_PERSISTENT_KERNEL_B, GROUP_SIZE_M, USE_TDM_STORE, NUM_WGPS)

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
    flops = 2 * M * N * K
    tflops = flops / per_iter_s / 1e12

    print(f"probe per-iter   : {probe_ms * 1e3:.2f} us")
    print(f"iters per graph  : {n_per_graph}")
    print(f"replays          : {n_replays}")
    print(f"total iters      : {total_iters}")
    print()
    print(f"total elapsed    : {elapsed_s:.6f} s")
    print(f"per-iter         : {per_iter_s * 1e6:.2f} us")
    print(f"TFLOPS           : {tflops:.3f}")


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(256, 256, 64)])
@pytest.mark.parametrize("NUM_BUFFERS", [3])
@pytest.mark.parametrize("TRANSPOSE_B", [True])
@pytest.mark.parametrize("M,N,K", [(2048, 2048, 2048)])
@pytest.mark.parametrize("DUMP", [False])
@pytest.mark.parametrize("USE_KERNEL_B", [False])
def test_runtime_gemm_tdm_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, DUMP, USE_KERNEL_B,
                                    USE_KERNEL_C=False, USE_PERSISTENT_KERNEL_B=False, GROUP_SIZE_M=4,
                                    USE_TDM_STORE=False, INPUT_DTYPE=torch.float16, CHECK=False, NUM_WGPS=256):
    if triton.cdiv(K, BLOCK_K) < NUM_BUFFERS:
        pytest.skip("Skip tests where K/BLOCK_K < NUM_BUFFERS")

    torch.manual_seed(42)

    a_device, b_device, c_device = _make_gemm_inputs(M, N, K, TRANSPOSE_B, INPUT_DTYPE)
    kernel = _launch_gemm(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, a_device, b_device, c_device,
                          USE_KERNEL_B, USE_KERNEL_C, USE_PERSISTENT_KERNEL_B, GROUP_SIZE_M, USE_TDM_STORE, NUM_WGPS)
    static_profile(kernel)

    c_triton = c_device.cpu()
    c_torch = a_device.cpu().to(
        torch.float32) @ (b_device.cpu().to(torch.float32) if not TRANSPOSE_B else b_device.cpu().T.to(torch.float32))
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-4, atol=1e-4)
    if CHECK:
        print("result verified", flush=True)
    if DUMP:
        print("triton")
        print(c_triton)
        print("torch")
        print(c_torch)
        print("Done.")


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(256, 256, 64)])
@pytest.mark.parametrize("NUM_BUFFERS", [3])
@pytest.mark.parametrize("TRANSPOSE_B", [True])
@pytest.mark.parametrize("M,N,K", [(2048, 2048, 2048)])
@pytest.mark.parametrize("DUMP", [False])
def test_runtime_gemm_tdm_predicated_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, DUMP,
                                               INPUT_DTYPE=torch.float16, CHECK=False):
    if triton.cdiv(K, BLOCK_K) < NUM_BUFFERS:
        pytest.skip("Skip tests where K/BLOCK_K < NUM_BUFFERS")

    torch.manual_seed(42)

    a_device, b_device, c_device = _make_gemm_inputs(M, N, K, TRANSPOSE_B, INPUT_DTYPE)
    kernel = _launch_gemm_predicated(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, a_device, b_device,
                                     c_device)
    static_profile(kernel)

    c_triton = c_device.cpu()
    c_torch = a_device.cpu().to(
        torch.float32) @ (b_device.cpu().to(torch.float32) if not TRANSPOSE_B else b_device.cpu().T.to(torch.float32))
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-4, atol=1e-4)
    if CHECK:
        print("result verified", flush=True)
    if DUMP:
        print("triton")
        print(c_triton)
        print("torch")
        print(c_torch)
        print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=256, help='problem M size')
    parser.add_argument("-N", type=int, default=256, help='problem N size')
    parser.add_argument("-K", type=int, default=1024, help='problem K size')
    parser.add_argument("--num-buffers", type=int, choices=[2, 3, 4], default=3, help='num shared memory buffers')
    parser.add_argument("--4warp-tdm", action="store_true", dest="four_warp_tdm",
                        help="Use 4-warp partial TDM copy (warps 4-7 skip TDM copies)")
    parser.add_argument("--dump", action="store_true", help="Print out result/golden tensors")
    parser.add_argument("--kernelB", action="store_true", help="Use the kernelB variant")
    parser.add_argument("--kernelC", action="store_true",
                        help="Use kernelC: kernelB schedule with partial TDM load issue")
    parser.add_argument("--persistent-kernelB", action="store_true", dest="persistent_kernel_b",
                        help="Use persistent kernelB variant")
    parser.add_argument("--num-wgps", type=int, default=256, help="Resident workgroups for persistent kernelB")
    parser.add_argument("--group-size-m", type=int, default=4,
                        help="GROUP_SIZE_M for kernelB/kernelC/persistent-kernelB tile swizzle")
    parser.add_argument("--tdmstore", action="store_true",
                        help="Use kernelB/kernelC/persistent-kernelB TDM async store epilogue")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 inputs instead of FP16")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--benchmark", action="store_true", help="Benchmark the kernel and report TFLOPS")
    mode_group.add_argument("--check", action="store_true", help="Check output against torch")
    parser.add_argument("--warmup", type=int, default=10, help="Benchmark warmup iterations")
    parser.add_argument("--probe-iters", type=int, default=20, help="Iterations for CUDA event timing probe")
    parser.add_argument("--graph-ms", type=float, default=100.0, help="Target CUDA graph body duration in ms")
    parser.add_argument("--n-replays", type=int, default=20, help="Number of CUDA graph replays to time")
    parser.add_argument("--iters-per-graph", type=int, default=None, help="Override graph body iteration count")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    NUM_BUFFERS = args.num_buffers
    NUM_WARPS = 8
    TRANSPOSE_B = True
    DUMP = args.dump
    USE_KERNEL_B = args.kernelB
    USE_KERNEL_C = args.kernelC
    USE_PERSISTENT_KERNEL_B = args.persistent_kernel_b
    GROUP_SIZE_M = args.group_size_m
    USE_TDM_STORE = args.tdmstore
    INPUT_DTYPE = torch.bfloat16 if args.bf16 else torch.float16
    if sum((USE_KERNEL_B, USE_KERNEL_C, USE_PERSISTENT_KERNEL_B)) > 1:
        raise ValueError("--kernelB, --kernelC, and --persistent-kernelB are mutually exclusive")
    if (USE_KERNEL_C or USE_PERSISTENT_KERNEL_B) and args.four_warp_tdm:
        raise ValueError("--kernelC/--persistent-kernelB and --4warp-tdm are mutually exclusive")
    if USE_TDM_STORE and (not (USE_KERNEL_B or USE_KERNEL_C or USE_PERSISTENT_KERNEL_B) or args.four_warp_tdm):
        raise ValueError("--tdmstore is currently wired only for --kernelB/--kernelC/--persistent-kernelB")
    if args.num_wgps <= 0:
        raise ValueError("--num-wgps must be positive")
    print(
        f"({M=}, {N=}, {K=}), ({BLOCK_M=}, {BLOCK_N=}, {BLOCK_K=}), {TRANSPOSE_B=}, {NUM_WARPS=}, {NUM_BUFFERS=}, {USE_KERNEL_B=}, {USE_KERNEL_C=}, {USE_PERSISTENT_KERNEL_B=}, {GROUP_SIZE_M=}, {USE_TDM_STORE=}, {args.num_wgps=}, {INPUT_DTYPE=}"
    )

    if args.benchmark:
        _benchmark_gemm(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, USE_KERNEL_B, USE_KERNEL_C,
                        USE_PERSISTENT_KERNEL_B, args.four_warp_tdm, GROUP_SIZE_M, USE_TDM_STORE, INPUT_DTYPE,
                        args.num_wgps, args.warmup, args.probe_iters, args.graph_ms, args.n_replays,
                        args.iters_per_graph)
        raise SystemExit

    if args.four_warp_tdm:
        test_runtime_gemm_tdm_predicated_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, DUMP,
                                                   INPUT_DTYPE, args.check)
    else:
        test_runtime_gemm_tdm_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, DUMP,
                                        USE_KERNEL_B, USE_KERNEL_C, USE_PERSISTENT_KERNEL_B, GROUP_SIZE_M,
                                        USE_TDM_STORE, INPUT_DTYPE, args.check, args.num_wgps)
