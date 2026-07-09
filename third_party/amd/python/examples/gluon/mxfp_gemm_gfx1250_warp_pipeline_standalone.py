import time
import torch
import triton
from triton._internal_testing import is_hip_gfx1250
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm
from triton.tools.mxfp import MXScaleTensor

try:
    import pytest
except ModuleNotFoundError:

    class _PytestMarkStub:

        @staticmethod
        def parametrize(*_args, **_kwargs):

            def decorator(fn):
                return fn

            return decorator

        @staticmethod
        def skipif(*_args, **_kwargs):

            def decorator(fn):
                return fn

            return decorator

    class _PytestStub:
        mark = _PytestMarkStub()

    pytest = _PytestStub()


@gluon.jit
def _tdm_async_load(desc, dest, TDM_WARP_USED_HINT: gl.constexpr):
    if TDM_WARP_USED_HINT == 0:
        tdm.async_load(desc, [0, 0], dest)
    else:
        tdm.async_load(desc, [0, 0], dest, warp_used_hint=TDM_WARP_USED_HINT)


@gluon.jit
def mxgemm_tdm_warp_pipeline_standalone_kernel(a_ptr, b_ptr, c_ptr, a_scale, b_scale, M, N, K, stride_am, stride_ak,
                                               stride_bk, stride_bn, stride_cm, stride_cn, stride_scale,
                                               DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr,
                                               SCALE_BLOCK: gl.constexpr, BLOCK_M: gl.constexpr,
                                               BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                                               GROUP_SIZE_M: gl.constexpr, NUM_BUFFERS: gl.constexpr,
                                               NUM_WARPS: gl.constexpr, USE_SCALES: gl.constexpr,
                                               TDM_WARP_USED_HINT: gl.constexpr,
                                               USE_PARTITIONED_LAYOUT: gl.constexpr):
    DIV_FACTOR_A: gl.constexpr = 2 if DTYPE_A == "e2m1" else 1
    DIV_FACTOR_B: gl.constexpr = 2 if DTYPE_B == "e2m1" else 1
    NUM_LOADS_IN_BATCH: gl.constexpr = 4 if USE_SCALES else 2
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // SCALE_BLOCK
    BLOCK_K_PACKED_A: gl.constexpr = BLOCK_K // DIV_FACTOR_A
    BLOCK_K_PACKED_B: gl.constexpr = BLOCK_K // DIV_FACTOR_B

    gl.static_assert(NUM_WARPS == 8)
    INSTR_M: gl.constexpr = 32 if (DTYPE_A == "e2m1" and DTYPE_B == "e2m1") else 16
    REG_BASES: gl.constexpr = []
    TILES_PER_WARP: gl.constexpr = 1
    WARP_BASES: gl.constexpr = [[0, TILES_PER_WARP], [TILES_PER_WARP, 0], [TILES_PER_WARP * 2, 0]]

    PAD_INTERVAL_A: gl.constexpr = 256 if BLOCK_K_PACKED_A <= 256 else BLOCK_K_PACKED_A
    PAD_INTERVAL_B: gl.constexpr = 256 if BLOCK_K_PACKED_B <= 256 else BLOCK_K_PACKED_B
    padded_layout_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[PAD_INTERVAL_A, 16]],
                                                                            [BLOCK_M, BLOCK_K_PACKED_A], [1, 0])
    padded_layout_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[PAD_INTERVAL_B, 16]],
                                                                            [BLOCK_N, BLOCK_K_PACKED_B], [1, 0])

    if USE_PARTITIONED_LAYOUT:
        _DOT_LAYOUTS: gl.constexpr = gl.amd.gfx1250.make_partitioned_dot_layouts(
            BLOCK_M, BLOCK_N, padded_layout_a, padded_layout_b, NUM_WARPS, [INSTR_M, 16, 128], a_transposed=False,
            b_transposed=True)
        shared_layout_a: gl.constexpr = _DOT_LAYOUTS[0]
        shared_layout_b: gl.constexpr = _DOT_LAYOUTS[1]
        WMMA_LAYOUT: gl.constexpr = _DOT_LAYOUTS[2]
        WMMA_LAYOUT_PACKED: gl.constexpr = WMMA_LAYOUT
    else:
        shared_layout_a: gl.constexpr = padded_layout_a
        shared_layout_b: gl.constexpr = padded_layout_b
        WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(3, True, WARP_BASES, REG_BASES, [INSTR_M, 16, 128])
        WMMA_LAYOUT_PACKED: gl.constexpr = gl.amd.AMDWMMALayout(3, True, WARP_BASES, REG_BASES, [INSTR_M, 16, 64])

    dot_layout_a: gl.constexpr = gl.DotOperandLayout(operand_index=0,
                                                     parent=WMMA_LAYOUT_PACKED if DTYPE_A == "e2m1" else WMMA_LAYOUT,
                                                     k_width=16)
    dot_layout_b: gl.constexpr = gl.DotOperandLayout(operand_index=1,
                                                     parent=WMMA_LAYOUT_PACKED if DTYPE_B == "e2m1" else WMMA_LAYOUT,
                                                     k_width=16)
    layout_a_scale: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_layout_a, [BLOCK_M, BLOCK_K_SCALE])
    layout_b_scale: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_layout_b, [BLOCK_N, BLOCK_K_SCALE])
    acc_layout: gl.constexpr = WMMA_LAYOUT

    shared_layout_a_scale: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [BLOCK_M, BLOCK_K_SCALE], [1, 0])
    shared_layout_b_scale: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [BLOCK_N, BLOCK_K_SCALE], [1, 0])

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_offs = pid_m * BLOCK_M * stride_am
    b_offs = pid_n * BLOCK_N * stride_bn
    a_scale_offs = pid_m * BLOCK_M * stride_scale
    b_scale_offs = pid_n * BLOCK_N * stride_scale

    a_desc = tdm.make_tensor_descriptor(base=a_ptr + a_offs, shape=(M, K // DIV_FACTOR_A),
                                        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, BLOCK_K_PACKED_A),
                                        layout=shared_layout_a)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + b_offs, shape=(N, K // DIV_FACTOR_B),
                                        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, BLOCK_K_PACKED_B),
                                        layout=shared_layout_b)

    if USE_SCALES:
        a_scale_desc = tdm.make_tensor_descriptor(
            base=a_scale + a_scale_offs, shape=(M, K // SCALE_BLOCK), strides=(stride_scale, 1),
            block_shape=(BLOCK_M, BLOCK_K_SCALE),
            layout=shared_layout_a_scale)
        b_scale_desc = tdm.make_tensor_descriptor(
            base=b_scale + b_scale_offs, shape=(N, K // SCALE_BLOCK), strides=(stride_scale, 1),
            block_shape=(BLOCK_N, BLOCK_K_SCALE),
            layout=shared_layout_b_scale)

    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, acc_layout))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, acc_layout))
    c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    a_buffer = gl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = gl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)
    if USE_SCALES:
        a_scale_buffer = gl.allocate_shared_memory(a_scale_desc.dtype, shape=[NUM_BUFFERS] + a_scale_desc.block_shape,
                                                   layout=a_scale_desc.layout)
        b_scale_buffer = gl.allocate_shared_memory(b_scale_desc.dtype, shape=[NUM_BUFFERS] + b_scale_desc.block_shape,
                                                   layout=b_scale_desc.layout)

    load_idx = 0
    wmma_idx = 0

    # Standalone copy of MXFPGEMMPipelinedProgram.warp_pipeline, with the issue_* helper bodies inlined.
    for _ in gl.static_range(NUM_BUFFERS - 1):
        slot = load_idx % NUM_BUFFERS
        if USE_SCALES:
            _tdm_async_load(a_scale_desc, a_scale_buffer.index(slot), TDM_WARP_USED_HINT)
            a_scale_desc = tdm.update_tensor_descriptor(a_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
            _tdm_async_load(b_scale_desc, b_scale_buffer.index(slot), TDM_WARP_USED_HINT)
            b_scale_desc = tdm.update_tensor_descriptor(b_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
        _tdm_async_load(a_desc, a_buffer.index(slot), TDM_WARP_USED_HINT)
        a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K_PACKED_A])
        _tdm_async_load(b_desc, b_buffer.index(slot), TDM_WARP_USED_HINT)
        b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K_PACKED_B])
        load_idx = load_idx + 1

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=acc_layout)
    loop_ub = gl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)
    tdm.async_wait((NUM_BUFFERS - 2) * NUM_LOADS_IN_BATCH)
    #tdm.async_wait(0)
    gl.assume(loop_ub >= 0)
    for _ in range(0, loop_ub):
        with gl.amd.warp_pipeline_stage("tdm+lds", priority=1):
            a = a_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=dot_layout_a)
            b = b_buffer.index(wmma_idx % NUM_BUFFERS).permute([1, 0]).load(layout=dot_layout_b)
            if USE_SCALES:
                a_scale_buffer_slice = a_scale_buffer.index(wmma_idx % NUM_BUFFERS)
                b_scale_buffer_slice = b_scale_buffer.index(wmma_idx % NUM_BUFFERS)
                scale_a = a_scale_buffer_slice.load(layout=layout_a_scale)
                scale_b = b_scale_buffer_slice.load(layout=layout_b_scale)

            wmma_idx += 1
            phase = wmma_idx + NUM_BUFFERS - 2
            slot = phase % NUM_BUFFERS
            if USE_SCALES:
                _tdm_async_load(a_scale_desc, a_scale_buffer.index(slot), TDM_WARP_USED_HINT)
                a_scale_desc = tdm.update_tensor_descriptor(a_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
                _tdm_async_load(b_scale_desc, b_scale_buffer.index(slot), TDM_WARP_USED_HINT)
                b_scale_desc = tdm.update_tensor_descriptor(b_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
            _tdm_async_load(a_desc, a_buffer.index(slot), TDM_WARP_USED_HINT)
            a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K_PACKED_A])
            _tdm_async_load(b_desc, b_buffer.index(slot), TDM_WARP_USED_HINT)
            b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K_PACKED_B])

        tdm.async_wait((NUM_BUFFERS - 2) * NUM_LOADS_IN_BATCH)
        #tdm.async_wait(0)
        with gl.amd.warp_pipeline_stage("wmma", priority=0):
            load_idx = load_idx + 1
            if USE_SCALES:
                accumulator = gl.amd.gfx1250.wmma_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator)
            else:
                accumulator = gl.amd.gfx1250.wmma_scaled(a, None, DTYPE_A, b, None, DTYPE_B, accumulator)

    for i in gl.static_range(NUM_BUFFERS - 1):
        tdm.async_wait((NUM_BUFFERS - 1 - i) * NUM_LOADS_IN_BATCH)
        #tdm.async_wait(0)
        a = a_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=dot_layout_a)
        b = b_buffer.index(wmma_idx % NUM_BUFFERS).permute([1, 0]).load(layout=dot_layout_b)
        if USE_SCALES:
            a_scale_buffer_slice = a_scale_buffer.index(wmma_idx % NUM_BUFFERS)
            b_scale_buffer_slice = b_scale_buffer.index(wmma_idx % NUM_BUFFERS)
            scale_a = a_scale_buffer_slice.load(layout=layout_a_scale)
            scale_b = b_scale_buffer_slice.load(layout=layout_b_scale)
        wmma_idx += 1
        if USE_SCALES:
            accumulator = gl.amd.gfx1250.wmma_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator)
        else:
            accumulator = gl.amd.gfx1250.wmma_scaled(a, None, DTYPE_A, b, None, DTYPE_B, accumulator)

    gl.amd.gfx1250.buffer_store(accumulator, c_ptr, c_offs, mask=c_mask)
    tdm.async_wait(0)


@gluon.jit
def mxgemm_tdm_warp_pipeline_local_address_kernel(a_ptr, b_ptr, c_ptr, a_scale, b_scale, M, N, K: gl.constexpr,
                                                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                                                  stride_scale, DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr,
                                                  SCALE_BLOCK: gl.constexpr, BLOCK_M: gl.constexpr,
                                                  BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                                                  GROUP_SIZE_M: gl.constexpr, NUM_BUFFERS: gl.constexpr,
                                                  NUM_WARPS: gl.constexpr, USE_SCALES: gl.constexpr,
                                                  TDM_WARP_USED_HINT: gl.constexpr):
    DIV_FACTOR_A: gl.constexpr = 2 if DTYPE_A == "e2m1" else 1
    DIV_FACTOR_B: gl.constexpr = 2 if DTYPE_B == "e2m1" else 1
    NUM_LOADS_IN_BATCH: gl.constexpr = 4 if USE_SCALES else 2
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // SCALE_BLOCK
    BLOCK_K_PACKED_A: gl.constexpr = BLOCK_K // DIV_FACTOR_A
    BLOCK_K_PACKED_B: gl.constexpr = BLOCK_K // DIV_FACTOR_B

    gl.static_assert(NUM_WARPS == 8)
    INSTR_M: gl.constexpr = 32 if (DTYPE_A == "e2m1" and DTYPE_B == "e2m1") else 16
    REG_BASES: gl.constexpr = []
    TILES_PER_WARP: gl.constexpr = 1
    WARP_BASES: gl.constexpr = [[0, TILES_PER_WARP], [TILES_PER_WARP, 0], [TILES_PER_WARP * 2, 0]]

    WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(3, True, WARP_BASES, REG_BASES, [INSTR_M, 16, 128])
    WMMA_LAYOUT_PACKED: gl.constexpr = gl.amd.AMDWMMALayout(3, True, WARP_BASES, REG_BASES, [INSTR_M, 16, 64])
    dot_layout_a: gl.constexpr = gl.DotOperandLayout(operand_index=0,
                                                     parent=WMMA_LAYOUT_PACKED if DTYPE_A == "e2m1" else WMMA_LAYOUT,
                                                     k_width=16)
    dot_layout_b: gl.constexpr = gl.DotOperandLayout(operand_index=1,
                                                     parent=WMMA_LAYOUT_PACKED if DTYPE_B == "e2m1" else WMMA_LAYOUT,
                                                     k_width=16)
    layout_a_scale: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_layout_a, [BLOCK_M, BLOCK_K_SCALE])
    layout_b_scale: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_layout_b, [BLOCK_N, BLOCK_K_SCALE])
    acc_layout: gl.constexpr = WMMA_LAYOUT

    PAD_INTERVAL_A: gl.constexpr = 256 if BLOCK_K_PACKED_A <= 256 else BLOCK_K_PACKED_A
    PAD_INTERVAL_B: gl.constexpr = 256 if BLOCK_K_PACKED_B <= 256 else BLOCK_K_PACKED_B
    shared_layout_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[PAD_INTERVAL_A, 16]],
                                                                            [BLOCK_M, BLOCK_K_PACKED_A], [1, 0])
    shared_layout_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[PAD_INTERVAL_B, 16]],
                                                                            [BLOCK_N, BLOCK_K_PACKED_B], [1, 0])
    shared_layout_a_scale: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [BLOCK_M, BLOCK_K_SCALE], [1, 0])
    shared_layout_b_scale: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [BLOCK_N, BLOCK_K_SCALE], [1, 0])

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_offs = pid_m * BLOCK_M * stride_am
    b_offs = pid_n * BLOCK_N * stride_bn
    a_scale_offs = pid_m * BLOCK_M * stride_scale
    b_scale_offs = pid_n * BLOCK_N * stride_scale

    a_desc = tdm.make_tensor_descriptor(base=a_ptr + a_offs, shape=(M, K // DIV_FACTOR_A),
                                        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, BLOCK_K_PACKED_A),
                                        layout=shared_layout_a)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + b_offs, shape=(N, K // DIV_FACTOR_B),
                                        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, BLOCK_K_PACKED_B),
                                        layout=shared_layout_b)

    if USE_SCALES:
        a_scale_desc = tdm.make_tensor_descriptor(
            base=a_scale + a_scale_offs, shape=(M, K // SCALE_BLOCK), strides=(stride_scale, 1),
            block_shape=(BLOCK_M, BLOCK_K_SCALE),
            layout=shared_layout_a_scale)
        b_scale_desc = tdm.make_tensor_descriptor(
            base=b_scale + b_scale_offs, shape=(N, K // SCALE_BLOCK), strides=(stride_scale, 1),
            block_shape=(BLOCK_N, BLOCK_K_SCALE),
            layout=shared_layout_b_scale)

    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, acc_layout))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, acc_layout))
    c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    a_buffer = gl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = gl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)
    if USE_SCALES:
        a_scale_buffer = gl.allocate_shared_memory(a_scale_desc.dtype, shape=[NUM_BUFFERS] + a_scale_desc.block_shape,
                                                   layout=a_scale_desc.layout)
        b_scale_buffer = gl.allocate_shared_memory(b_scale_desc.dtype, shape=[NUM_BUFFERS] + b_scale_desc.block_shape,
                                                   layout=b_scale_desc.layout)

    load_idx = 0
    wmma_idx = 0

    for _ in gl.static_range(NUM_BUFFERS - 1):
        slot = load_idx % NUM_BUFFERS
        if USE_SCALES:
            _tdm_async_load(a_scale_desc, a_scale_buffer.index(slot), TDM_WARP_USED_HINT)
            a_scale_desc = tdm.update_tensor_descriptor(a_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
            _tdm_async_load(b_scale_desc, b_scale_buffer.index(slot), TDM_WARP_USED_HINT)
            b_scale_desc = tdm.update_tensor_descriptor(b_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
        _tdm_async_load(a_desc, a_buffer.index(slot), TDM_WARP_USED_HINT)
        a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K_PACKED_A])
        _tdm_async_load(b_desc, b_buffer.index(slot), TDM_WARP_USED_HINT)
        b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K_PACKED_B])
        load_idx = load_idx + 1

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=acc_layout)
    LOOP_UB: gl.constexpr = (K + BLOCK_K - 1) // BLOCK_K - (NUM_BUFFERS - 1)
    tdm.async_wait(0)
    gl.static_assert(LOOP_UB >= 0)
    a_addr = a_buffer.index(wmma_idx % NUM_BUFFERS).local_address(dot_layout_a)
    b_addr = b_buffer.index(wmma_idx % NUM_BUFFERS).permute([1, 0]).local_address(dot_layout_b)
    if USE_SCALES:
        scale_a_addr = a_scale_buffer.index(wmma_idx % NUM_BUFFERS).local_address(layout_a_scale)
        scale_b_addr = b_scale_buffer.index(wmma_idx % NUM_BUFFERS).local_address(layout_b_scale)

    for _ in gl.static_range(LOOP_UB):
        with gl.amd.warp_pipeline_stage("tdm+lds", priority=1):
            a = a_addr.load()
            b = b_addr.load()
            if USE_SCALES:
                scale_a = scale_a_addr.load()
                scale_b = scale_b_addr.load()
            wmma_idx += 1
            phase = wmma_idx + NUM_BUFFERS - 2
            slot = phase % NUM_BUFFERS
            if USE_SCALES:
                _tdm_async_load(a_scale_desc, a_scale_buffer.index(slot), TDM_WARP_USED_HINT)
                a_scale_desc = tdm.update_tensor_descriptor(a_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
                _tdm_async_load(b_scale_desc, b_scale_buffer.index(slot), TDM_WARP_USED_HINT)
                b_scale_desc = tdm.update_tensor_descriptor(b_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
            _tdm_async_load(a_desc, a_buffer.index(slot), TDM_WARP_USED_HINT)
            a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K_PACKED_A])
            _tdm_async_load(b_desc, b_buffer.index(slot), TDM_WARP_USED_HINT)
            b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K_PACKED_B])

        tdm.async_wait(0)
        with gl.amd.warp_pipeline_stage("wmma", priority=0):
            load_idx = load_idx + 1
            if USE_SCALES:
                accumulator = gl.amd.gfx1250.wmma_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator)
            else:
                accumulator = gl.amd.gfx1250.wmma_scaled(a, None, DTYPE_A, b, None, DTYPE_B, accumulator)
            a_addr = a_buffer.index(wmma_idx % NUM_BUFFERS).local_address(dot_layout_a)
            b_addr = b_buffer.index(wmma_idx % NUM_BUFFERS).permute([1, 0]).local_address(dot_layout_b)
            if USE_SCALES:
                scale_a_addr = a_scale_buffer.index(wmma_idx % NUM_BUFFERS).local_address(layout_a_scale)
                scale_b_addr = b_scale_buffer.index(wmma_idx % NUM_BUFFERS).local_address(layout_b_scale)

    for i in gl.static_range(NUM_BUFFERS - 1):
        tdm.async_wait(0)
        a = a_addr.load()
        b = b_addr.load()
        if USE_SCALES:
            scale_a = scale_a_addr.load()
            scale_b = scale_b_addr.load()
        wmma_idx += 1
        if USE_SCALES:
            accumulator = gl.amd.gfx1250.wmma_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator)
        else:
            accumulator = gl.amd.gfx1250.wmma_scaled(a, None, DTYPE_A, b, None, DTYPE_B, accumulator)
        if i != NUM_BUFFERS - 2:
            a_addr = a_buffer.index(wmma_idx % NUM_BUFFERS).local_address(dot_layout_a)
            b_addr = b_buffer.index(wmma_idx % NUM_BUFFERS).permute([1, 0]).local_address(dot_layout_b)
            if USE_SCALES:
                scale_a_addr = a_scale_buffer.index(wmma_idx % NUM_BUFFERS).local_address(layout_a_scale)
                scale_b_addr = b_scale_buffer.index(wmma_idx % NUM_BUFFERS).local_address(layout_b_scale)

    gl.amd.gfx1250.buffer_store(accumulator, c_ptr, c_offs, mask=c_mask)
    tdm.async_wait(0)


@gluon.jit
def mxgemm_tdm_warp_pipeline_slice_mn_kernel(a_ptr, b_ptr, c_ptr, a_scale, b_scale, M, N, K, stride_am, stride_ak,
                                             stride_bk, stride_bn, stride_cm, stride_cn, stride_scale,
                                             DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr,
                                             SCALE_BLOCK: gl.constexpr, BLOCK_M: gl.constexpr,
                                             BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                                             GROUP_SIZE_M: gl.constexpr, NUM_BUFFERS: gl.constexpr,
                                             NUM_WARPS: gl.constexpr):
    DIV_FACTOR_A: gl.constexpr = 2 if DTYPE_A == "e2m1" else 1
    DIV_FACTOR_B: gl.constexpr = 2 if DTYPE_B == "e2m1" else 1
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // SCALE_BLOCK
    BLOCK_K_PACKED_A: gl.constexpr = BLOCK_K // DIV_FACTOR_A
    BLOCK_K_PACKED_B: gl.constexpr = BLOCK_K // DIV_FACTOR_B
    NUM_LOADS_IN_SLICE_STEP: gl.constexpr = 4
    SLICE_STAGE_WAIT: gl.constexpr = (NUM_BUFFERS - 1) * NUM_LOADS_IN_SLICE_STEP
    SUBTILE_M: gl.constexpr = BLOCK_M // 2
    SUBTILE_N: gl.constexpr = BLOCK_N // 2

    gl.static_assert(NUM_WARPS == 8)
    gl.static_assert(NUM_BUFFERS == 2)
    gl.static_assert(BLOCK_M % 2 == 0)
    gl.static_assert(BLOCK_N % 2 == 0)
    gl.static_assert(BLOCK_K % SCALE_BLOCK == 0)

    INSTR_M: gl.constexpr = 32 if (DTYPE_A == "e2m1" and DTYPE_B == "e2m1") else 16
    REG_BASES: gl.constexpr = []
    TILES_PER_WARP: gl.constexpr = 1
    WARP_BASES: gl.constexpr = [[0, TILES_PER_WARP], [TILES_PER_WARP, 0], [TILES_PER_WARP * 2, 0]]

    WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(3, True, WARP_BASES, REG_BASES, [INSTR_M, 16, 128])
    WMMA_LAYOUT_PACKED: gl.constexpr = gl.amd.AMDWMMALayout(3, True, WARP_BASES, REG_BASES, [INSTR_M, 16, 64])
    dot_layout_a: gl.constexpr = gl.DotOperandLayout(operand_index=0,
                                                     parent=WMMA_LAYOUT_PACKED if DTYPE_A == "e2m1" else WMMA_LAYOUT,
                                                     k_width=16)
    dot_layout_b: gl.constexpr = gl.DotOperandLayout(operand_index=1,
                                                     parent=WMMA_LAYOUT_PACKED if DTYPE_B == "e2m1" else WMMA_LAYOUT,
                                                     k_width=16)
    layout_a_scale: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_layout_a, [SUBTILE_M, BLOCK_K_SCALE])
    layout_b_scale: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_layout_b, [SUBTILE_N, BLOCK_K_SCALE])
    acc_layout: gl.constexpr = WMMA_LAYOUT

    PAD_INTERVAL_A: gl.constexpr = 256 if BLOCK_K_PACKED_A <= 256 else BLOCK_K_PACKED_A
    PAD_INTERVAL_B: gl.constexpr = 256 if BLOCK_K_PACKED_B <= 256 else BLOCK_K_PACKED_B
    shared_layout_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[PAD_INTERVAL_A, 16]],
                                                                            [BLOCK_M, BLOCK_K_PACKED_A], [1, 0])
    shared_layout_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[PAD_INTERVAL_B, 16]],
                                                                            [BLOCK_N, BLOCK_K_PACKED_B], [1, 0])
    shared_layout_a_scale: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [BLOCK_M, BLOCK_K_SCALE], [1, 0])
    shared_layout_b_scale: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [BLOCK_N, BLOCK_K_SCALE], [1, 0])

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_offs = pid_m * BLOCK_M * stride_am
    b_offs = pid_n * BLOCK_N * stride_bn
    a_scale_offs = pid_m * BLOCK_M * stride_scale
    b_scale_offs = pid_n * BLOCK_N * stride_scale

    a_desc = tdm.make_tensor_descriptor(base=a_ptr + a_offs, shape=(M, K // DIV_FACTOR_A),
                                        strides=(stride_am, stride_ak),
                                        block_shape=(BLOCK_M, BLOCK_K_PACKED_A),
                                        layout=shared_layout_a)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + b_offs, shape=(N, K // DIV_FACTOR_B),
                                        strides=(stride_bn, stride_bk),
                                        block_shape=(BLOCK_N, BLOCK_K_PACKED_B),
                                        layout=shared_layout_b)

    a_scale_desc = tdm.make_tensor_descriptor(
        base=a_scale + a_scale_offs, shape=(M, K // SCALE_BLOCK), strides=(stride_scale, 1),
        block_shape=(BLOCK_M, BLOCK_K_SCALE),
        layout=shared_layout_a_scale)
    b_scale_desc = tdm.make_tensor_descriptor(
        base=b_scale + b_scale_offs, shape=(N, K // SCALE_BLOCK), strides=(stride_scale, 1),
        block_shape=(BLOCK_N, BLOCK_K_SCALE),
        layout=shared_layout_b_scale)

    a_buffer = gl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = gl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)
    a_scale_buffer = gl.allocate_shared_memory(a_scale_desc.dtype, shape=[NUM_BUFFERS] + a_scale_desc.block_shape,
                                               layout=a_scale_desc.layout)
    b_scale_buffer = gl.allocate_shared_memory(b_scale_desc.dtype, shape=[NUM_BUFFERS] + b_scale_desc.block_shape,
                                               layout=b_scale_desc.layout)

    load_idx = 0
    wmma_idx = 0

    # Double-buffered prologue: load K-steps 0 and 1 before the quadrant pipeline.
    for _ in gl.static_range(NUM_BUFFERS):
        slot = load_idx % NUM_BUFFERS
        tdm.async_load(a_scale_desc, [0, 0], a_scale_buffer.index(slot))
        a_scale_desc = tdm.update_tensor_descriptor(a_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
        tdm.async_load(a_desc, [0, 0], a_buffer.index(slot))
        a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K_PACKED_A])
        tdm.async_load(b_scale_desc, [0, 0], b_scale_buffer.index(slot))
        b_scale_desc = tdm.update_tensor_descriptor(b_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
        tdm.async_load(b_desc, [0, 0], b_buffer.index(slot))
        b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K_PACKED_B])
        load_idx = load_idx + 1

    acc_tl = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=acc_layout)
    acc_bl = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=acc_layout)
    acc_tr = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=acc_layout)
    acc_br = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=acc_layout)
    loop_ub = gl.cdiv(K, BLOCK_K) - NUM_BUFFERS
    gl.assume(loop_ub >= 0)
    slot = wmma_idx % NUM_BUFFERS
    load_slot = load_idx % NUM_BUFFERS

    for _ in range(0, loop_ub):
        tdm.async_wait(SLICE_STAGE_WAIT)
        with gl.amd.warp_pipeline_stage("lds_tl", priority=1):
            a_top = a_buffer.index(slot).slice(0, SUBTILE_M, 0).load(layout=dot_layout_a)
            scale_a_top = a_scale_buffer.index(slot).slice(0, SUBTILE_M, 0).load(layout=layout_a_scale)
            b_left = b_buffer.index(slot).slice(0, SUBTILE_N, 0).permute([1, 0]).load(layout=dot_layout_b)
            scale_b_left = b_scale_buffer.index(slot).slice(0, SUBTILE_N, 0).load(layout=layout_b_scale)

        with gl.amd.warp_pipeline_stage("wmma_tl", priority=0):
            acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, scale_a_top, DTYPE_A, b_left, scale_b_left, DTYPE_B, acc_tl)

        tdm.async_wait(SLICE_STAGE_WAIT)
        with gl.amd.warp_pipeline_stage("lds_bl", priority=1):
            a_bot = a_buffer.index(slot).slice(SUBTILE_M, SUBTILE_M, 0).load(layout=dot_layout_a)
            scale_a_bot = a_scale_buffer.index(slot).slice(SUBTILE_M, SUBTILE_M, 0).load(layout=layout_a_scale)

        with gl.amd.warp_pipeline_stage("wmma_bl", priority=0):
            acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, scale_a_bot, DTYPE_A, b_left, scale_b_left, DTYPE_B, acc_bl)

        tdm.async_wait(SLICE_STAGE_WAIT)
        with gl.amd.warp_pipeline_stage("lds_tr", priority=1):
            b_right = b_buffer.index(slot).slice(SUBTILE_N, SUBTILE_N, 0).permute([1, 0]).load(layout=dot_layout_b)
            scale_b_right = b_scale_buffer.index(slot).slice(SUBTILE_N, SUBTILE_N, 0).load(layout=layout_b_scale)
            tdm.async_load(a_scale_desc, [0, 0], a_scale_buffer.index(load_slot))
            a_scale_desc = tdm.update_tensor_descriptor(a_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
            tdm.async_load(a_desc, [0, 0], a_buffer.index(load_slot))
            a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K_PACKED_A])

        with gl.amd.warp_pipeline_stage("wmma_tr", priority=0):
            acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, scale_a_top, DTYPE_A, b_right, scale_b_right, DTYPE_B, acc_tr)

        tdm.async_wait(SLICE_STAGE_WAIT)
        with gl.amd.warp_pipeline_stage("lds_br", priority=1):
            tdm.async_load(b_scale_desc, [0, 0], b_scale_buffer.index(load_slot))
            b_scale_desc = tdm.update_tensor_descriptor(b_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
            tdm.async_load(b_desc, [0, 0], b_buffer.index(load_slot))
            b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K_PACKED_B])

        with gl.amd.warp_pipeline_stage("wmma_br", priority=0):
            acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, scale_a_bot, DTYPE_A, b_right, scale_b_right, DTYPE_B, acc_br)
            wmma_idx += 1
            load_idx = load_idx + 1
            slot = wmma_idx % NUM_BUFFERS
            load_slot = load_idx % NUM_BUFFERS

    for i in gl.static_range(NUM_BUFFERS):
        tdm.async_wait((NUM_BUFFERS - 1 - i) * NUM_LOADS_IN_SLICE_STEP)
        slot = wmma_idx % NUM_BUFFERS
        a_top = a_buffer.index(slot).slice(0, SUBTILE_M, 0).load(layout=dot_layout_a)
        scale_a_top = a_scale_buffer.index(slot).slice(0, SUBTILE_M, 0).load(layout=layout_a_scale)
        b_left = b_buffer.index(slot).slice(0, SUBTILE_N, 0).permute([1, 0]).load(layout=dot_layout_b)
        scale_b_left = b_scale_buffer.index(slot).slice(0, SUBTILE_N, 0).load(layout=layout_b_scale)
        a_bot = a_buffer.index(slot).slice(SUBTILE_M, SUBTILE_M, 0).load(layout=dot_layout_a)
        scale_a_bot = a_scale_buffer.index(slot).slice(SUBTILE_M, SUBTILE_M, 0).load(layout=layout_a_scale)
        b_right = b_buffer.index(slot).slice(SUBTILE_N, SUBTILE_N, 0).permute([1, 0]).load(layout=dot_layout_b)
        scale_b_right = b_scale_buffer.index(slot).slice(SUBTILE_N, SUBTILE_N, 0).load(layout=layout_b_scale)
        acc_tl = gl.amd.gfx1250.wmma_scaled(a_top, scale_a_top, DTYPE_A, b_left, scale_b_left, DTYPE_B, acc_tl)
        acc_bl = gl.amd.gfx1250.wmma_scaled(a_bot, scale_a_bot, DTYPE_A, b_left, scale_b_left, DTYPE_B, acc_bl)
        acc_tr = gl.amd.gfx1250.wmma_scaled(a_top, scale_a_top, DTYPE_A, b_right, scale_b_right, DTYPE_B, acc_tr)
        acc_br = gl.amd.gfx1250.wmma_scaled(a_bot, scale_a_bot, DTYPE_A, b_right, scale_b_right, DTYPE_B, acc_br)
        wmma_idx += 1

    acc_top = gl.join(acc_tl, acc_tr).permute(0, 2, 1).reshape((SUBTILE_M, BLOCK_N))
    acc_bot = gl.join(acc_bl, acc_br).permute(0, 2, 1).reshape((SUBTILE_M, BLOCK_N))
    accumulator = gl.join(acc_top, acc_bot).permute(2, 0, 1).reshape((BLOCK_M, BLOCK_N))
    accumulator = gl.convert_layout(accumulator, acc_layout)

    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, acc_layout))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, acc_layout))
    c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.gfx1250.buffer_store(accumulator, c_ptr, c_offs, mask=c_mask)


@gluon.jit
def fp8_gemm_tdm_warp_pipeline_slice_mn_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk,
                                               stride_bn, stride_cm, stride_cn, BLOCK_M: gl.constexpr,
                                               BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                                               GROUP_SIZE_M: gl.constexpr, NUM_BUFFERS: gl.constexpr,
                                               NUM_WARPS: gl.constexpr):
    BLOCK_K_PACKED_A: gl.constexpr = BLOCK_K
    BLOCK_K_PACKED_B: gl.constexpr = BLOCK_K
    NUM_LOADS_IN_SLICE_STEP: gl.constexpr = 2
    SLICE_STAGE_WAIT: gl.constexpr = (NUM_BUFFERS - 1) * NUM_LOADS_IN_SLICE_STEP
    SUBTILE_M: gl.constexpr = BLOCK_M // 2
    SUBTILE_N: gl.constexpr = BLOCK_N // 2

    gl.static_assert(NUM_WARPS == 8)
    gl.static_assert(NUM_BUFFERS == 2)
    gl.static_assert(BLOCK_M % 2 == 0)
    gl.static_assert(BLOCK_N % 2 == 0)

    TILES_PER_WARP: gl.constexpr = 1
    WARP_BASES: gl.constexpr = [[0, TILES_PER_WARP], [TILES_PER_WARP, 0], [TILES_PER_WARP * 2, 0]]
    WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 128])
    dot_layout_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=WMMA_LAYOUT, k_width=16)
    dot_layout_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=WMMA_LAYOUT, k_width=16)
    acc_layout: gl.constexpr = WMMA_LAYOUT

    PAD_INTERVAL_A: gl.constexpr = 256 if BLOCK_K_PACKED_A <= 256 else BLOCK_K_PACKED_A
    PAD_INTERVAL_B: gl.constexpr = 256 if BLOCK_K_PACKED_B <= 256 else BLOCK_K_PACKED_B
    shared_layout_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[PAD_INTERVAL_A, 16]],
                                                                            [BLOCK_M, BLOCK_K_PACKED_A], [1, 0])
    shared_layout_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[PAD_INTERVAL_B, 16]],
                                                                            [BLOCK_N, BLOCK_K_PACKED_B], [1, 0])

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_offs = pid_m * BLOCK_M * stride_am
    b_offs = pid_n * BLOCK_N * stride_bn

    a_desc = tdm.make_tensor_descriptor(base=a_ptr + a_offs, shape=(M, K), strides=(stride_am, stride_ak),
                                        block_shape=(BLOCK_M, BLOCK_K_PACKED_A), layout=shared_layout_a)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + b_offs, shape=(N, K), strides=(stride_bn, stride_bk),
                                        block_shape=(BLOCK_N, BLOCK_K_PACKED_B), layout=shared_layout_b)

    a_buffer = gl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = gl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    load_idx = 0
    wmma_idx = 0

    for _ in gl.static_range(NUM_BUFFERS):
        slot = load_idx % NUM_BUFFERS
        tdm.async_load(a_desc, [0, 0], a_buffer.index(slot))
        a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K_PACKED_A])
        tdm.async_load(b_desc, [0, 0], b_buffer.index(slot))
        b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K_PACKED_B])
        load_idx = load_idx + 1

    acc_tl = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=acc_layout)
    acc_bl = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=acc_layout)
    acc_tr = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=acc_layout)
    acc_br = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=acc_layout)
    loop_ub = gl.cdiv(K, BLOCK_K) - NUM_BUFFERS
    gl.assume(loop_ub >= 0)

    slot = wmma_idx % NUM_BUFFERS
    load_slot = load_idx % NUM_BUFFERS
    for _ in range(0, loop_ub):
        tdm.async_wait(SLICE_STAGE_WAIT)
        with gl.amd.warp_pipeline_stage("lds_tl", priority=1):
            a_top = a_buffer.index(slot).slice(0, SUBTILE_M, 0).load(layout=dot_layout_a)
            a_bot = a_buffer.index(slot).slice(SUBTILE_M, SUBTILE_M, 0).load(layout=dot_layout_a)
            b_left = b_buffer.index(slot).slice(0, SUBTILE_N, 0).permute([1, 0]).load(layout=dot_layout_b)

        with gl.amd.warp_pipeline_stage("wmma_tl", priority=0):
            acc_tl = gl.amd.gfx1250.wmma(a_top, b_left, acc_tl)

        tdm.async_wait(SLICE_STAGE_WAIT)
        with gl.amd.warp_pipeline_stage("lds_bl", priority=1):
            b_right = b_buffer.index(slot).slice(SUBTILE_N, SUBTILE_N, 0).permute([1, 0]).load(layout=dot_layout_b)

        with gl.amd.warp_pipeline_stage("wmma_bl", priority=0):
            acc_bl = gl.amd.gfx1250.wmma(a_bot, b_left, acc_bl)

        tdm.async_wait(SLICE_STAGE_WAIT)
        with gl.amd.warp_pipeline_stage("lds_tr", priority=1):
            tdm.async_load(a_desc, [0, 0], a_buffer.index(load_slot))
            a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K_PACKED_A])

        with gl.amd.warp_pipeline_stage("wmma_tr", priority=0):
            acc_tr = gl.amd.gfx1250.wmma(a_top, b_right, acc_tr)

        tdm.async_wait(SLICE_STAGE_WAIT)
        with gl.amd.warp_pipeline_stage("lds_br", priority=1):
            tdm.async_load(b_desc, [0, 0], b_buffer.index(load_slot))
            b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K_PACKED_B])

        with gl.amd.warp_pipeline_stage("wmma_br", priority=0):
            acc_br = gl.amd.gfx1250.wmma(a_bot, b_right, acc_br)
            wmma_idx += 1
            load_idx = load_idx + 1
            slot = wmma_idx % NUM_BUFFERS
            load_slot = load_idx % NUM_BUFFERS

    for i in gl.static_range(NUM_BUFFERS):
        tdm.async_wait((NUM_BUFFERS - 1 - i) * NUM_LOADS_IN_SLICE_STEP)
        slot = wmma_idx % NUM_BUFFERS
        a_top = a_buffer.index(slot).slice(0, SUBTILE_M, 0).load(layout=dot_layout_a)
        b_left = b_buffer.index(slot).slice(0, SUBTILE_N, 0).permute([1, 0]).load(layout=dot_layout_b)
        a_bot = a_buffer.index(slot).slice(SUBTILE_M, SUBTILE_M, 0).load(layout=dot_layout_a)
        b_right = b_buffer.index(slot).slice(SUBTILE_N, SUBTILE_N, 0).permute([1, 0]).load(layout=dot_layout_b)
        acc_tl = gl.amd.gfx1250.wmma(a_top, b_left, acc_tl)
        acc_bl = gl.amd.gfx1250.wmma(a_bot, b_left, acc_bl)
        acc_tr = gl.amd.gfx1250.wmma(a_top, b_right, acc_tr)
        acc_br = gl.amd.gfx1250.wmma(a_bot, b_right, acc_br)
        wmma_idx += 1

    acc_top = gl.join(acc_tl, acc_tr).permute(0, 2, 1).reshape((SUBTILE_M, BLOCK_N))
    acc_bot = gl.join(acc_bl, acc_br).permute(0, 2, 1).reshape((SUBTILE_M, BLOCK_N))
    accumulator = gl.join(acc_top, acc_bot).permute(2, 0, 1).reshape((BLOCK_M, BLOCK_N))
    accumulator = gl.convert_layout(accumulator, acc_layout)

    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, acc_layout))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, acc_layout))
    c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.gfx1250.buffer_store(accumulator, c_ptr, c_offs, mask=c_mask)


@gluon.jit
def fp8_gemm_tdm_warp_pipeline_kernel_b(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                                        stride_cm, stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                                        BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr, NUM_BUFFERS: gl.constexpr,
                                        NUM_WARPS: gl.constexpr):
    gl.static_assert(NUM_WARPS == 8)
    gl.static_assert(NUM_BUFFERS >= 3)
    DIV_FACTOR_A: gl.constexpr = 1
    DIV_FACTOR_B: gl.constexpr = 1
    BLOCK_K_PACKED_A: gl.constexpr = BLOCK_K // DIV_FACTOR_A
    BLOCK_K_PACKED_B: gl.constexpr = BLOCK_K // DIV_FACTOR_B

    TILES_PER_WARP: gl.constexpr = 1
    WARP_BASES: gl.constexpr = [[0, TILES_PER_WARP], [TILES_PER_WARP, 0], [TILES_PER_WARP * 2, 0]]
    WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 128])
    dot_layout_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=WMMA_LAYOUT, k_width=16)
    dot_layout_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=WMMA_LAYOUT, k_width=16)
    acc_layout: gl.constexpr = WMMA_LAYOUT

    PAD_INTERVAL_A: gl.constexpr = 256 if BLOCK_K_PACKED_A <= 256 else BLOCK_K_PACKED_A
    PAD_INTERVAL_B: gl.constexpr = 256 if BLOCK_K_PACKED_B <= 256 else BLOCK_K_PACKED_B
    shared_layout_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[PAD_INTERVAL_A, 16]],
                                                                            [BLOCK_M, BLOCK_K_PACKED_A], [1, 0])
    shared_layout_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[PAD_INTERVAL_B, 16]],
                                                                            [BLOCK_N, BLOCK_K_PACKED_B], [1, 0])

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_offs = pid_m * BLOCK_M * stride_am
    b_offs = pid_n * BLOCK_N * stride_bn
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + a_offs, shape=(M, K),
                                        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, BLOCK_K_PACKED_A),
                                        layout=shared_layout_a)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + b_offs, shape=(N, K),
                                        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, BLOCK_K_PACKED_B),
                                        layout=shared_layout_b)
    a_buffer = gl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = gl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    phase = 0
    producer = 0
    for _ in gl.static_range(NUM_BUFFERS - 1):
        slot = producer % NUM_BUFFERS
        tdm.async_load(a_desc, [0, 0], a_buffer.index(slot))
        a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K_PACKED_A])
        tdm.async_load(b_desc, [0, 0], b_buffer.index(slot))
        b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K_PACKED_B])
        producer += 1

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=acc_layout)
    loop_ub = gl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)
    tdm.async_wait((NUM_BUFFERS - 3) * 2)
    gl.assume(loop_ub >= 0)
    for _ in range(0, loop_ub):
        with gl.amd.warp_pipeline_stage("stage0", priority=0):
            read_slot = phase % NUM_BUFFERS
            write_phase = phase + (NUM_BUFFERS - 1)
            write_slot = write_phase % NUM_BUFFERS
            a = a_buffer.index(read_slot).load(layout=dot_layout_a)
            b = b_buffer.index(read_slot).permute([1, 0]).load(layout=dot_layout_b)
            phase += 1
            tdm.async_load(a_desc, [0, 0], a_buffer.index(write_slot))
            a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K_PACKED_A])
            tdm.async_load(b_desc, [0, 0], b_buffer.index(write_slot))
            b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K_PACKED_B])
        tdm.async_wait((NUM_BUFFERS - 3) * 2)
        with gl.amd.warp_pipeline_stage("stage1", priority=1):
            accumulator = gl.amd.gfx1250.wmma(a, b, accumulator)

    for _ in gl.static_range(NUM_BUFFERS - 1):
        with gl.amd.warp_pipeline_stage("stage0_epilogue", priority=0):
            read_slot = phase % NUM_BUFFERS
            a = a_buffer.index(read_slot).load(layout=dot_layout_a)
            b = b_buffer.index(read_slot).permute([1, 0]).load(layout=dot_layout_b)
            phase += 1
        tdm.async_wait(0)
        with gl.amd.warp_pipeline_stage("stage1_epilogue", priority=1):
            accumulator = gl.amd.gfx1250.wmma(a, b, accumulator)

    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, acc_layout))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, acc_layout))
    c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.gfx1250.buffer_store(accumulator, c_ptr, c_offs, mask=c_mask)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Standalone warp pipeline test requires gfx1250")
@pytest.mark.parametrize("KERNEL_MODE", ["baseline", "local_address", "sliced", "fp8_no_scale", "sliced_fp8_no_scale"])
def test_runtime_mxgemm_tdm_warp_pipeline_standalone(KERNEL_MODE):
    run_mxgemm_tdm_warp_pipeline_standalone(use_local_address=KERNEL_MODE == "local_address",
                                            use_sliced=KERNEL_MODE in ("sliced", "sliced_fp8_no_scale"),
                                            use_scales=KERNEL_MODE not in ("fp8_no_scale", "sliced_fp8_no_scale"),
                                            NUM_BUFFERS=2 if KERNEL_MODE in ("sliced", "sliced_fp8_no_scale") else 3)


def _validate_options(K, BLOCK_K, NUM_BUFFERS, NUM_WARPS, DTYPE_A, DTYPE_B, use_local_address, use_kernel_c,
                      use_partitioned_layout, use_sliced=False):
    if use_sliced and use_local_address:
        raise ValueError("--use-sliced is not compatible with --use-local-address")
    if use_sliced and use_kernel_c:
        raise ValueError("--use-sliced is not compatible with --kernelC")
    if use_sliced and use_partitioned_layout:
        raise ValueError("--use-sliced is not compatible with --partitioned-layout")
    if use_sliced and NUM_WARPS != 8:
        raise ValueError("--use-sliced requires NUM_WARPS == 8")
    if use_kernel_c and NUM_BUFFERS < 3:
        raise ValueError("kernelC requires NUM_BUFFERS >= 3")
    if use_kernel_c and NUM_WARPS != 8:
        raise ValueError("kernelC requires NUM_WARPS == 8 for its 4-warp TDM hint")
    if use_partitioned_layout:
        if use_local_address:
            raise ValueError("--partitioned-layout is only wired for the direct local_load path")
        if use_kernel_c:
            raise ValueError("--partitioned-layout is not compatible with --kernelC")
        if NUM_WARPS != 8:
            raise ValueError("--partitioned-layout requires NUM_WARPS == 8")
        if DTYPE_A not in ("float8_e4m3", "float8_e5m2") or DTYPE_B not in ("float8_e4m3", "float8_e5m2"):
            raise ValueError("--partitioned-layout is only wired for FP8 inputs")
    effective_num_buffers = 2 if use_sliced else NUM_BUFFERS
    if triton.cdiv(K, BLOCK_K) < effective_num_buffers:
        raise ValueError("K/BLOCK_K must be at least NUM_BUFFERS")


def _make_mxgemm_inputs(M, N, K, SCALE_BLOCK, DTYPE_A, DTYPE_B, use_scales=True, make_reference=True):
    torch_dtype = {"float8_e5m2": torch.float8_e5m2, "float8_e4m3": torch.float8_e4m3fn}

    a = torch.randint(20, 40, (M, K), dtype=torch.uint8).view(torch_dtype[DTYPE_A])
    b = torch.randint(20, 40, (K, N), dtype=torch.uint8).view(torch_dtype[DTYPE_B])
    c_ref = None
    if use_scales:
        a_scale = MXScaleTensor(size=(M, (K + SCALE_BLOCK - 1) // SCALE_BLOCK)).random(low=1.0, high=32.0)
        b_scale = MXScaleTensor(size=(N, (K + SCALE_BLOCK - 1) // SCALE_BLOCK)).random(low=1.0, high=32.0)

        if make_reference:
            a_scale_f32 = a_scale.to(torch.float32).repeat_interleave(SCALE_BLOCK, dim=1)[:M, :K]
            b_scale_f32 = b_scale.to(torch.float32).repeat_interleave(SCALE_BLOCK, dim=1).T.contiguous()[:K, :N]
            c_ref = torch.matmul(a.to(torch.float32) * a_scale_f32, b.to(torch.float32) * b_scale_f32).to(torch.float32)
    else:
        a_scale = None
        b_scale = None
        if make_reference:
            c_ref = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(torch.float32)

    c_d = torch.zeros(M, N, dtype=torch.float32).cuda()
    a_d = a.contiguous().cuda()
    b_d = b.T.contiguous().cuda()
    if use_scales:
        a_scale_d = a_scale.data.cuda()
        b_scale_d = b_scale.data.cuda()
    else:
        a_scale_d = torch.empty(1, dtype=torch.uint8, device="cuda")
        b_scale_d = torch.empty(1, dtype=torch.uint8, device="cuda")

    return a_d, b_d, c_d, a_scale_d, b_scale_d, c_ref


def _launch_mxgemm_tdm_warp_pipeline_standalone(a_d, b_d, c_d, a_scale_d, b_scale_d, use_local_address=False, M=512,
                                                N=512, K=512, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128,
                                                SCALE_BLOCK=32, GROUP_SIZE_M=8, NUM_BUFFERS=3, NUM_WARPS=8,
                                                DTYPE_A="float8_e4m3", DTYPE_B="float8_e5m2", use_scales=True,
                                                use_kernel_c=False, use_partitioned_layout=False, use_sliced=False):
    _validate_options(K, BLOCK_K, NUM_BUFFERS, NUM_WARPS, DTYPE_A, DTYPE_B, use_local_address, use_kernel_c,
                      use_partitioned_layout, use_sliced=use_sliced)

    stride_am, stride_ak = a_d.stride(0), a_d.stride(1)
    stride_bk, stride_bn = b_d.stride(1), b_d.stride(0)
    stride_cm, stride_cn = c_d.stride(0), c_d.stride(1)
    stride_scale = b_scale_d.stride(0) if use_scales else 0

    grid = [triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1]
    dtype_converter = {"float8_e5m2": "e5m2", "float8_e4m3": "e4m3"}
    tdm_warp_used_hint = 0b00001111 if use_kernel_c else 0
    effective_num_buffers = 2 if use_sliced else NUM_BUFFERS
    if use_sliced:
        if use_scales:
            return mxgemm_tdm_warp_pipeline_slice_mn_kernel[grid](
                a_d, b_d, c_d, a_scale_d, b_scale_d, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                stride_cn, stride_scale, dtype_converter[DTYPE_A], dtype_converter[DTYPE_B], SCALE_BLOCK, BLOCK_M,
                BLOCK_N, BLOCK_K, GROUP_SIZE_M, effective_num_buffers, NUM_WARPS, num_warps=NUM_WARPS, num_ctas=1,
                waves_per_eu=NUM_WARPS // 4)
        return fp8_gemm_tdm_warp_pipeline_slice_mn_kernel[grid](
            a_d, b_d, c_d, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M, BLOCK_N,
            BLOCK_K, GROUP_SIZE_M, effective_num_buffers, NUM_WARPS, num_warps=NUM_WARPS, num_ctas=1,
            waves_per_eu=NUM_WARPS // 4)
    if use_local_address:
        return mxgemm_tdm_warp_pipeline_local_address_kernel[grid](
            a_d, b_d, c_d, a_scale_d, b_scale_d, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
            stride_cn, stride_scale, dtype_converter[DTYPE_A], dtype_converter[DTYPE_B], SCALE_BLOCK, BLOCK_M, BLOCK_N,
            BLOCK_K, GROUP_SIZE_M, NUM_BUFFERS, NUM_WARPS, use_scales, tdm_warp_used_hint, num_warps=NUM_WARPS,
            num_ctas=1, waves_per_eu=NUM_WARPS // 4)
    return mxgemm_tdm_warp_pipeline_standalone_kernel[grid](
        a_d, b_d, c_d, a_scale_d, b_scale_d, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, dtype_converter[DTYPE_A], dtype_converter[DTYPE_B], SCALE_BLOCK, BLOCK_M, BLOCK_N, BLOCK_K,
        GROUP_SIZE_M, NUM_BUFFERS, NUM_WARPS, use_scales, tdm_warp_used_hint, use_partitioned_layout,
        num_warps=NUM_WARPS, num_ctas=1, waves_per_eu=NUM_WARPS // 4)


def run_mxgemm_tdm_warp_pipeline_standalone(use_local_address=False, M=512, N=512, K=512, BLOCK_M=128, BLOCK_N=128,
                                            BLOCK_K=128, SCALE_BLOCK=32, GROUP_SIZE_M=8, NUM_BUFFERS=3, NUM_WARPS=8,
                                            DTYPE_A="float8_e4m3", DTYPE_B="float8_e5m2", seed=0, use_scales=True,
                                            use_kernel_c=False, use_partitioned_layout=False, use_sliced=False):
    _validate_options(K, BLOCK_K, NUM_BUFFERS, NUM_WARPS, DTYPE_A, DTYPE_B, use_local_address, use_kernel_c,
                      use_partitioned_layout, use_sliced=use_sliced)
    torch.manual_seed(seed)
    a_d, b_d, c_d, a_scale_d, b_scale_d, c_ref = _make_mxgemm_inputs(M, N, K, SCALE_BLOCK, DTYPE_A, DTYPE_B,
                                                                      use_scales=use_scales)
    _launch_mxgemm_tdm_warp_pipeline_standalone(
        a_d, b_d, c_d, a_scale_d, b_scale_d, use_local_address=use_local_address, M=M, N=N, K=K, BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SCALE_BLOCK=SCALE_BLOCK, GROUP_SIZE_M=GROUP_SIZE_M, NUM_BUFFERS=NUM_BUFFERS,
        NUM_WARPS=NUM_WARPS, DTYPE_A=DTYPE_A, DTYPE_B=DTYPE_B, use_scales=use_scales, use_kernel_c=use_kernel_c,
        use_partitioned_layout=use_partitioned_layout, use_sliced=use_sliced)

    torch.testing.assert_close(c_d.cpu(), c_ref.cpu(), rtol=1e-4 if not use_scales else 1e-5, atol=1e-4 if not use_scales else 1e-8)
    kernel_mode = "kernelC" if use_kernel_c else "default"
    print(f"Pass mode={kernel_mode} use_local_address={use_local_address} use_scales={use_scales} "
          f"use_partitioned_layout={use_partitioned_layout} use_sliced={use_sliced}")


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


def benchmark_mxgemm_tdm_warp_pipeline_standalone(use_local_address=False, M=512, N=512, K=512, BLOCK_M=128,
                                                  BLOCK_N=128, BLOCK_K=128, SCALE_BLOCK=32, GROUP_SIZE_M=8,
                                                  NUM_BUFFERS=3, NUM_WARPS=8, DTYPE_A="float8_e4m3",
                                                  DTYPE_B="float8_e5m2", seed=0, use_scales=True,
                                                  use_kernel_c=False, warmup=10, probe_iters=20,
                                                  graph_ms=100.0, n_replays=20, iters_per_graph=None,
                                                  use_partitioned_layout=False, use_sliced=False):
    _validate_options(K, BLOCK_K, NUM_BUFFERS, NUM_WARPS, DTYPE_A, DTYPE_B, use_local_address, use_kernel_c,
                      use_partitioned_layout, use_sliced=use_sliced)
    torch.manual_seed(seed)
    a_d, b_d, c_d, a_scale_d, b_scale_d, _ = _make_mxgemm_inputs(
        M, N, K, SCALE_BLOCK, DTYPE_A, DTYPE_B, use_scales=use_scales, make_reference=False)

    def launch():
        return _launch_mxgemm_tdm_warp_pipeline_standalone(
            a_d, b_d, c_d, a_scale_d, b_scale_d, use_local_address=use_local_address, M=M, N=N, K=K, BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SCALE_BLOCK=SCALE_BLOCK, GROUP_SIZE_M=GROUP_SIZE_M,
            NUM_BUFFERS=NUM_BUFFERS, NUM_WARPS=NUM_WARPS, DTYPE_A=DTYPE_A, DTYPE_B=DTYPE_B, use_scales=use_scales,
            use_kernel_c=use_kernel_c, use_partitioned_layout=use_partitioned_layout, use_sliced=use_sliced)

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


if __name__ == "__main__":
    import argparse

    supported_dtypes = ("float8_e4m3", "float8_e5m2")
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-local-address", action="store_true", help="Precompute LDS addresses before local_load.")
    parser.add_argument("--use-sliced", action="store_true", help="Use the 2x2 SliceMN warp-pipeline prototype.")
    parser.add_argument("--no-scales", action="store_true", help="Skip MX scale copies, loads, and scaled operands.")
    parser.add_argument("--use-fp8-no-scale", action="store_true", help="Alias for --no-scales.")
    parser.add_argument("--kernelC", action="store_true", help="Use partial TDM load issue.")
    parser.add_argument("--partitioned-layout", action="store_true",
                        help="Use partitioned shared layout for the default FP8 direct-load path.")
    parser.add_argument("-M", type=int, default=512)
    parser.add_argument("-N", type=int, default=512)
    parser.add_argument("-K", type=int, default=512)
    parser.add_argument("-BM", "--block-m", type=int, default=128)
    parser.add_argument("-BN", "--block-n", type=int, default=128)
    parser.add_argument("-BK", "--block-k", type=int, default=128)
    parser.add_argument("--scale-block", type=int, default=32)
    parser.add_argument("--group-size-m", type=int, default=8)
    parser.add_argument("--num-buffers", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--num-warps", type=int, default=8, choices=[8])
    parser.add_argument("--dtype-a", type=str, default="float8_e4m3", choices=supported_dtypes)
    parser.add_argument("--dtype-b", type=str, default="float8_e5m2", choices=supported_dtypes)
    parser.add_argument("--seed", type=int, default=0)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--benchmark", action="store_true", help="Benchmark the kernel and report TFLOPS.")
    mode_group.add_argument("--check", action="store_true", help="Check output against torch.")
    parser.add_argument("--warmup", type=int, default=10, help="Benchmark warmup iterations.")
    parser.add_argument("--probe-iters", type=int, default=20, help="Iterations for CUDA event timing probe.")
    parser.add_argument("--graph-ms", type=float, default=100.0, help="Target CUDA graph body duration in ms.")
    parser.add_argument("--n-replays", type=int, default=20, help="Number of CUDA graph replays to time.")
    parser.add_argument("--iters-per-graph", type=int, default=None, help="Override graph body iteration count.")
    args = parser.parse_args()

    use_scales = not (args.no_scales or args.use_fp8_no_scale)
    print(
        f"(M={args.M}, N={args.N}, K={args.K}), (BLOCK_M={args.block_m}, BLOCK_N={args.block_n}, BLOCK_K={args.block_k}), "
        f"NUM_WARPS={args.num_warps}, NUM_BUFFERS={args.num_buffers}, use_local_address={args.use_local_address}, "
        f"use_scales={use_scales}, kernelC={args.kernelC}, partitioned_layout={args.partitioned_layout}, "
        f"use_sliced={args.use_sliced}, "
        f"DTYPE_A={args.dtype_a}, DTYPE_B={args.dtype_b}")
    if args.benchmark:
        benchmark_mxgemm_tdm_warp_pipeline_standalone(
            use_local_address=args.use_local_address, M=args.M, N=args.N, K=args.K, BLOCK_M=args.block_m,
            BLOCK_N=args.block_n, BLOCK_K=args.block_k, SCALE_BLOCK=args.scale_block, GROUP_SIZE_M=args.group_size_m,
            NUM_BUFFERS=args.num_buffers, NUM_WARPS=args.num_warps, DTYPE_A=args.dtype_a, DTYPE_B=args.dtype_b,
            seed=args.seed, use_scales=use_scales, use_kernel_c=args.kernelC, warmup=args.warmup,
            probe_iters=args.probe_iters, graph_ms=args.graph_ms, n_replays=args.n_replays,
            iters_per_graph=args.iters_per_graph, use_partitioned_layout=args.partitioned_layout,
            use_sliced=args.use_sliced)
    else:
        run_mxgemm_tdm_warp_pipeline_standalone(
            use_local_address=args.use_local_address, M=args.M, N=args.N, K=args.K, BLOCK_M=args.block_m,
            BLOCK_N=args.block_n, BLOCK_K=args.block_k, SCALE_BLOCK=args.scale_block, GROUP_SIZE_M=args.group_size_m,
            NUM_BUFFERS=args.num_buffers, NUM_WARPS=args.num_warps, DTYPE_A=args.dtype_a, DTYPE_B=args.dtype_b,
            seed=args.seed, use_scales=use_scales, use_kernel_c=args.kernelC,
            use_partitioned_layout=args.partitioned_layout, use_sliced=args.use_sliced)

