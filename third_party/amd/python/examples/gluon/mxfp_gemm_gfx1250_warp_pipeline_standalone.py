import pytest
import torch
import triton
from triton._internal_testing import is_hip_gfx1250
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm
from triton.tools.mxfp import MXScaleTensor


@gluon.jit
def mxgemm_tdm_warp_pipeline_standalone_kernel(a_ptr, b_ptr, c_ptr, a_scale, b_scale, M, N, K, stride_am, stride_ak,
                                               stride_bk, stride_bn, stride_cm, stride_cn, stride_scale,
                                               DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr,
                                               SCALE_BLOCK: gl.constexpr, BLOCK_M: gl.constexpr,
                                               BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                                               GROUP_SIZE_M: gl.constexpr, NUM_BUFFERS: gl.constexpr,
                                               NUM_WARPS: gl.constexpr):
    DIV_FACTOR_A: gl.constexpr = 2 if DTYPE_A == "e2m1" else 1
    DIV_FACTOR_B: gl.constexpr = 2 if DTYPE_B == "e2m1" else 1
    NUM_LOADS_IN_BATCH: gl.constexpr = 4
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // SCALE_BLOCK
    BLOCK_K_PACKED_A: gl.constexpr = BLOCK_K // DIV_FACTOR_A
    BLOCK_K_PACKED_B: gl.constexpr = BLOCK_K // DIV_FACTOR_B

    gl.static_assert(NUM_WARPS == 4 or NUM_WARPS == 8)
    INSTR_M: gl.constexpr = 32 if (DTYPE_A == "e2m1" and DTYPE_B == "e2m1") else 16
    REG_BASES: gl.constexpr = []
    TILES_PER_WARP: gl.constexpr = 1
    if NUM_WARPS == 4:
        WARP_BASES: gl.constexpr = [[0, TILES_PER_WARP], [TILES_PER_WARP, 0]]
    else:
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
    a_scale_buffer = gl.allocate_shared_memory(a_scale_desc.dtype, shape=[NUM_BUFFERS] + a_scale_desc.block_shape,
                                               layout=a_scale_desc.layout)
    b_scale_buffer = gl.allocate_shared_memory(b_scale_desc.dtype, shape=[NUM_BUFFERS] + b_scale_desc.block_shape,
                                               layout=b_scale_desc.layout)

    load_idx = 0
    wmma_idx = 0

    # Standalone copy of MXFPGEMMPipelinedProgram.warp_pipeline, with the issue_* helper bodies inlined.
    for _ in gl.static_range(NUM_BUFFERS - 1):
        slot = load_idx % NUM_BUFFERS
        tdm.async_load(a_scale_desc, [0, 0], a_scale_buffer.index(slot))
        a_scale_desc = tdm.update_tensor_descriptor(a_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
        tdm.async_load(b_scale_desc, [0, 0], b_scale_buffer.index(slot))
        b_scale_desc = tdm.update_tensor_descriptor(b_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
        tdm.async_load(a_desc, [0, 0], a_buffer.index(slot))
        a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K_PACKED_A])
        tdm.async_load(b_desc, [0, 0], b_buffer.index(slot))
        b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K_PACKED_B])
        load_idx = load_idx + 1

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=acc_layout)
    loop_ub = gl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)
    #tdm.async_wait((NUM_BUFFERS - 2) * NUM_LOADS_IN_BATCH)
    tdm.async_wait(0)
    gl.assume(loop_ub >= 0)
    for _ in range(0, loop_ub):
        with gl.amd.warp_pipeline_stage("tdm+lds", priority=1):
            a = a_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=dot_layout_a)
            b = b_buffer.index(wmma_idx % NUM_BUFFERS).permute([1, 0]).load(layout=dot_layout_b)
            a_scale_buffer_slice = a_scale_buffer.index(wmma_idx % NUM_BUFFERS)
            b_scale_buffer_slice = b_scale_buffer.index(wmma_idx % NUM_BUFFERS)
            scale_a = a_scale_buffer_slice.load(layout=layout_a_scale)
            scale_b = b_scale_buffer_slice.load(layout=layout_b_scale)

            wmma_idx += 1
            phase = wmma_idx + NUM_BUFFERS - 2
            slot = phase % NUM_BUFFERS
            tdm.async_load(a_scale_desc, [0, 0], a_scale_buffer.index(slot))
            a_scale_desc = tdm.update_tensor_descriptor(a_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
            tdm.async_load(b_scale_desc, [0, 0], b_scale_buffer.index(slot))
            b_scale_desc = tdm.update_tensor_descriptor(b_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
            tdm.async_load(a_desc, [0, 0], a_buffer.index(slot))
            a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K_PACKED_A])
            tdm.async_load(b_desc, [0, 0], b_buffer.index(slot))
            b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K_PACKED_B])

        #tdm.async_wait((NUM_BUFFERS - 2) * NUM_LOADS_IN_BATCH)
        tdm.async_wait(0)
        with gl.amd.warp_pipeline_stage("wmma", priority=0):
            load_idx = load_idx + 1
            accumulator = gl.amd.gfx1250.wmma_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator)

    for i in gl.static_range(NUM_BUFFERS - 1):
        #tdm.async_wait((NUM_BUFFERS - 1 - i) * NUM_LOADS_IN_BATCH)
        tdm.async_wait(0)
        a = a_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=dot_layout_a)
        b = b_buffer.index(wmma_idx % NUM_BUFFERS).permute([1, 0]).load(layout=dot_layout_b)
        a_scale_buffer_slice = a_scale_buffer.index(wmma_idx % NUM_BUFFERS)
        b_scale_buffer_slice = b_scale_buffer.index(wmma_idx % NUM_BUFFERS)
        scale_a = a_scale_buffer_slice.load(layout=layout_a_scale)
        scale_b = b_scale_buffer_slice.load(layout=layout_b_scale)
        wmma_idx += 1
        accumulator = gl.amd.gfx1250.wmma_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator)

    gl.amd.gfx1250.buffer_store(accumulator, c_ptr, c_offs, mask=c_mask)
    tdm.async_wait(0)


@gluon.jit
def mxgemm_tdm_warp_pipeline_local_address_kernel(a_ptr, b_ptr, c_ptr, a_scale, b_scale, M, N, K: gl.constexpr,
                                                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                                                  stride_scale, DTYPE_A: gl.constexpr, DTYPE_B: gl.constexpr,
                                                  SCALE_BLOCK: gl.constexpr, BLOCK_M: gl.constexpr,
                                                  BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                                                  GROUP_SIZE_M: gl.constexpr, NUM_BUFFERS: gl.constexpr,
                                                  NUM_WARPS: gl.constexpr):
    DIV_FACTOR_A: gl.constexpr = 2 if DTYPE_A == "e2m1" else 1
    DIV_FACTOR_B: gl.constexpr = 2 if DTYPE_B == "e2m1" else 1
    NUM_LOADS_IN_BATCH: gl.constexpr = 4
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // SCALE_BLOCK
    BLOCK_K_PACKED_A: gl.constexpr = BLOCK_K // DIV_FACTOR_A
    BLOCK_K_PACKED_B: gl.constexpr = BLOCK_K // DIV_FACTOR_B

    gl.static_assert(NUM_WARPS == 4 or NUM_WARPS == 8)
    INSTR_M: gl.constexpr = 32 if (DTYPE_A == "e2m1" and DTYPE_B == "e2m1") else 16
    REG_BASES: gl.constexpr = []
    TILES_PER_WARP: gl.constexpr = 1
    if NUM_WARPS == 4:
        WARP_BASES: gl.constexpr = [[0, TILES_PER_WARP], [TILES_PER_WARP, 0]]
    else:
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
    a_scale_buffer = gl.allocate_shared_memory(a_scale_desc.dtype, shape=[NUM_BUFFERS] + a_scale_desc.block_shape,
                                               layout=a_scale_desc.layout)
    b_scale_buffer = gl.allocate_shared_memory(b_scale_desc.dtype, shape=[NUM_BUFFERS] + b_scale_desc.block_shape,
                                               layout=b_scale_desc.layout)

    load_idx = 0
    wmma_idx = 0

    for _ in gl.static_range(NUM_BUFFERS - 1):
        slot = load_idx % NUM_BUFFERS
        tdm.async_load(a_scale_desc, [0, 0], a_scale_buffer.index(slot))
        a_scale_desc = tdm.update_tensor_descriptor(a_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
        tdm.async_load(b_scale_desc, [0, 0], b_scale_buffer.index(slot))
        b_scale_desc = tdm.update_tensor_descriptor(b_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
        tdm.async_load(a_desc, [0, 0], a_buffer.index(slot))
        a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K_PACKED_A])
        tdm.async_load(b_desc, [0, 0], b_buffer.index(slot))
        b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K_PACKED_B])
        load_idx = load_idx + 1

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=acc_layout)
    LOOP_UB: gl.constexpr = (K + BLOCK_K - 1) // BLOCK_K - (NUM_BUFFERS - 1)
    tdm.async_wait(0)
    gl.static_assert(LOOP_UB >= 0)
    a_addr = a_buffer.index(wmma_idx % NUM_BUFFERS).local_address(dot_layout_a)
    b_addr = b_buffer.index(wmma_idx % NUM_BUFFERS).permute([1, 0]).local_address(dot_layout_b)
    scale_a_addr = a_scale_buffer.index(wmma_idx % NUM_BUFFERS).local_address(layout_a_scale)
    scale_b_addr = b_scale_buffer.index(wmma_idx % NUM_BUFFERS).local_address(layout_b_scale)

    for _ in gl.static_range(LOOP_UB):
        with gl.amd.warp_pipeline_stage("tdm+lds", priority=1):
            a = a_addr.load()
            b = b_addr.load()
            scale_a = scale_a_addr.load()
            scale_b = scale_b_addr.load()
            wmma_idx += 1
            phase = wmma_idx + NUM_BUFFERS - 2
            slot = phase % NUM_BUFFERS
            tdm.async_load(a_scale_desc, [0, 0], a_scale_buffer.index(slot))
            a_scale_desc = tdm.update_tensor_descriptor(a_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
            tdm.async_load(b_scale_desc, [0, 0], b_scale_buffer.index(slot))
            b_scale_desc = tdm.update_tensor_descriptor(b_scale_desc, add_offsets=[0, BLOCK_K_SCALE])
            tdm.async_load(a_desc, [0, 0], a_buffer.index(slot))
            a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K_PACKED_A])
            tdm.async_load(b_desc, [0, 0], b_buffer.index(slot))
            b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K_PACKED_B])

        tdm.async_wait(0)
        with gl.amd.warp_pipeline_stage("wmma", priority=0):
            load_idx = load_idx + 1
            accumulator = gl.amd.gfx1250.wmma_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator)
            a_addr = a_buffer.index(wmma_idx % NUM_BUFFERS).local_address(dot_layout_a)
            b_addr = b_buffer.index(wmma_idx % NUM_BUFFERS).permute([1, 0]).local_address(dot_layout_b)
            scale_a_addr = a_scale_buffer.index(wmma_idx % NUM_BUFFERS).local_address(layout_a_scale)
            scale_b_addr = b_scale_buffer.index(wmma_idx % NUM_BUFFERS).local_address(layout_b_scale)

    for i in gl.static_range(NUM_BUFFERS - 1):
        tdm.async_wait(0)
        a = a_addr.load()
        b = b_addr.load()
        scale_a = scale_a_addr.load()
        scale_b = scale_b_addr.load()
        wmma_idx += 1
        accumulator = gl.amd.gfx1250.wmma_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator)
        if i != NUM_BUFFERS - 2:
            a_addr = a_buffer.index(wmma_idx % NUM_BUFFERS).local_address(dot_layout_a)
            b_addr = b_buffer.index(wmma_idx % NUM_BUFFERS).permute([1, 0]).local_address(dot_layout_b)
            scale_a_addr = a_scale_buffer.index(wmma_idx % NUM_BUFFERS).local_address(layout_a_scale)
            scale_b_addr = b_scale_buffer.index(wmma_idx % NUM_BUFFERS).local_address(layout_b_scale)

    gl.amd.gfx1250.buffer_store(accumulator, c_ptr, c_offs, mask=c_mask)
    tdm.async_wait(0)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Standalone warp pipeline test requires gfx1250")
@pytest.mark.parametrize("USE_LOCAL_ADDRESS", [False, True])
def test_runtime_mxgemm_tdm_warp_pipeline_standalone(USE_LOCAL_ADDRESS):
    run_mxgemm_tdm_warp_pipeline_standalone(use_local_address=USE_LOCAL_ADDRESS)


def run_mxgemm_tdm_warp_pipeline_standalone(use_local_address=False, M=512, N=512, K=512, BLOCK_M=128, BLOCK_N=128,
                                            BLOCK_K=128, SCALE_BLOCK=32, GROUP_SIZE_M=8, NUM_BUFFERS=3, NUM_WARPS=8,
                                            DTYPE_A="float8_e4m3", DTYPE_B="float8_e5m2", seed=0):
    torch.manual_seed(seed)
    torch_dtype = {"float8_e5m2": torch.float8_e5m2, "float8_e4m3": torch.float8_e4m3fn}

    a = torch.randint(20, 40, (M, K), dtype=torch.uint8).view(torch_dtype[DTYPE_A])
    b = torch.randint(20, 40, (K, N), dtype=torch.uint8).view(torch_dtype[DTYPE_B])
    a_scale = MXScaleTensor(size=(M, (K + SCALE_BLOCK - 1) // SCALE_BLOCK)).random(low=1.0, high=32.0)
    b_scale = MXScaleTensor(size=(N, (K + SCALE_BLOCK - 1) // SCALE_BLOCK)).random(low=1.0, high=32.0)

    a_scale_f32 = a_scale.to(torch.float32).repeat_interleave(SCALE_BLOCK, dim=1)[:M, :K]
    b_scale_f32 = b_scale.to(torch.float32).repeat_interleave(SCALE_BLOCK, dim=1).T.contiguous()[:K, :N]
    c_ref = torch.matmul(a.to(torch.float32) * a_scale_f32, b.to(torch.float32) * b_scale_f32).to(torch.float32)

    c_d = torch.zeros(M, N, dtype=torch.float32).cuda()
    a_d = a.contiguous().cuda()
    b_d = b.T.contiguous().cuda()
    a_scale_d = a_scale.data.cuda()
    b_scale_d = b_scale.data.cuda()

    stride_am, stride_ak = a_d.stride(0), a_d.stride(1)
    stride_bk, stride_bn = b_d.stride(1), b_d.stride(0)
    stride_cm, stride_cn = c_d.stride(0), c_d.stride(1)
    stride_scale = b_scale_d.stride(0)

    grid = [triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1]
    dtype_converter = {"float8_e5m2": "e5m2", "float8_e4m3": "e4m3"}
    if use_local_address:
        mxgemm_tdm_warp_pipeline_local_address_kernel[grid](
            a_d, b_d, c_d, a_scale_d, b_scale_d, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
            stride_cn, stride_scale, dtype_converter[DTYPE_A], dtype_converter[DTYPE_B], SCALE_BLOCK, BLOCK_M, BLOCK_N,
            BLOCK_K, GROUP_SIZE_M, NUM_BUFFERS, NUM_WARPS, num_warps=NUM_WARPS, num_ctas=1,
            waves_per_eu=NUM_WARPS // 4)
    else:
        mxgemm_tdm_warp_pipeline_standalone_kernel[grid](
            a_d, b_d, c_d, a_scale_d, b_scale_d, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
            stride_cn, stride_scale, dtype_converter[DTYPE_A], dtype_converter[DTYPE_B], SCALE_BLOCK, BLOCK_M, BLOCK_N,
            BLOCK_K, GROUP_SIZE_M, NUM_BUFFERS, NUM_WARPS, num_warps=NUM_WARPS, num_ctas=1,
            waves_per_eu=NUM_WARPS // 4)

    torch.testing.assert_close(c_d.cpu(), c_ref.cpu(), rtol=1e-5, atol=1e-8)
    print(f"Pass use_local_address={use_local_address}")


if __name__ == "__main__":
    import argparse

    supported_dtypes = ("float8_e4m3", "float8_e5m2")
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-local-address", action="store_true", help="Precompute LDS addresses before local_load.")
    parser.add_argument("-M", type=int, default=512)
    parser.add_argument("-N", type=int, default=512)
    parser.add_argument("-K", type=int, default=512)
    parser.add_argument("-BM", "--block-m", type=int, default=128)
    parser.add_argument("-BN", "--block-n", type=int, default=128)
    parser.add_argument("-BK", "--block-k", type=int, default=128)
    parser.add_argument("--scale-block", type=int, default=32)
    parser.add_argument("--group-size-m", type=int, default=8)
    parser.add_argument("--num-buffers", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--num-warps", type=int, default=8, choices=[4, 8])
    parser.add_argument("--dtype-a", type=str, default="float8_e4m3", choices=supported_dtypes)
    parser.add_argument("--dtype-b", type=str, default="float8_e5m2", choices=supported_dtypes)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_mxgemm_tdm_warp_pipeline_standalone(use_local_address=args.use_local_address, M=args.M, N=args.N, K=args.K,
                                            BLOCK_M=args.block_m, BLOCK_N=args.block_n, BLOCK_K=args.block_k,
                                            SCALE_BLOCK=args.scale_block, GROUP_SIZE_M=args.group_size_m,
                                            NUM_BUFFERS=args.num_buffers, NUM_WARPS=args.num_warps,
                                            DTYPE_A=args.dtype_a, DTYPE_B=args.dtype_b, seed=args.seed)

