"""Five-stage MXFP8 shape benchmark, derived from kernels_best_0927.

BK256 retains parent-B, two data slots, three scale slots, and unroll six.
BK512 uses full accumulators, two slots, five stages, and unroll six.
Inputs match AITER/FlyDSL positive uniform E4M3 with block-128 E8M0 scales.
Auto output stores use two LDS buffers on the nine measured BK256 shapes;
the three BK512 shapes retain their original full-tile store.
"""
from kernels_best_0927 import *
import json
import statistics

@gluon.jit
def mxfp8_load_b2_parent(
        buffer, slot, start_n: gl.constexpr, start_k: gl.constexpr,
        B2_LOAD_LAYOUT: gl.constexpr, DOT_B: gl.constexpr):
    rank3 = buffer.index(slot).slice(
        start_n, 128, 1).slice(
        start_k, 128, 2).permute([2, 0, 1]).load(
        layout=B2_LOAD_LAYOUT)
    rank2 = rank3.reshape((128, buffer.type.shape[1] * 128))
    return gl.convert_layout(rank2, DOT_B)

@gluon.jit
def mxfp8_bk256_b2_consume_and_refill(
        a_buf, b0_buf, b1_buf, as_buf, bs_buf,
        data_slot, scale_slot, refill_scale_slot,
        a_desc, b0_desc, b1_desc, as_desc, bs_desc,
        acc0, acc1, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        B2_LOAD_LAYOUT: gl.constexpr,
        B2_FRAGMENT_LOAD_LAYOUT: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_A_LOAD: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr, SCALE_B_LOAD: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_N: gl.constexpr, REFILL_SCALE: gl.constexpr,
        TWO_TDM: gl.constexpr, WAITCNT: gl.constexpr,
        DATA_SCALE_WAITCNT: gl.constexpr,
        FRAGMENT_LOADS: gl.constexpr, SPREAD_ADVANCE: gl.constexpr,
        PARENT_B: gl.constexpr, WMMA_BEFORE_REFILL: gl.constexpr):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    tdm.async_wait(WAITCNT if TWO_TDM else DATA_SCALE_WAITCNT)

    next_a_desc = a_desc
    next_b0_desc = b0_desc
    next_b1_desc = b1_desc
    next_as_desc = as_desc
    next_bs_desc = bs_desc
    with gl.amd.warp_pipeline_stage(
            "b2_stage0_load_low_n0", phase_gap=1):
        as0 = mxfp8_load_scale(
            as_buf, scale_slot, 0, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as0 = gl.convert_layout(as0, SCALE_A_LAYOUT, assert_trivial=True)
        bs0 = mxfp8_load_scale(
            bs_buf, scale_slot, 0, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs0_grid = bs0.reshape((CTA_N, cta_n, 4))
        bs0_lo = gl.convert_layout(gl.amd.slice(
            bs0_grid, [CTA_N, half_n, 4], [0, 0, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        bs0_hi = gl.convert_layout(gl.amd.slice(
            bs0_grid, [CTA_N, half_n, 4], [0, half_n, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        if FRAGMENT_LOADS:
            a00 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(
                0, 32, 1).load(layout=DOT_A)
            b00 = mxfp8_load_b2_fragment(
                b0_buf, data_slot, 0, B2_FRAGMENT_LOAD_LAYOUT, DOT_B)
            a01 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(
                32, 32, 1).load(layout=DOT_A)
            b01 = mxfp8_load_b2_fragment(
                b0_buf, data_slot, 32, B2_FRAGMENT_LOAD_LAYOUT, DOT_B)
            a02 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(
                64, 32, 1).load(layout=DOT_A)
            b02 = mxfp8_load_b2_fragment(
                b0_buf, data_slot, 64, B2_FRAGMENT_LOAD_LAYOUT, DOT_B)
            a03 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(
                96, 32, 1).load(layout=DOT_A)
            b03 = mxfp8_load_b2_fragment(
                b0_buf, data_slot, 96, B2_FRAGMENT_LOAD_LAYOUT, DOT_B)
            a0_lo = gl.join(a00, a01).permute(0, 2, 1).reshape(
                (BLOCK_M, 64))
            a0_hi = gl.join(a02, a03).permute(0, 2, 1).reshape(
                (BLOCK_M, 64))
            a0 = gl.join(a0_lo, a0_hi).permute(0, 2, 1).reshape(
                (BLOCK_M, 128))
            a0 = gl.convert_layout(a0, DOT_A, assert_trivial=True)
            b0_lo0 = gl.join(b00, b01).permute(2, 0, 1).reshape(
                (64, packed_n))
            b0_lo1 = gl.join(b02, b03).permute(2, 0, 1).reshape(
                (64, packed_n))
            b0_lo = gl.join(b0_lo0, b0_lo1).permute(2, 0, 1).reshape(
                (128, packed_n))
            b0_lo = gl.convert_layout(
                b0_lo, DOT_B, assert_trivial=True)
        else:
            a0 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(
                0, 128, 1).load(layout=DOT_A)
            if PARENT_B:
                b0_lo = mxfp8_load_b2_parent(
                    b0_buf, data_slot, 0, 0, B2_LOAD_LAYOUT, DOT_B)
            else:
                b0_lo = mxfp8_load_b2(
                    b0_buf, data_slot, 0, B2_LOAD_LAYOUT, DOT_B)
        if SPREAD_ADVANCE:
            next_a_desc = tdm.update_tensor_descriptor(
                a_desc, add_offsets=[0, 256])

        if PARENT_B:
            b0_hi = mxfp8_load_b2_parent(
                b0_buf, data_slot, 128, 0, B2_LOAD_LAYOUT, DOT_B)
        else:
            b0_hi = mxfp8_load_b2(
                b1_buf, data_slot, 0, B2_LOAD_LAYOUT, DOT_B)
        if SPREAD_ADVANCE:
            next_b0_desc = tdm.update_tensor_descriptor(
                b0_desc, add_offsets=[0, 0, 256])

    with gl.amd.warp_pipeline_stage("b2_stage1_compute_low_n0"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0_lo, bs0_lo, "e4m3", acc0)

        acc1 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0_hi, bs0_hi, "e4m3", acc1)

    with gl.amd.warp_pipeline_stage("b2_stage4_load_high"):
        if not FRAGMENT_LOADS:
            a1 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(
                128, 128, 1).load(layout=DOT_A)
        as1 = mxfp8_load_scale(
            as_buf, scale_slot, 4, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as1 = gl.convert_layout(as1, SCALE_A_LAYOUT, assert_trivial=True)
        bs1 = mxfp8_load_scale(
            bs_buf, scale_slot, 4, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs1_grid = bs1.reshape((CTA_N, cta_n, 4))
        bs1_lo = gl.convert_layout(gl.amd.slice(
            bs1_grid, [CTA_N, half_n, 4], [0, 0, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        bs1_hi = gl.convert_layout(gl.amd.slice(
            bs1_grid, [CTA_N, half_n, 4], [0, half_n, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        if FRAGMENT_LOADS:
            a10 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(
                128, 32, 1).load(layout=DOT_A)
            b10 = mxfp8_load_b2_fragment(
                b0_buf, data_slot, 128, B2_FRAGMENT_LOAD_LAYOUT, DOT_B)
            a11 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(
                160, 32, 1).load(layout=DOT_A)
            b11 = mxfp8_load_b2_fragment(
                b0_buf, data_slot, 160, B2_FRAGMENT_LOAD_LAYOUT, DOT_B)
            a12 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(
                192, 32, 1).load(layout=DOT_A)
            b12 = mxfp8_load_b2_fragment(
                b0_buf, data_slot, 192, B2_FRAGMENT_LOAD_LAYOUT, DOT_B)
            a13 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(
                224, 32, 1).load(layout=DOT_A)
            b13 = mxfp8_load_b2_fragment(
                b0_buf, data_slot, 224, B2_FRAGMENT_LOAD_LAYOUT, DOT_B)
            a1_lo = gl.join(a10, a11).permute(0, 2, 1).reshape(
                (BLOCK_M, 64))
            a1_hi = gl.join(a12, a13).permute(0, 2, 1).reshape(
                (BLOCK_M, 64))
            a1 = gl.join(a1_lo, a1_hi).permute(0, 2, 1).reshape(
                (BLOCK_M, 128))
            a1 = gl.convert_layout(a1, DOT_A, assert_trivial=True)
            b1_lo0 = gl.join(b10, b11).permute(2, 0, 1).reshape(
                (64, packed_n))
            b1_lo1 = gl.join(b12, b13).permute(2, 0, 1).reshape(
                (64, packed_n))
            b1_lo = gl.join(b1_lo0, b1_lo1).permute(2, 0, 1).reshape(
                (128, packed_n))
            b1_lo = gl.convert_layout(
                b1_lo, DOT_B, assert_trivial=True)
            b1_hi = mxfp8_load_b2(
                b1_buf, data_slot, 128, B2_LOAD_LAYOUT, DOT_B)
        elif PARENT_B:
            b1_lo = mxfp8_load_b2_parent(
                b0_buf, data_slot, 0, 128, B2_LOAD_LAYOUT, DOT_B)
            b1_hi = mxfp8_load_b2_parent(
                b0_buf, data_slot, 128, 128, B2_LOAD_LAYOUT, DOT_B)
        else:
            b1_lo = mxfp8_load_b2(
                b0_buf, data_slot, 128, B2_LOAD_LAYOUT, DOT_B)
            b1_hi = mxfp8_load_b2(
                b1_buf, data_slot, 128, B2_LOAD_LAYOUT, DOT_B)
        if SPREAD_ADVANCE and not PARENT_B:
            next_b1_desc = tdm.update_tensor_descriptor(
                b1_desc, add_offsets=[0, 0, 256])
        if SPREAD_ADVANCE:
            if REFILL_SCALE:
                next_as_desc = tdm.update_tensor_descriptor(
                    as_desc, add_offsets=[0, 1024])
                next_bs_desc = tdm.update_tensor_descriptor(
                    bs_desc, add_offsets=[0, 1024])

        gl.amd.gfx1250.cluster.arrive()
        if not SPREAD_ADVANCE:
            next_a_desc = tdm.update_tensor_descriptor(
                a_desc, add_offsets=[0, 256])
            next_b0_desc = tdm.update_tensor_descriptor(
                b0_desc, add_offsets=[0, 0, 256])
            if not PARENT_B:
                next_b1_desc = tdm.update_tensor_descriptor(
                    b1_desc, add_offsets=[0, 0, 256])
            if REFILL_SCALE:
                next_as_desc = tdm.update_tensor_descriptor(
                    as_desc, add_offsets=[0, 1024])
                next_bs_desc = tdm.update_tensor_descriptor(
                    bs_desc, add_offsets=[0, 1024])

    with gl.amd.warp_pipeline_stage("b2_stage5_compute_high_both"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1_lo, bs1_lo, "e4m3", acc0)

        acc1 = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1_hi, bs1_hi, "e4m3", acc1)

    with gl.amd.warp_pipeline_stage("b2_stage6_refill"):
        gl.amd.gfx1250.cluster.wait()
        if PARENT_B:
            mxfp8_issue_b2_parent_data_without_update(
                a_desc, b0_desc, a_buf, b0_buf, data_slot)
            if REFILL_SCALE:
                mxfp8_issue_leading4_scale_without_update(
                    as_desc, bs_desc, as_buf, bs_buf, refill_scale_slot)
        elif TWO_TDM:
            mxfp8_issue_b2_two_tdm_without_update(
                a_desc, b0_desc, b1_desc, as_desc, bs_desc,
                a_buf, b0_buf, b1_buf, as_buf, bs_buf,
                data_slot, refill_scale_slot, REFILL_SCALE)
        else:
            mxfp8_issue_b2_data_without_update(
                a_desc, b0_desc, b1_desc, a_buf, b0_buf, b1_buf, data_slot)
            if REFILL_SCALE:
                mxfp8_issue_leading4_scale_without_update(
                    as_desc, bs_desc, as_buf, bs_buf, refill_scale_slot)
        a_desc = next_a_desc
        b0_desc = next_b0_desc
        b1_desc = next_b1_desc
        as_desc = next_as_desc
        bs_desc = next_bs_desc
    return a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1

@gluon.jit
def mxfp8_bk256_b2_consume_tail(
        a_buf, b0_buf, b1_buf, as_buf, bs_buf,
        data_slot, scale_slot, acc0, acc1,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        B2_LOAD_LAYOUT: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_A_LOAD: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr, SCALE_B_LOAD: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_N: gl.constexpr, PARENT_B: gl.constexpr):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    with gl.amd.warp_pipeline_stage("b2_tail_load_low"):
        a0 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        as0 = mxfp8_load_scale(
            as_buf, scale_slot, 0, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as0 = gl.convert_layout(as0, SCALE_A_LAYOUT, assert_trivial=True)
        bs0 = mxfp8_load_scale(
            bs_buf, scale_slot, 0, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs0_grid = bs0.reshape((CTA_N, cta_n, 4))
        bs0_lo = gl.convert_layout(gl.amd.slice(
            bs0_grid, [CTA_N, half_n, 4], [0, 0, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        bs0_hi = gl.convert_layout(gl.amd.slice(
            bs0_grid, [CTA_N, half_n, 4], [0, half_n, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        if PARENT_B:
            b0_lo = mxfp8_load_b2_parent(
                b0_buf, data_slot, 0, 0, B2_LOAD_LAYOUT, DOT_B)
            b0_hi = mxfp8_load_b2_parent(
                b0_buf, data_slot, 128, 0, B2_LOAD_LAYOUT, DOT_B)
        else:
            b0_lo = mxfp8_load_b2(
                b0_buf, data_slot, 0, B2_LOAD_LAYOUT, DOT_B)
            b0_hi = mxfp8_load_b2(
                b1_buf, data_slot, 0, B2_LOAD_LAYOUT, DOT_B)
    with gl.amd.warp_pipeline_stage("b2_tail_compute_low"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0_lo, bs0_lo, "e4m3", acc0)
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0_hi, bs0_hi, "e4m3", acc1)
    with gl.amd.warp_pipeline_stage("b2_tail_load_high"):
        a1 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        as1 = mxfp8_load_scale(
            as_buf, scale_slot, 4, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as1 = gl.convert_layout(as1, SCALE_A_LAYOUT, assert_trivial=True)
        bs1 = mxfp8_load_scale(
            bs_buf, scale_slot, 4, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs1_grid = bs1.reshape((CTA_N, cta_n, 4))
        bs1_lo = gl.convert_layout(gl.amd.slice(
            bs1_grid, [CTA_N, half_n, 4], [0, 0, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        bs1_hi = gl.convert_layout(gl.amd.slice(
            bs1_grid, [CTA_N, half_n, 4], [0, half_n, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        if PARENT_B:
            b1_lo = mxfp8_load_b2_parent(
                b0_buf, data_slot, 0, 128, B2_LOAD_LAYOUT, DOT_B)
            b1_hi = mxfp8_load_b2_parent(
                b0_buf, data_slot, 128, 128, B2_LOAD_LAYOUT, DOT_B)
        else:
            b1_lo = mxfp8_load_b2(
                b0_buf, data_slot, 128, B2_LOAD_LAYOUT, DOT_B)
            b1_hi = mxfp8_load_b2(
                b1_buf, data_slot, 128, B2_LOAD_LAYOUT, DOT_B)
    with gl.amd.warp_pipeline_stage("b2_tail_compute_high"):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1_lo, bs1_lo, "e4m3", acc0)
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1_hi, bs1_hi, "e4m3", acc1)
    return acc0, acc1

@gluon.jit
def fp8_scaled_cluster_bk256_b2_phase1_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale_a, stride_scale_b, GRID_MN: gl.constexpr,
        SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B2: gl.constexpr,
        SHARED_SCALE_A: gl.constexpr, SHARED_SCALE_B: gl.constexpr,
        WMMA_LAYOUT: gl.constexpr, LOAD_WMMA_LAYOUT: gl.constexpr,
        B2_LOAD_LAYOUT: gl.constexpr,
        B2_FRAGMENT_LOAD_LAYOUT: gl.constexpr,
        OUTPUT_CGA_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_M: gl.constexpr, CTA_N: gl.constexpr,
        TWO_TDM: gl.constexpr, WAITCNT: gl.constexpr,
        UNROLL_SIX: gl.constexpr, DOUBLE_SCALE: gl.constexpr,
        FRAGMENT_LOADS: gl.constexpr,
        SPREAD_ADVANCE: gl.constexpr, PARENT_B: gl.constexpr,
        WMMA_BEFORE_REFILL: gl.constexpr, TWO_BUFFER_OUTPUT: gl.constexpr):
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
        M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    dot_b_load: gl.constexpr = gl.DotOperandLayout(
        1, LOAD_WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [BLOCK_M, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [BLOCK_N // 2, 4])
    scale_b_load: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b_load, [BLOCK_N, 4])

    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, BLOCK_M, 256], SHARED_LAYOUT_A)
    if PARENT_B:
        b0_buf = gl.allocate_shared_memory(
            b_ptr.type.element_ty, [2, CTA_N, 256, 256], SHARED_LAYOUT_B2)
        b1_buf = b0_buf
    else:
        b0_buf = gl.allocate_shared_memory(
            b_ptr.type.element_ty, [2, CTA_N, 128, 256], SHARED_LAYOUT_B2)
        b1_buf = gl.allocate_shared_memory(
            b_ptr.type.element_ty, [2, CTA_N, 128, 256], SHARED_LAYOUT_B2)
    scale_slots: gl.constexpr = 2 if DOUBLE_SCALE else 3
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty,
        [scale_slots, BLOCK_M // 128, 1024], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty,
        [scale_slots, BLOCK_N // 128, 1024], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, 256),
        layout=SHARED_LAYOUT_A)
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn
    if PARENT_B:
        b0_desc = tdm.make_tensor_descriptor(
            base=b_base, shape=(CTA_N, N // CTA_N, K),
            strides=(256 * stride_bn, stride_bn, stride_bk),
            block_shape=(CTA_N, 256, 256), layout=SHARED_LAYOUT_B2)
        b1_desc = b0_desc
    else:
        b0_desc = tdm.make_tensor_descriptor(
            base=b_base, shape=(CTA_N, N // CTA_N, K),
            strides=(256 * stride_bn, stride_bn, stride_bk),
            block_shape=(4, 128, 256), layout=SHARED_LAYOUT_B2)
        b1_desc = tdm.make_tensor_descriptor(
            base=b_base + 128 * stride_bn, shape=(CTA_N, N // CTA_N, K),
            strides=(256 * stride_bn, stride_bn, stride_bk),
            block_shape=(4, 128, 256), layout=SHARED_LAYOUT_B2)
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + (pid_m * BLOCK_M) // 128 * stride_scale_a,
        shape=(M // 128, K // 32 * 128), strides=(stride_scale_a, 1),
        block_shape=(BLOCK_M // 128, 1024), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * BLOCK_N) // 128 * stride_scale_b,
        shape=(N // 128, K // 32 * 128), strides=(stride_scale_b, 1),
        block_shape=(BLOCK_N // 128, 1024), layout=SHARED_SCALE_B)

    if PARENT_B:
        as_desc, bs_desc = mxfp8_issue_leading4_scale(
            as_desc, bs_desc, as_buf, bs_buf, 0)
        a_desc, b0_desc = mxfp8_issue_b2_parent_data(
            a_desc, b0_desc, a_buf, b0_buf, 0)
        b1_desc = b0_desc
        as_desc, bs_desc = mxfp8_issue_leading4_scale(
            as_desc, bs_desc, as_buf, bs_buf, 1)
        a_desc, b0_desc = mxfp8_issue_b2_parent_data(
            a_desc, b0_desc, a_buf, b0_buf, 1)
        b1_desc = b0_desc
    elif TWO_TDM:
        a_desc, b0_desc, b1_desc, as_desc, bs_desc = (
            mxfp8_issue_b2_two_tdm(
                a_desc, b0_desc, b1_desc, as_desc, bs_desc,
                a_buf, b0_buf, b1_buf, as_buf, bs_buf, 0, 0))
        a_desc, b0_desc, b1_desc, as_desc, bs_desc = (
            mxfp8_issue_b2_two_tdm(
                a_desc, b0_desc, b1_desc, as_desc, bs_desc,
                a_buf, b0_buf, b1_buf, as_buf, bs_buf, 1, 1))
    else:
        as_desc, bs_desc = mxfp8_issue_leading4_scale(
            as_desc, bs_desc, as_buf, bs_buf, 0)
        a_desc, b0_desc, b1_desc = mxfp8_issue_b2_data(
            a_desc, b0_desc, b1_desc, a_buf, b0_buf, b1_buf, 0)
        as_desc, bs_desc = mxfp8_issue_leading4_scale(
            as_desc, bs_desc, as_buf, bs_buf, 1)
        a_desc, b0_desc, b1_desc = mxfp8_issue_b2_data(
            a_desc, b0_desc, b1_desc, a_buf, b0_buf, b1_buf, 1)
    if not DOUBLE_SCALE:
        as_desc, bs_desc = mxfp8_issue_leading4_scale(
            as_desc, bs_desc, as_buf, bs_buf, 2)

    acc0 = gl.zeros(
        (BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    acc1 = gl.zeros(
        (BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 256)
    gl.assume(iter_max >= 4)
    gl.assume(iter_max % 2 == 0)
    data_scale_waitcnt: gl.constexpr = 3 if DOUBLE_SCALE else 4
    if UNROLL_SIX:
        for _ in range(0, (iter_max - 4) // 6):
            for inner in gl.static_range(6):
                (a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1) = (
                    mxfp8_bk256_b2_consume_and_refill(
                        a_buf, b0_buf, b1_buf, as_buf, bs_buf,
                        inner % 2, inner % 3, inner % 3,
                        a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1,
                        dot_a, dot_b, B2_LOAD_LAYOUT, B2_FRAGMENT_LOAD_LAYOUT,
                        scale_a_layout, scale_a_layout,
                        scale_b_layout, scale_b_load,
                        BLOCK_M, BLOCK_N, CTA_N, True, TWO_TDM, WAITCNT,
                        data_scale_waitcnt,
                        FRAGMENT_LOADS, SPREAD_ADVANCE, PARENT_B,
                        WMMA_BEFORE_REFILL))
        for cleanup_pair in range(0, ((iter_max - 4) % 6) // 2):
            cleanup_tile = cleanup_pair * 2
            s0 = cleanup_tile % 3
            s1 = (cleanup_tile + 1) % 3
            (a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1) = (
                mxfp8_bk256_b2_consume_and_refill(
                    a_buf, b0_buf, b1_buf, as_buf, bs_buf, 0, s0, s0,
                    a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1,
                    dot_a, dot_b, B2_LOAD_LAYOUT, B2_FRAGMENT_LOAD_LAYOUT,
                    scale_a_layout, scale_a_layout,
                    scale_b_layout, scale_b_load,
                    BLOCK_M, BLOCK_N, CTA_N, True, TWO_TDM, WAITCNT,
                    data_scale_waitcnt,
                    FRAGMENT_LOADS, SPREAD_ADVANCE, PARENT_B,
                    WMMA_BEFORE_REFILL))
            (a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1) = (
                mxfp8_bk256_b2_consume_and_refill(
                    a_buf, b0_buf, b1_buf, as_buf, bs_buf, 1, s1, s1,
                    a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1,
                    dot_a, dot_b, B2_LOAD_LAYOUT, B2_FRAGMENT_LOAD_LAYOUT,
                    scale_a_layout, scale_a_layout,
                    scale_b_layout, scale_b_load,
                    BLOCK_M, BLOCK_N, CTA_N, True, TWO_TDM, WAITCNT,
                    data_scale_waitcnt,
                    FRAGMENT_LOADS, SPREAD_ADVANCE, PARENT_B,
                    WMMA_BEFORE_REFILL))
    elif DOUBLE_SCALE:
        for _ in range(0, (iter_max - 4) // 2):
            (a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1) = (
                mxfp8_bk256_b2_consume_and_refill(
                    a_buf, b0_buf, b1_buf, as_buf, bs_buf, 0, 0, 0,
                    a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1,
                    dot_a, dot_b, B2_LOAD_LAYOUT, B2_FRAGMENT_LOAD_LAYOUT,
                    scale_a_layout, scale_a_layout,
                    scale_b_layout, scale_b_load,
                    BLOCK_M, BLOCK_N, CTA_N, True, TWO_TDM, WAITCNT,
                    data_scale_waitcnt,
                    FRAGMENT_LOADS, SPREAD_ADVANCE, PARENT_B,
                    WMMA_BEFORE_REFILL))
            (a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1) = (
                mxfp8_bk256_b2_consume_and_refill(
                    a_buf, b0_buf, b1_buf, as_buf, bs_buf, 1, 1, 1,
                    a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1,
                    dot_a, dot_b, B2_LOAD_LAYOUT, B2_FRAGMENT_LOAD_LAYOUT,
                    scale_a_layout, scale_a_layout,
                    scale_b_layout, scale_b_load,
                    BLOCK_M, BLOCK_N, CTA_N, True, TWO_TDM, WAITCNT,
                    data_scale_waitcnt,
                    FRAGMENT_LOADS, SPREAD_ADVANCE, PARENT_B,
                    WMMA_BEFORE_REFILL))
    else:
        for pair_idx in range(0, (iter_max - 4) // 2):
            tile0 = pair_idx * 2
            s0 = tile0 % 3
            s1 = (tile0 + 1) % 3
            (a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1) = (
                mxfp8_bk256_b2_consume_and_refill(
                    a_buf, b0_buf, b1_buf, as_buf, bs_buf, 0, s0, s0,
                    a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1,
                    dot_a, dot_b, B2_LOAD_LAYOUT, B2_FRAGMENT_LOAD_LAYOUT,
                    scale_a_layout, scale_a_layout,
                    scale_b_layout, scale_b_load,
                    BLOCK_M, BLOCK_N, CTA_N, True, TWO_TDM, WAITCNT,
                    data_scale_waitcnt,
                    FRAGMENT_LOADS, SPREAD_ADVANCE, PARENT_B,
                    WMMA_BEFORE_REFILL))
            (a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1) = (
                mxfp8_bk256_b2_consume_and_refill(
                    a_buf, b0_buf, b1_buf, as_buf, bs_buf, 1, s1, s1,
                    a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1,
                    dot_a, dot_b, B2_LOAD_LAYOUT, B2_FRAGMENT_LOAD_LAYOUT,
                    scale_a_layout, scale_a_layout,
                    scale_b_layout, scale_b_load,
                    BLOCK_M, BLOCK_N, CTA_N, True, TWO_TDM, WAITCNT,
                    data_scale_waitcnt,
                    FRAGMENT_LOADS, SPREAD_ADVANCE, PARENT_B,
                    WMMA_BEFORE_REFILL))
    residual = iter_max - 4
    if DOUBLE_SCALE:
        s0: gl.constexpr = 0
        s1: gl.constexpr = 1
    else:
        s0 = residual % 3
        s1 = (residual + 1) % 3
    (a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1) = (
        mxfp8_bk256_b2_consume_and_refill(
            a_buf, b0_buf, b1_buf, as_buf, bs_buf, 0, s0, s0,
            a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1,
            dot_a, dot_b, B2_LOAD_LAYOUT, B2_FRAGMENT_LOAD_LAYOUT,
            scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load,
            BLOCK_M, BLOCK_N, CTA_N, True, TWO_TDM, WAITCNT,
            data_scale_waitcnt,
            FRAGMENT_LOADS, SPREAD_ADVANCE, PARENT_B, WMMA_BEFORE_REFILL))
    (a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1) = (
        mxfp8_bk256_b2_consume_and_refill(
            a_buf, b0_buf, b1_buf, as_buf, bs_buf, 1, s1, s1,
            a_desc, b0_desc, b1_desc, as_desc, bs_desc, acc0, acc1,
            dot_a, dot_b, B2_LOAD_LAYOUT, B2_FRAGMENT_LOAD_LAYOUT,
            scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load,
            BLOCK_M, BLOCK_N, CTA_N, DOUBLE_SCALE, TWO_TDM, WAITCNT,
            data_scale_waitcnt,
            FRAGMENT_LOADS, SPREAD_ADVANCE, PARENT_B, WMMA_BEFORE_REFILL))
    tdm.async_wait(WAITCNT if TWO_TDM else 3)
    acc0, acc1 = mxfp8_bk256_b2_consume_tail(
        a_buf, b0_buf, b1_buf, as_buf, bs_buf,
        0, 0 if DOUBLE_SCALE else (iter_max - 2) % 3,
        acc0, acc1, dot_a, dot_b, B2_LOAD_LAYOUT,
        scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load,
        BLOCK_M, BLOCK_N, CTA_N, PARENT_B)
    tdm.async_wait(0)
    acc0, acc1 = mxfp8_bk256_b2_consume_tail(
        a_buf, b0_buf, b1_buf, as_buf, bs_buf,
        1, 1 if DOUBLE_SCALE else (iter_max - 1) % 3,
        acc0, acc1, dot_a, dot_b, B2_LOAD_LAYOUT,
        scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load,
        BLOCK_M, BLOCK_N, CTA_N, PARENT_B)
    snapshot_cluster_wait()
    a_buf._keep_alive()
    b0_buf._keep_alive()
    if not PARENT_B:
        b1_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    if TWO_BUFFER_OUTPUT:
        mxfp8_tdm_store_split_n2_pipelined(
            c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1,
            OUTPUT_CGA_LAYOUT, CTA_M, CTA_N)
    else:
        mxfp8_tdm_store_split_n2(
            c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1,
            OUTPUT_CGA_LAYOUT, CTA_M, CTA_N)

@gluon.constexpr_function
def output_cga_rank5(cga):
    return tuple(tuple(basis) + (0,) for basis in cga)


@gluon.jit
def store_full_two_buffers(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        OUTPUT_CGA: gl.constexpr, CTA_M: gl.constexpr, CTA_N: gl.constexpr,
        TILE_M: gl.constexpr, TILE_N: gl.constexpr):
    """Stage alternating 16-column strips in two independent LDS buffers.

    N bit 16 is a register bit in both small-tile WMMA layouts, so splitting
    it needs no cross-warp conversion. Each descriptor scatters its strips
    into the corresponding half of every 32-column output segment.
    """
    SHAPE: gl.constexpr = (CTA_M, CTA_N, TILE_M, TILE_N // 32, 16)
    SLICE_SHAPE: gl.constexpr = (CTA_M, CTA_N, TILE_M, TILE_N // 32, 1, 16)
    cga: gl.constexpr = output_cga_rank5(OUTPUT_CGA)
    layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[16, 8]], SHAPE, [4, 3, 2, 1, 0], cga)
    shared0 = gl.allocate_shared_memory(c_ptr.type.element_ty, SHAPE, layout)
    shared1 = gl.allocate_shared_memory(c_ptr.type.element_ty, SHAPE, layout)
    output = acc.reshape((CTA_M, TILE_M, CTA_N, TILE_N // 32, 2, 16))
    output = output.permute((0, 2, 1, 3, 4, 5))
    output0 = gl.amd.slice(output, SLICE_SHAPE, (0, 0, 0, 0, 0, 0)).reshape(SHAPE)
    output1 = gl.amd.slice(output, SLICE_SHAPE, (0, 0, 0, 0, 1, 0)).reshape(SHAPE)
    desc = tdm.make_tensor_descriptor(
        base=c_ptr, shape=(M // TILE_M, N // TILE_N, TILE_M, TILE_N // 32, 32),
        strides=(TILE_M * stride_cm, TILE_N * stride_cn, stride_cm, 32 * stride_cn, stride_cn),
        block_shape=SHAPE, layout=layout)
    shared0.store(output0.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [pid_m * CTA_M, pid_n * CTA_N, 0, 0, 0], shared0)
    shared1.store(output1.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [pid_m * CTA_M, pid_n * CTA_N, 0, 0, 16], shared1)
    tdm.async_wait(0)


@gluon.jit
def tiled_bk512_five(a_buf, b_buf, as_buf, bs_buf, slot, a_desc, b_desc, as_desc, bs_desc, acc, dot_a: gl.constexpr, dot_b: gl.constexpr, scale_a_layout: gl.constexpr, scale_b_layout: gl.constexpr, block_m: gl.constexpr, block_n: gl.constexpr):
    tdm.async_wait(3)
    with gl.amd.warp_pipeline_stage('tiled_stage0_load_k0', phase_gap=1):
        a0 = a_buf.index(slot).slice(0, 128, 1).load(layout=dot_a)
        as0 = tiled_load_scale(as_buf, slot, 0, scale_a_layout)
        bs0 = tiled_load_scale(bs_buf, slot, 0, scale_b_layout)
        a1 = a_buf.index(slot).slice(128, 128, 1).load(layout=dot_a)
        as1 = tiled_load_scale(as_buf, slot, 4, scale_a_layout)
        bs1 = tiled_load_scale(bs_buf, slot, 4, scale_b_layout)
        b0 = b_buf.index(slot).slice(0, 128, 1).permute([1, 0]).load(layout=dot_b)
        b1 = b_buf.index(slot).slice(128, 128, 1).permute([1, 0]).load(layout=dot_b)
    with gl.amd.warp_pipeline_stage('tiled_stage2_compute_k0'):
        acc = gl.amd.gfx1250.wmma_scaled(a0, as0, 'e4m3', b0, bs0, 'e4m3', acc)
        acc = gl.amd.gfx1250.wmma_scaled(a1, as1, 'e4m3', b1, bs1, 'e4m3', acc)
    with gl.amd.warp_pipeline_stage('tiled_stage4_load_k2'):
        a2 = a_buf.index(slot).slice(256, 128, 1).load(layout=dot_a)
        as2 = tiled_load_scale(as_buf, slot, 8, scale_a_layout)
        b2 = b_buf.index(slot).slice(256, 128, 1).permute([1, 0]).load(layout=dot_b)
        bs2 = tiled_load_scale(bs_buf, slot, 8, scale_b_layout)
        a3 = a_buf.index(slot).slice(384, 128, 1).load(layout=dot_a)
        as3 = tiled_load_scale(as_buf, slot, 12, scale_a_layout)
        b3 = b_buf.index(slot).slice(384, 128, 1).permute([1, 0]).load(layout=dot_b)
        bs3 = tiled_load_scale(bs_buf, slot, 12, scale_b_layout)
        gl.amd.gfx1250.cluster.arrive()
    with gl.amd.warp_pipeline_stage('tiled_stage6_compute_k2'):
        acc = gl.amd.gfx1250.wmma_scaled(a2, as2, 'e4m3', b2, bs2, 'e4m3', acc)
        acc = gl.amd.gfx1250.wmma_scaled(a3, as3, 'e4m3', b3, bs3, 'e4m3', acc)
    with gl.amd.warp_pipeline_stage('tiled_stage8_wait_refill'):
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc, as_desc, bs_desc = tiled_issue_refill(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, slot, 512, 16)
    return (a_desc, b_desc, as_desc, bs_desc, acc)

@gluon.jit
def mxfp8_bk512_five_gfx1250(a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_as, stride_bs, GRID_MN: gl.constexpr, shared_a: gl.constexpr, shared_b: gl.constexpr, shared_as: gl.constexpr, shared_bs: gl.constexpr, wmma: gl.constexpr, load_wmma: gl.constexpr, output_cga: gl.constexpr, block_m: gl.constexpr, block_n: gl.constexpr, block_k: gl.constexpr, cta_m_count: gl.constexpr, cta_n_count: gl.constexpr, cta_tile_m: gl.constexpr, cta_tile_n: gl.constexpr, split_n: gl.constexpr, preshuffle_scales: gl.constexpr, TWO_BUFFER_OUTPUT: gl.constexpr):
    gl.static_assert(gl.num_ctas() == cta_m_count * cta_n_count)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(M, N, block_m, block_n, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_a, [block_m, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_b, [block_n if not split_n else block_n // 2, 4])
    if split_n:
        dot_b_load: gl.constexpr = gl.DotOperandLayout(1, load_wmma, 16)
        scale_b_load: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(dot_b_load, [block_n, 4])
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [2, block_m, block_k], shared_a)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [2, block_n, block_k], shared_b)
    scale_k: gl.constexpr = block_k // 32
    scale_step: gl.constexpr = scale_k * 128 if preshuffle_scales else scale_k
    if preshuffle_scales:
        as_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty, [2, block_m // 128, scale_step], shared_as)
        bs_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty, [2, block_n // 128, scale_step], shared_bs)
    else:
        as_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty, [2, block_m, scale_k], shared_as)
        bs_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty, [2, block_n, scale_k], shared_bs)
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + pid_m * block_m * stride_am, shape=(M, K), strides=(stride_am, stride_ak), block_shape=(block_m, block_k), layout=shared_a)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + pid_n * block_n * stride_bn, shape=(N, K), strides=(stride_bn, stride_bk), block_shape=(block_n, block_k), layout=shared_b)
    if preshuffle_scales:
        as_desc = tdm.make_tensor_descriptor(base=a_scale_ptr + pid_m * block_m // 128 * stride_as, shape=(M // 128, K // 32 * 128), strides=(stride_as, 1), block_shape=(block_m // 128, scale_step), layout=shared_as)
        bs_desc = tdm.make_tensor_descriptor(base=b_scale_ptr + pid_n * block_n // 128 * stride_bs, shape=(N // 128, K // 32 * 128), strides=(stride_bs, 1), block_shape=(block_n // 128, scale_step), layout=shared_bs)
    else:
        as_desc = tdm.make_tensor_descriptor(base=a_scale_ptr + pid_m * block_m * stride_as, shape=(M, K // 32), strides=(stride_as, 1), block_shape=(block_m, scale_k), layout=shared_as)
        bs_desc = tdm.make_tensor_descriptor(base=b_scale_ptr + pid_n * block_n * stride_bs, shape=(N, K // 32), strides=(stride_bs, 1), block_shape=(block_n, scale_k), layout=shared_bs)
    for slot in gl.static_range(2):
        a_desc, b_desc, as_desc, bs_desc = tiled_issue_refill(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, slot, block_k, scale_step)
    iter_max = gl.cdiv(K, block_k)
    gl.assume(iter_max >= 3)
    if split_n:
        acc0 = gl.zeros((block_m, block_n // 2), dtype=gl.float32, layout=wmma)
        acc1 = gl.zeros((block_m, block_n // 2), dtype=gl.float32, layout=wmma)
        for group in range(0, (iter_max - 2) // 6):
            for inner in gl.static_range(6):
                tile_idx = group * 6 + inner
                split_slot = tile_idx % 2
                a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = tiled_bk256_stage8(a_buf, b_buf, as_buf, bs_buf, split_slot, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b, dot_b_load, scale_a_layout, scale_b_layout, scale_b_load, block_m, block_n, cta_n_count, cta_tile_n)
        for tile_idx in range((iter_max - 2) // 6 * 6, iter_max - 2):
            split_slot = tile_idx % 2
            a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = tiled_bk256_stage8(a_buf, b_buf, as_buf, bs_buf, split_slot, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b, dot_b_load, scale_a_layout, scale_b_layout, scale_b_load, block_m, block_n, cta_n_count, cta_tile_n)
        tdm.async_wait(3)
        penultimate_slot = (iter_max - 2) % 2
        acc0, acc1 = tiled_bk256_tail(a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc0, acc1, dot_a, dot_b, dot_b_load, scale_a_layout, scale_b_layout, scale_b_load, block_m, block_n, cta_n_count, cta_tile_n)
        tdm.async_wait(0)
        last_slot = (iter_max - 1) % 2
        acc0, acc1 = tiled_bk256_tail(a_buf, b_buf, as_buf, bs_buf, last_slot, acc0, acc1, dot_a, dot_b, dot_b_load, scale_a_layout, scale_b_layout, scale_b_load, block_m, block_n, cta_n_count, cta_tile_n)
        snapshot_cluster_wait()
        a_buf._keep_alive()
        b_buf._keep_alive()
        as_buf._keep_alive()
        bs_buf._keep_alive()
        tiled_store_split_n2(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1, output_cga, cta_m_count, cta_n_count, cta_tile_m, cta_tile_n)
    else:
        acc = gl.zeros((block_m, block_n), dtype=gl.float32, layout=wmma)
        for group in range(0, (iter_max - 2) // 6):
            for inner in gl.static_range(6):
                tile_idx = group * 6 + inner
                full_slot = tile_idx % 2
                if block_k == 256:
                    a_desc, b_desc, as_desc, bs_desc, acc = tiled_bk256_single_stage8(a_buf, b_buf, as_buf, bs_buf, full_slot, a_desc, b_desc, as_desc, bs_desc, acc, dot_a, dot_b, scale_a_layout, scale_b_layout, block_m, block_n)
                else:
                    a_desc, b_desc, as_desc, bs_desc, acc = tiled_bk512_five(a_buf, b_buf, as_buf, bs_buf, full_slot, a_desc, b_desc, as_desc, bs_desc, acc, dot_a, dot_b, scale_a_layout, scale_b_layout, block_m, block_n)
        for tile_idx in range((iter_max - 2) // 6 * 6, iter_max - 2):
            full_slot = tile_idx % 2
            if block_k == 256:
                a_desc, b_desc, as_desc, bs_desc, acc = tiled_bk256_single_stage8(a_buf, b_buf, as_buf, bs_buf, full_slot, a_desc, b_desc, as_desc, bs_desc, acc, dot_a, dot_b, scale_a_layout, scale_b_layout, block_m, block_n)
            else:
                a_desc, b_desc, as_desc, bs_desc, acc = tiled_bk512_five(a_buf, b_buf, as_buf, bs_buf, full_slot, a_desc, b_desc, as_desc, bs_desc, acc, dot_a, dot_b, scale_a_layout, scale_b_layout, block_m, block_n)
        tdm.async_wait(3)
        penultimate_slot = (iter_max - 2) % 2
        if block_k == 256:
            acc = tiled_bk256_single_tail(a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, dot_a, dot_b, scale_a_layout, scale_b_layout, block_m, block_n)
        else:
            acc = tiled_consume_tail(a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, dot_a, dot_b, scale_a_layout, scale_b_layout, block_m, block_n, block_k)
        tdm.async_wait(0)
        last_slot = (iter_max - 1) % 2
        if block_k == 256:
            acc = tiled_bk256_single_tail(a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b, scale_a_layout, scale_b_layout, block_m, block_n)
        else:
            acc = tiled_consume_tail(a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b, scale_a_layout, scale_b_layout, block_m, block_n, block_k)
        snapshot_cluster_wait()
        a_buf._keep_alive()
        b_buf._keep_alive()
        as_buf._keep_alive()
        bs_buf._keep_alive()
        if TWO_BUFFER_OUTPUT:
            store_full_two_buffers(c_ptr, pid_m, pid_n, stride_cm, stride_cn,
                                   M, N, acc, output_cga, cta_m_count,
                                   cta_n_count, cta_tile_m, cta_tile_n)
        else:
            snapshot_tdm_store_full_tile(c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc, wmma, block_m, block_n, cta_n_count)

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

TILED_CONFIGS = {"64x64x512": (64,64,512,False,False), "128x128x512": (128,128,512,False,False)}

def parent_layout(shared_b, cluster):
    inner = shared_b.partition_layout
    cga = tuple((basis[0], 0, basis[1]) for basis in inner.cga_layout)
    padded = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [cluster, 64, 256], [2, 1, 0], cga)
    shared = PartitionedSharedLayout(2, 2, 1, padded)
    load = gl.DistributedLinearLayout(
        reg_bases=[[1,0,0],[2,0,0],[4,0,0],[8,0,0],[32,0,0],[64,0,0],[0,0,16],[0,0,32]],
        lane_bases=[[0,0,1],[0,0,2],[0,0,4],[0,0,8],[16,0,0]],
        warp_bases=[[0,0,64],[0,0,0],[0,0,0]],
        block_bases=[[0,0,0]] * (cluster.bit_length()-1) + [[0,1 << i,0] for i in range(cluster.bit_length()-1)],
        shape=[128,cluster,128])
    return shared, load


SHAPES = [
    (512,6144,7168,256,256,2,15.94,16.31),
    (512,7168,3072,128,512,4,8.23,8.27),
    (512,8192,1536,128,512,4,5.91,5.81),
    (512,2048,7168,64,512,4,9.69,9.47),
    (512,65536,1536,256,256,2,20.13,20.08),
    (512,7168,16384,256,256,2,27.12,25.57),
    (16384,6144,7168,256,256,4,194.68,194.05),
    (16384,7168,3072,256,256,4,106.14,105.69),
    (16384,8192,1536,256,256,4,69.27,70.88),
    (16384,2048,7168,256,256,4,68.20,65.31),
    (16384,65536,1536,256,256,4,600.29,562.03),
    (16384,7168,16384,256,256,4,546.19,544.31),
]


def flydsl_scale(values):
    # AITER fp4_utils.f32_to_mx_e8m0_scale, RoundUp, FP8_E4M3.
    bits = ((values * 448.0) / 448.0).view(torch.int32)
    exp = (bits >> 23) & 255
    return (exp + (((bits & 0x7fffff) != 0) & (exp < 255))).to(torch.uint8)


def make_inputs(m, n, k, packed, seed):
    torch.manual_seed(seed)
    a = (torch.rand((m,k), device='cuda', dtype=torch.float32)/10).to(torch.float8_e4m3fn)
    b = (torch.rand((n,k), device='cuda', dtype=torch.float32)/10).to(torch.float8_e4m3fn)
    sa = flydsl_scale(torch.rand((m,k//128), device='cuda', dtype=torch.float32))
    sb = flydsl_scale(torch.rand((n//128,k//128), device='cuda', dtype=torch.float32))
    sa32 = sa.repeat_interleave(4, dim=1)
    sb32 = sb.repeat_interleave(128, dim=0).repeat_interleave(4, dim=1)
    asp = pack_scale(sa32,4) if packed else sa32
    bsp = pack_scale(sb32,4) if packed else sb32
    c = torch.empty((m,n), device='cuda', dtype=torch.bfloat16)
    return (a,b,c,asp,bsp),sa,sb


# Shape-specific winners from the paired two-buffer output experiment.
TWO_BUFFER_SHAPES = {
    (512, 6144, 7168),
    (512, 65536, 1536),
    (512, 7168, 16384),
    (16384, 6144, 7168),
    (16384, 7168, 3072),
    (16384, 8192, 1536),
    (16384, 2048, 7168),
    (16384, 65536, 1536),
    (16384, 7168, 16384),
}


def selected_output_store(shape, mode="auto"):
    if mode not in ("auto", "serial", "double"):
        raise ValueError(f"unknown output store mode: {mode}")
    if mode == "auto":
        return "double" if tuple(shape[:3]) in TWO_BUFFER_SHAPES else "serial"
    return mode


def launcher(shape, output_store="auto"):
    two_buffer = selected_output_store(shape, output_store) == "double"
    m,n,k,tile,bk,cluster,*_ = shape
    bm=bn=tile*cluster
    assert m%bm == n%bn == k%bk == 0
    grid=(m//bm*(n//bn),1)
    if bk==256:
        sa,sb,sas,sbs,w,lw,cga=build_mxfp8_bk256_opt_layouts(cluster,cluster)
        bp,bl=parent_layout(sb,cluster)
    else:
        sa,sb,sas,sbs,w,lw,cga=build_tiled_layouts(f'{tile}x{tile}x{bk}',cluster)
    def launch(tensors):
        a,b,c,asc,bsc=tensors
        common=(a,b,c,asc,bsc,m,n,k,k,1,1,k,n,1,asc.stride(0),bsc.stride(0))
        if bk==256:
            return fp8_scaled_cluster_bk256_b2_phase1_gfx1250[grid](
                *common, GRID_MN=grid[0],SHARED_LAYOUT_A=sa,SHARED_LAYOUT_B2=bp,
                SHARED_SCALE_A=sas,SHARED_SCALE_B=sbs,WMMA_LAYOUT=w,LOAD_WMMA_LAYOUT=lw,
                B2_LOAD_LAYOUT=bl,B2_FRAGMENT_LOAD_LAYOUT=bl,OUTPUT_CGA_LAYOUT=cga,
                BLOCK_M=bm,BLOCK_N=bn,CTA_M=cluster,CTA_N=cluster,
                TWO_TDM=False,WAITCNT=2,UNROLL_SIX=True,DOUBLE_SCALE=False,
                FRAGMENT_LOADS=False,SPREAD_ADVANCE=False,PARENT_B=True,WMMA_BEFORE_REFILL=False,TWO_BUFFER_OUTPUT=two_buffer,
                num_warps=8,num_ctas=cluster*cluster,waves_per_eu=2,llvm_fn_attrs=AGPR_ATTRS)
        return mxfp8_bk512_five_gfx1250[grid](
            *common,GRID_MN=grid[0],shared_a=sa,shared_b=sb,shared_as=sas,shared_bs=sbs,
            wmma=w,load_wmma=lw,output_cga=cga,block_m=bm,block_n=bn,block_k=bk,
            cta_m_count=cluster,cta_n_count=cluster,cta_tile_m=tile,cta_tile_n=tile,
            split_n=False,preshuffle_scales=False,TWO_BUFFER_OUTPUT=two_buffer,num_warps=8,num_ctas=cluster*cluster,
            waves_per_eu=2,llvm_fn_attrs=AGPR_ATTRS)
    return launch


def check(tensors, sa, sb):
    # Full output checked in row chunks to bound FP32 reference memory.
    a,b,c,*_=tensors
    bs = torch.exp2(sb.float()-127).repeat_interleave(128,0).repeat_interleave(128,1)
    bf = b.float()*bs
    del bs
    worst=0.0
    for start in range(0,a.shape[0],256):
        af=a[start:start+256].float()*torch.exp2(sa[start:start+256].float()-127).repeat_interleave(128,1)
        ref=af @ bf.T
        actual=c[start:start+256].float()
        err=(actual-ref).abs() / ref.abs().clamp_min(1e-6)
        worst=max(worst,err.max().item())
        torch.testing.assert_close(actual,ref,rtol=0.008,atol=1e-5)
    return worst


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--shape-index',type=int,nargs='*')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--copies',type=int,default=100)
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--output-store',choices=['auto','serial','double'],default='auto')
    parser.add_argument('--output',default='experiments/mxfp8_five_shapes_20260929/results.json')
    args=parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    print('Triton:',triton.__file__, 'GPU:',torch.cuda.get_device_name(),flush=True)
    results=[]
    for index in (range(len(SHAPES)) if args.shape_index is None else args.shape_index):
        shape=SHAPES[index]; m,n,k,tile,bk,cluster,base,apre=shape
        print('START',index,shape,flush=True)
        tensors,sa,sb=make_inputs(m,n,k,bk==256,args.seed)
        launch=launcher(shape,args.output_store)
        compiled=launch(tensors)
        torch.cuda.synchronize()
        error=check(tensors,sa,sb)
        print('CHECK',index,error, 'resources',compiled.n_regs,compiled.n_spills,compiled.metadata.shared,flush=True)
        del sa,sb
        row=dict(index=index,output_store=selected_output_store(shape,args.output_store),M=m,N=n,K=k,cta=[tile,tile,bk],cluster=[cluster,cluster],base_us=base,apre_us=apre,max_relative_error=error,vgpr=compiled.n_regs,spills=compiled.n_spills,lds=compiled.metadata.shared)
        if not args.check_only:
            torch.cuda.empty_cache()
            free,total=torch.cuda.mem_get_info()
            bytes_per_copy=sum(t.numel()*t.element_size() for t in tensors)
            # Leave at least 2 GiB and half the currently free memory for compilation/graphs.
            copies=max(1,min(args.copies,1+int(max(0,free-max(2*1024**3,free//2))//bytes_per_copy)))
            pool=[tensors]+[tuple(t.clone() for t in tensors) for _ in range(copies-1)]
            for i in range(20): launch(pool[i%copies])
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for i in range(100): launch(pool[i%copies])
            for _ in range(3): graph.replay()
            times=[]
            for _ in range(args.rounds):
                start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                start.record(); graph.replay(); end.record(); end.synchronize()
                times.append(start.elapsed_time(end)*1000/100)
            us=statistics.median(times)
            row.update(copies=copies,bytes_per_copy=bytes_per_copy,launches_per_graph=100,times_us=times,us=us,tflops=2*m*n*k/us/1e6)
            print('PERF',json.dumps(row),flush=True)
            del graph,pool
        results.append(row)
        Path(args.output).write_text(json.dumps(dict(seed=args.seed,input='FlyDSL uniform E4M3, E8M0 RoundUp FP8 block128',b_reuse=False,results=results),indent=2)+'\n')
        del tensors,compiled,launch
        gc.collect(); torch.cuda.empty_cache()


if __name__=='__main__':
    main()
