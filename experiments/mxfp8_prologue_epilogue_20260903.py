"""Compare retained and newer GFX1250 MXFP8 prologue/epilogue schedules.

This harness records the experiment that selected the canonical split
epilogue. It compares the adopted kernel with all combinations of:

* single-slot prologue: make slot 0 cluster-visible before issuing slot 1;
* pipelined tail: retain warp-pipeline stages while draining the final slots;
* split epilogue: serialize two half-width TDM stores through a smaller LDS
  allocation that can reuse the now-dead operand arena.
* middle/late cluster wait: move refill after the second K128 WMMA, with the
  cluster wait immediately before or after that WMMA.

Use a long-K correctness check before trusting timing because an unsafe TDM
wait can pass shallow tests and fail only after repeated slot reuse.
"""

import argparse
from pathlib import Path
import sys
import time

import torch
import triton
from triton._C.libtriton.gluon_ir import make_cga_layout
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from third_party.amd.python.examples.gluon.gfx1250_gemm.kernels import (
    AGPR_ATTRS,
    MXScaleTensor,
    build_mxfp8_layouts as build_canonical_mxfp8_layouts,
    fp8_scaled_cluster_bf16_style_kernel_gfx1250,
    init_data,
    make_trig_tensor,
    mxfp8_consume_tile,
    mxfp8_issue_leading4_scale,
    mxfp8_load_scale,
    pack_scale,
    snapshot_cluster_wait,
    snapshot_get_xcd_swizzled_pids,
    snapshot_tdm_store_full_tile,
    torch_gemm_mxfp,
)


VARIANTS = {
    "canonical": None,
    # Explicit negative control; excluded from the default variant list.
    "no_tail_control": (False, False, True, 0),
    # Warp-pipelined tail drain is enabled for every schedule variant.
    "legacy_full": (False, True, False, 0),
    "legacy_prologue": (True, True, False, 0),
    "split": (False, True, True, 0),
    "prologue_split": (True, True, True, 0),
    # The hardware E4M3 WMMA is fixed at K128, so these place the barrier
    # between the two K128 WMMAs of a BK256 tile rather than splitting a WMMA.
    "middle_wait": (False, True, True, 1),
    "late_wait": (False, True, True, 2),
}
DEFAULT_VARIANTS = ("canonical",)


def build_mxfp8_variant_layouts(split_output):
    """Build legacy 256-column or split-output 128-column accumulator slices."""
    slice_n = 128 if split_output else 256
    cga_c = make_cga_layout([4, 4], [4, 4], [0, 1])
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        1024, 1024, local_a, local_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=slice_n)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga_c)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        256, 256, padded_a, padded_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=slice_n)
    shared_as = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [8, 1024], [1, 0], cga_a)
    shared_bs = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [8, 1024], [1, 0], cga_b)
    return shared_a, shared_b, shared_as, shared_bs, wmma


@gluon.jit
def mxfp8_variant_issue_data(
        a_desc, b_desc, a_buf, b_buf, tile_idx, slot,
        B_WARP_HINT: gl.constexpr):
    tile_k = tile_idx * 256
    a_load = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, tile_k])
    b_load = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, tile_k])
    tdm.async_load_fused([
        (a_load, a_buf.index(slot), 0b00000011),
        (b_load, b_buf.index(slot), B_WARP_HINT),
    ])


@gluon.jit
def mxfp8_variant_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, slot, a_desc, b_desc, as_desc,
        bs_desc, refill_idx, acc, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        B_WARP_HINT: gl.constexpr, WAIT_ORDER: gl.constexpr):
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a0 = a_buf.index(slot).slice(0, 1024, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        as0 = mxfp8_load_scale(
            as_buf, slot, 0, SCALE_A_LAYOUT, 1024, 8, 4)
        b0 = b_buf.index(slot).slice(0, 1024, 0).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs0 = mxfp8_load_scale(
            bs_buf, slot, 0, SCALE_B_LAYOUT, 1024, 8, 4)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0, bs0, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a1 = a_buf.index(slot).slice(0, 1024, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        as1 = mxfp8_load_scale(
            as_buf, slot, 4, SCALE_A_LAYOUT, 1024, 8, 4)
        b1 = b_buf.index(slot).slice(0, 1024, 0).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs1 = mxfp8_load_scale(
            bs_buf, slot, 4, SCALE_B_LAYOUT, 1024, 8, 4)
        gl.amd.gfx1250.cluster.arrive()
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        if WAIT_ORDER == 0:
            gl.amd.gfx1250.cluster.wait()
            mxfp8_variant_issue_data(
                a_desc, b_desc, a_buf, b_buf, refill_idx, slot, B_WARP_HINT)
            mxfp8_issue_leading4_scale(
                as_desc, bs_desc, as_buf, bs_buf, refill_idx, slot)
            acc = gl.amd.gfx1250.wmma_scaled(
                a1, as1, "e4m3", b1, bs1, "e4m3", acc)
        elif WAIT_ORDER == 1:
            gl.amd.gfx1250.cluster.wait()
            acc = gl.amd.gfx1250.wmma_scaled(
                a1, as1, "e4m3", b1, bs1, "e4m3", acc)
            mxfp8_variant_issue_data(
                a_desc, b_desc, a_buf, b_buf, refill_idx, slot, B_WARP_HINT)
            mxfp8_issue_leading4_scale(
                as_desc, bs_desc, as_buf, bs_buf, refill_idx, slot)
        else:
            acc = gl.amd.gfx1250.wmma_scaled(
                a1, as1, "e4m3", b1, bs1, "e4m3", acc)
            gl.amd.gfx1250.cluster.wait()
            mxfp8_variant_issue_data(
                a_desc, b_desc, a_buf, b_buf, refill_idx, slot, B_WARP_HINT)
            mxfp8_issue_leading4_scale(
                as_desc, bs_desc, as_buf, bs_buf, refill_idx, slot)
    return acc


@gluon.jit
def mxfp8_consume_pipelined_tail(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, DOT_A: gl.constexpr,
        DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr):
    with gl.amd.warp_pipeline_stage("stage0_tail", priority=0):
        a0 = a_buf.index(slot).slice(0, 1024, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        as0 = mxfp8_load_scale(
            as_buf, slot, 0, SCALE_A_LAYOUT, 1024, 8, 4)
        b0 = b_buf.index(slot).slice(0, 1024, 0).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs0 = mxfp8_load_scale(
            bs_buf, slot, 0, SCALE_B_LAYOUT, 1024, 8, 4)
    with gl.amd.warp_pipeline_stage("stage1_tail", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0, bs0, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("stage0_tail", priority=0):
        a1 = a_buf.index(slot).slice(0, 1024, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        as1 = mxfp8_load_scale(
            as_buf, slot, 4, SCALE_A_LAYOUT, 1024, 8, 4)
        b1 = b_buf.index(slot).slice(0, 1024, 0).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs1 = mxfp8_load_scale(
            bs_buf, slot, 4, SCALE_B_LAYOUT, 1024, 8, 4)
    with gl.amd.warp_pipeline_stage("stage1_tail", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1, bs1, "e4m3", acc)
    return acc


@gluon.jit
def mxfp8_tdm_store_serial_n2(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc):
    cta_m: gl.constexpr = 256
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    cga_layout: gl.constexpr = (
        (1, 0, 0, 0), (2, 0, 0, 0),
        (0, 1, 0, 0), (0, 2, 0, 0))
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[half_n, 8]], [4, 4, cta_m, half_n],
        [3, 2, 1, 0], cga_layout)
    shared = gl.allocate_shared_memory(
        c_ptr.type.element_ty, [4, 4, cta_m, half_n], shared_layout)
    acc4 = acc.reshape((4, cta_m, 4, cta_n)).permute((0, 2, 1, 3))
    output0 = gl.amd.slice(
        acc4, [4, 4, cta_m, half_n], [0, 0, 0, 0])
    output1 = gl.amd.slice(
        acc4, [4, 4, cta_m, half_n], [0, 0, 0, half_n])
    desc = tdm.make_tensor_descriptor(
        base=c_ptr,
        shape=(M // cta_m, N // cta_n, cta_m, cta_n),
        strides=(cta_m * stride_cm, cta_n * stride_cn, stride_cm, stride_cn),
        block_shape=(4, 4, cta_m, half_n),
        layout=shared_layout)
    base_m = pid_m * 4
    base_n = pid_n * 4
    shared.store(output0.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, 0], shared)
    tdm.async_wait(0)
    shared.store(output1.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, half_n], shared)
    tdm.async_wait(0)


@gluon.jit
def mxfp8_prologue_epilogue_variant_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
        SINGLE_SLOT_PROLOGUE: gl.constexpr, PIPELINE_TAIL: gl.constexpr,
        SPLIT_EPILOGUE: gl.constexpr, WAIT_ORDER: gl.constexpr):
    gl.static_assert(gl.num_ctas() == 16)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
        M, N, 1024, 1024, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [1024, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [1024, 4])
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, 1024, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, 1024, 256], SHARED_LAYOUT_B)
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty, [2, 8, 1024], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty, [2, 8, 1024], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * 1024 * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(1024, 256),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * 1024 * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(1024, 256),
        layout=SHARED_LAYOUT_B)
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + (pid_m * 1024) // 128 * stride_scale,
        shape=(M // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(8, 1024), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * 1024) // 128 * stride_scale,
        shape=(N // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(8, 1024), layout=SHARED_SCALE_B)
    b_warp_hint: gl.constexpr = (
        0b11001100 if SPLIT_EPILOGUE else 0b00001100)

    if SINGLE_SLOT_PROLOGUE:
        mxfp8_variant_issue_data(
            a_desc, b_desc, a_buf, b_buf, 0, 0, b_warp_hint)
        mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 0, 0)
        tdm.async_wait(0)
        snapshot_cluster_wait()
        mxfp8_variant_issue_data(
            a_desc, b_desc, a_buf, b_buf, 1, 1, b_warp_hint)
        mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 1, 1)
    else:
        for prefetch_idx in gl.static_range(2):
            mxfp8_variant_issue_data(
                a_desc, b_desc, a_buf, b_buf, prefetch_idx, prefetch_idx,
                b_warp_hint)
            mxfp8_issue_leading4_scale(
                as_desc, bs_desc, as_buf, bs_buf, prefetch_idx, prefetch_idx)
        tdm.async_wait(2)
        snapshot_cluster_wait()

    acc = gl.zeros((1024, 1024), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 256)
    gl.assume(iter_max >= 2)
    for tile_idx in range(0, iter_max - 2):
        slot = tile_idx % 2
        acc = mxfp8_variant_consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, slot, a_desc, b_desc, as_desc,
            bs_desc, tile_idx + 2, acc, dot_a, dot_b, scale_a_layout,
            scale_b_layout, b_warp_hint, WAIT_ORDER)
        tdm.async_wait(2)

    penultimate_slot = (iter_max - 2) % 2
    if PIPELINE_TAIL:
        acc = mxfp8_consume_pipelined_tail(
            a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, dot_a,
            dot_b, scale_a_layout, scale_b_layout)
    else:
        acc = mxfp8_consume_tile(
            a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, dot_a,
            dot_b, scale_a_layout, scale_b_layout)
    tdm.async_wait(0)
    snapshot_cluster_wait()
    last_slot = (iter_max - 1) % 2
    if PIPELINE_TAIL:
        acc = mxfp8_consume_pipelined_tail(
            a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b,
            scale_a_layout, scale_b_layout)
    else:
        acc = mxfp8_consume_tile(
            a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b,
            scale_a_layout, scale_b_layout)

    if SPLIT_EPILOGUE:
        # The output allocation aliases cluster-distributed input LDS. Ensure
        # every CTA has completed its final operand reads before repurposing it.
        snapshot_cluster_wait()
        # End all operand lifetimes at the same point so the shared-memory
        # allocator can place the smaller output stage in the freed arena.
        a_buf._keep_alive()
        b_buf._keep_alive()
        as_buf._keep_alive()
        bs_buf._keep_alive()
        mxfp8_tdm_store_serial_n2(
            c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc)
    else:
        snapshot_tdm_store_full_tile(
            c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
            WMMA_LAYOUT, 1024, 1024, 4)


def make_inputs(args):
    """Prepare one input set shared by every variant in this process."""
    torch.manual_seed(args.seed)
    if args.input_mode == "trig":
        a = make_trig_tensor((args.M, args.K), torch.float8_e4m3fn, False)
        b = make_trig_tensor((args.K, args.N), torch.float8_e4m3fn, True)
    else:
        a = init_data("float8_e4m3", args.M, args.K)
        b = init_data("float8_e4m3", args.K, args.N)
    scale_k = args.K // 32
    a_scale_obj = MXScaleTensor(
        size=(args.M, scale_k)).random(low=1.0, high=32.0)
    b_scale_obj = MXScaleTensor(
        size=(args.N, scale_k)).random(low=1.0, high=32.0)
    reference = None
    if args.check:
        reference = torch_gemm_mxfp(
            a, b, a_scale_obj, b_scale_obj, 32,
            args.M, args.N, args.K).to(torch.bfloat16)
    a_d = a.contiguous().cuda()
    b_d = b.T.contiguous().cuda()
    as_d = pack_scale(a_scale_obj.data, 4).cuda()
    bs_d = pack_scale(b_scale_obj.data, 4).cuda()
    c_d = torch.empty(
        (args.M, args.N), dtype=torch.bfloat16, device="cuda")
    return a_d, b_d, as_d, bs_d, c_d, reference


def make_launch(name, tensors, args):
    """Bind one compile-time schedule choice to a zero-argument launch."""
    a_d, b_d, as_d, bs_d, c_d, _ = tensors
    layouts = (
        build_canonical_mxfp8_layouts()
        if name == "canonical"
        else build_mxfp8_variant_layouts(VARIANTS[name][2]))
    grid = (triton.cdiv(args.M, 1024) * triton.cdiv(args.N, 1024), 1)

    def launch():
        shared_a, shared_b, shared_as, shared_bs, wmma = layouts
        common = (
            a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
            a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1), bs_d.stride(0),
        )
        constexpr = dict(
            GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
            SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
            num_warps=8, waves_per_eu=2, num_ctas=16,
            llvm_fn_attrs=AGPR_ATTRS)
        if name == "canonical":
            return fp8_scaled_cluster_bf16_style_kernel_gfx1250[grid](
                *common, **constexpr)
        single_prologue, pipeline_tail, split_epilogue, wait_order = (
            VARIANTS[name])
        return mxfp8_prologue_epilogue_variant_gfx1250[grid](
            *common, SINGLE_SLOT_PROLOGUE=single_prologue,
            PIPELINE_TAIL=pipeline_tail, SPLIT_EPILOGUE=split_epilogue,
            WAIT_ORDER=wait_order,
            **constexpr)

    return launch


def benchmark(launch, args):
    """Return wall-clock graph-replay latency after a fixed warmup."""
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(args.iters_per_graph):
                launch()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.replays):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / (args.iters_per_graph * args.replays)


def print_resources(compiled):
    """Print code-object resource directives for occupancy comparison."""
    shared = getattr(compiled.metadata, "shared", None)
    if shared is not None:
        print(f"dynamic_shared_bytes {shared}", flush=True)
    assembly = compiled.asm.get("amdgcn", "")
    keys = (
        ".amdhsa_next_free_vgpr", ".amdhsa_next_free_sgpr",
        ".amdhsa_group_segment_fixed_size", ".amdhsa_private_segment_fixed_size",
    )
    for line in assembly.splitlines():
        if any(key in line for key in keys):
            print(line.strip(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=DEFAULT_VARIANTS,
        help="schedule variants to run (default: canonical optimized schedule)")
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-K", type=int, default=65536)
    parser.add_argument("--input-mode", choices=("random", "trig"),
                        default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--compare-baseline", action="store_true",
        help="compare each variant with the canonical GPU result (supports long K)")
    parser.add_argument("--print-resources", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters-per-graph", type=int, default=50)
    parser.add_argument("--replays", type=int, default=20)
    args = parser.parse_args()
    if (not args.check and not args.compare_baseline and not args.benchmark
            and not args.print_resources):
        parser.error(
            "select --check, --compare-baseline, --print-resources, "
            "and/or --benchmark")
    if args.M % 1024 or args.N % 1024 or args.K % 256:
        parser.error("M/N must be divisible by 1024 and K by 256")
    if args.K // 256 < 2:
        parser.error("K must contain at least two BK256 tiles")

    tensors = make_inputs(args)
    baseline_output = None
    if args.compare_baseline:
        tensors[4].zero_()
        make_launch("canonical", tensors, args)()
        torch.cuda.synchronize()
        baseline_output = tensors[4].clone()
    for name in args.variants:
        print(f"\n=== {name} ===", flush=True)
        launch = make_launch(name, tensors, args)
        if args.print_resources:
            print_resources(launch())
        if args.check:
            tensors[4].zero_()
            launch()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                tensors[4].cpu(), tensors[5], rtol=1e-2, atol=5e-1)
            print("result verified", flush=True)
        if args.compare_baseline:
            tensors[4].zero_()
            launch()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                tensors[4], baseline_output, rtol=1e-2, atol=5e-1)
            print("matches baseline", flush=True)
        if args.benchmark:
            seconds = benchmark(launch, args)
            tflops = 2 * args.M * args.N * args.K / seconds / 1e12
            print(f"per-iter: {seconds * 1e6:.2f} us", flush=True)
            print(f"TFLOPS  : {tflops:.3f}", flush=True)


if __name__ == "__main__":
    main()
