import argparse
import time

import torch
import triton

try:
    from .f16_gemm_warp_pipeline_gfx1250 import _launch_gemm as _launch_f16_gemm
    from .mxfp_gemm_gfx1250 import (
        MXScaleTensor,
        init_data,
        mxgemm_tdm_pipelined_kernel,
        pack_scale,
        torch_gemm_mxfp,
    )
except ImportError:
    from f16_gemm_warp_pipeline_gfx1250 import _launch_gemm as _launch_f16_gemm
    from mxfp_gemm_gfx1250 import (
        MXScaleTensor,
        init_data,
        mxgemm_tdm_pipelined_kernel,
        pack_scale,
        torch_gemm_mxfp,
    )


MXFP_DTYPE_TO_KERNEL = {
    "float8_e5m2": "e5m2",
    "float8_e4m3": "e4m3",
    "float4": "e2m1",
}


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
    if triton.cdiv(args.K, args.BK) < args.num_buffers:
        raise ValueError("K/BLOCK_K must be at least NUM_BUFFERS")

    torch.manual_seed(args.seed)
    a = torch.randn((args.M, args.K), dtype=torch.float16, device="cuda")
    b = torch.randn((args.K, args.N), dtype=torch.float16)
    if args.transpose_b:
        b = b.T.contiguous()
    b = b.cuda()
    c = torch.zeros((args.M, args.N), dtype=torch.float32, device="cuda")

    def launch():
        return _launch_f16_gemm(args.BM, args.BN, args.BK, args.num_buffers, args.transpose_b, args.M, args.N, args.K,
                                a, b, c, args.kernelB)

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
    if args.num_warps != 8:
        raise ValueError("The gfx1250 MXFP warp-pipelined path requires --num-warps 8")
    if args.num_buffers not in (3, 4):
        raise ValueError("The gfx1250 MXFP 8-warp path requires --num-buffers 3 or 4")
    if args.schedule not in ("baseline", "sliceK"):
        raise ValueError("PINGPONG warp-pipeline supports --schedule baseline or sliceK")
    if args.async_copy_scale:
        raise ValueError("The 8-warp MXFP warp-pipelined path does not support --async-copy-scale")
    if args.K < args.BK or triton.cdiv(args.K, args.BK) < args.num_buffers - 1:
        raise ValueError("K/BLOCK_K must provide enough tiles for the requested pipeline depth")

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
    grid = (triton.cdiv(args.M, args.BM) * triton.cdiv(args.N, args.BN), 1, 1)

    def launch():
        return mxgemm_tdm_pipelined_kernel[grid](
            a_d, b_d, c_d, a_scale_d, b_scale_d, args.M, args.N, args.K, stride_am, stride_ak, stride_bk, stride_bn,
            stride_cm, stride_cn, stride_scale, MXFP_DTYPE_TO_KERNEL[args.dtype_a], MXFP_DTYPE_TO_KERNEL[args.dtype_b],
            args.scale_block, args.BM, args.BN, args.BK, args.group_size_m, args.transpose_b, args.num_buffers,
            args.scale_preshuffled, False, args.with_a_scale, args.schedule, args.num_warps, True,
            L2_PREFETCH_DISTANCE=args.l2_prefetch_distance,
            RESOLVE_PARTITION_CONFLICTS=args.resolve_partition_conflicts, num_warps=args.num_warps, num_ctas=1,
            waves_per_eu=args.num_warps // 4)

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
    parser.add_argument("-BK", type=int, default=128, help="BLOCK_K")
    parser.add_argument("--dtype-a", default="float16", choices=["float16", "float8_e4m3", "float8_e5m2", "float4"])
    parser.add_argument("--dtype-b", default=None, choices=["float16", "float8_e4m3", "float8_e5m2", "float4"])
    parser.add_argument("--num-warps", type=int, default=8, choices=[4, 8])
    parser.add_argument("--num-buffers", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--group-size-m", type=int, default=8, choices=[1, 2, 4, 8])
    parser.add_argument("--transpose-b", action="store_true", default=True)
    parser.add_argument("--no-transpose-b", action="store_false", dest="transpose_b")
    parser.add_argument("--kernelB", action="store_true", help="Use the fp16 kernelB variant")
    parser.add_argument("--scale-block", type=int, default=32, help="MXFP scale block")
    parser.add_argument("--scale-preshuffled", action="store_true", help="Use preshuffled MXFP scale layout")
    parser.add_argument("--with-a-scale", action="store_true", help="Use A scales on the MXFP path")
    parser.add_argument("--async-copy-scale", action="store_true", help="Reserved for non-warp-pipelined MXFP schedules")
    parser.add_argument("--schedule", default="baseline", choices=["baseline", "sliceK"],
                        help="MXFP warp-pipelined schedule")
    parser.add_argument("--l2-prefetch-distance", type=int, default=-1, choices=[-1, 0, 1, 2, 3, 4, 5])
    parser.add_argument("--resolve-partition-conflicts", action="store_true")
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
        f"{args.transpose_b=}, {args.schedule=}"
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
