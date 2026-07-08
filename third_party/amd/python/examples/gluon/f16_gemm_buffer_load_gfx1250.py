import time

import torch

import triton
from triton._internal_testing import is_hip_gfx1250
from triton.experimental import gluon
import triton.experimental.gluon.language as gl

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
def _load_ab_tile(a_ptr, b_ptr, tile_k, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, offs_m, offs_n, a_k_base,
                  b_k_base, BLOCK_K: gl.constexpr, dot_layout_a: gl.constexpr, dot_layout_b: gl.constexpr):
    k_start = tile_k * BLOCK_K
    offs_ak = k_start + a_k_base
    offs_bk = k_start + b_k_base

    a_offsets = offs_m[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_mask = (offs_m[:, None] < M) & (offs_ak[None, :] < K)
    b_mask = (offs_bk[:, None] < K) & (offs_n[None, :] < N)

    a = gl.amd.gfx1250.buffer_load(a_ptr, a_offsets, mask=a_mask, other=0.0)
    b = gl.amd.gfx1250.buffer_load(b_ptr, b_offsets, mask=b_mask, other=0.0)
    a = gl.convert_layout(a, dot_layout_a)
    b = gl.convert_layout(b, dot_layout_b)
    return a, b


@gluon.jit
def f16_gemm_buffer_load_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                                stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                                GROUP_SIZE_M: gl.constexpr, NUM_BUFFERS: gl.constexpr, NUM_WARPS: gl.constexpr):
    a_dtype: gl.constexpr = a_ptr.type.element_ty
    b_dtype: gl.constexpr = b_ptr.type.element_ty
    gl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    gl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    gl.static_assert(NUM_BUFFERS >= 1, "NUM_BUFFERS must be at least 1")
    gl.static_assert(NUM_BUFFERS <= 3, "This register-buffered example supports up to 3 buffers")
    gl.static_assert(NUM_WARPS == 4 or NUM_WARPS == 8, "NUM_WARPS must be 4 or 8")
    gl.static_assert(BLOCK_M % 16 == 0 and BLOCK_N % 16 == 0 and BLOCK_K % 32 == 0)

    if NUM_WARPS == 4:
        WARP_BASES: gl.constexpr = [[0, 1], [1, 0]]
    else:
        WARP_BASES: gl.constexpr = [[0, 1], [1, 0], [2, 0]]

    LOAD_LAYOUT: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])
    dot_layout_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=WMMA_LAYOUT, k_width=8)
    dot_layout_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=WMMA_LAYOUT, k_width=8)

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LOAD_LAYOUT))
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, LOAD_LAYOUT))
    a_k_base = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, LOAD_LAYOUT))
    b_k_base = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, LOAD_LAYOUT))

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    num_k_tiles = gl.cdiv(K, BLOCK_K)

    # NUM_BUFFERS is the register prefetch distance. Each loop iteration issues
    # one future tile load and computes exactly one current tile.
    if NUM_BUFFERS == 1:
        a0, b0 = _load_ab_tile(a_ptr, b_ptr, 0, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, offs_m, offs_n,
                               a_k_base, b_k_base, BLOCK_K, dot_layout_a, dot_layout_b)

        for k_iter in range(0, num_k_tiles):
            a_next, b_next = _load_ab_tile(a_ptr, b_ptr, k_iter + 1, M, N, K, stride_am, stride_ak, stride_bk,
                                           stride_bn, offs_m, offs_n, a_k_base, b_k_base, BLOCK_K, dot_layout_a,
                                           dot_layout_b)
            accumulator = gl.amd.gfx1250.wmma(a0, b0, accumulator)
            a0 = a_next
            b0 = b_next
    elif NUM_BUFFERS == 2:
        a0, b0 = _load_ab_tile(a_ptr, b_ptr, 0, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, offs_m, offs_n,
                               a_k_base, b_k_base, BLOCK_K, dot_layout_a, dot_layout_b)
        a1, b1 = _load_ab_tile(a_ptr, b_ptr, 1, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, offs_m, offs_n,
                               a_k_base, b_k_base, BLOCK_K, dot_layout_a, dot_layout_b)

        for k_iter in range(0, num_k_tiles):
            a_next, b_next = _load_ab_tile(a_ptr, b_ptr, k_iter + 2, M, N, K, stride_am, stride_ak, stride_bk,
                                           stride_bn, offs_m, offs_n, a_k_base, b_k_base, BLOCK_K, dot_layout_a,
                                           dot_layout_b)
            accumulator = gl.amd.gfx1250.wmma(a0, b0, accumulator)
            a0 = a1
            b0 = b1
            a1 = a_next
            b1 = b_next
    else:
        a0, b0 = _load_ab_tile(a_ptr, b_ptr, 0, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, offs_m, offs_n,
                               a_k_base, b_k_base, BLOCK_K, dot_layout_a, dot_layout_b)
        a1, b1 = _load_ab_tile(a_ptr, b_ptr, 1, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, offs_m, offs_n,
                               a_k_base, b_k_base, BLOCK_K, dot_layout_a, dot_layout_b)
        a2, b2 = _load_ab_tile(a_ptr, b_ptr, 2, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, offs_m, offs_n,
                               a_k_base, b_k_base, BLOCK_K, dot_layout_a, dot_layout_b)

        for k_iter in range(0, num_k_tiles):
            a_next, b_next = _load_ab_tile(a_ptr, b_ptr, k_iter + 3, M, N, K, stride_am, stride_ak, stride_bk,
                                           stride_bn, offs_m, offs_n, a_k_base, b_k_base, BLOCK_K, dot_layout_a,
                                           dot_layout_b)
            accumulator = gl.amd.gfx1250.wmma(a0, b0, accumulator)
            a0 = a1
            b0 = b1
            a1 = a2
            b1 = b2
            a2 = a_next
            b2 = b_next

    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT))
    c_offsets = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.gfx1250.buffer_store(accumulator, c_ptr, c_offsets, mask=c_mask)


def _make_inputs(M, N, K, dtype=torch.float16):
    a = torch.randn((M, K), dtype=dtype)
    b = torch.randn((K, N), dtype=dtype)
    c = torch.zeros((M, N), dtype=torch.float32)
    return a, b, c


def run_f16_gemm_buffer_load(M=512, N=512, K=512, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_SIZE_M=8,
                             NUM_BUFFERS=1, NUM_WARPS=8, dtype=torch.float16, seed=0, check=True):
    if triton.cdiv(K, BLOCK_K) < 1:
        raise ValueError("K/BLOCK_K must have at least one tile")
    if NUM_BUFFERS < 1:
        raise ValueError("NUM_BUFFERS must be at least 1")

    torch.manual_seed(seed)
    a, b, c = _make_inputs(M, N, K, dtype)
    a_d = a.cuda()
    # Keep B as [N, K] so each logical B[k, n] has contiguous K in memory.
    b_d = b.T.contiguous().cuda()
    c_d = c.cuda()

    stride_am, stride_ak = a_d.stride(0), a_d.stride(1)
    stride_bk, stride_bn = b_d.stride(1), b_d.stride(0)
    stride_cm, stride_cn = c_d.stride(0), c_d.stride(1)

    grid = [triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1]
    f16_gemm_buffer_load_kernel[grid](
        a_d, b_d, c_d, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M, BLOCK_N,
        BLOCK_K, GROUP_SIZE_M, NUM_BUFFERS, NUM_WARPS, num_warps=NUM_WARPS, waves_per_eu=NUM_WARPS // 4)

    if check:
        torch.testing.assert_close(c_d.cpu(), a.to(torch.float32) @ b.to(torch.float32), rtol=1e-3, atol=1e-2)
    return c_d


def benchmark_f16_gemm_buffer_load(M=512, N=512, K=512, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_SIZE_M=8,
                                   NUM_BUFFERS=1, NUM_WARPS=8, dtype=torch.float16, seed=0, warmup=10, rep=100):
    torch.manual_seed(seed)
    a, b, c = _make_inputs(M, N, K, dtype)
    a_d = a.cuda()
    b_d = b.T.contiguous().cuda()
    c_d = c.cuda()

    stride_am, stride_ak = a_d.stride(0), a_d.stride(1)
    stride_bk, stride_bn = b_d.stride(1), b_d.stride(0)
    stride_cm, stride_cn = c_d.stride(0), c_d.stride(1)
    grid = [triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1]

    def launch():
        f16_gemm_buffer_load_kernel[grid](
            a_d, b_d, c_d, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M, BLOCK_N,
            BLOCK_K, GROUP_SIZE_M, NUM_BUFFERS, NUM_WARPS, num_warps=NUM_WARPS, waves_per_eu=NUM_WARPS // 4)

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(rep):
        launch()
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start
    per_iter_s = elapsed_s / rep
    tflops = (2 * M * N * K) / per_iter_s / 1e12
    print(f"per-iter: {per_iter_s * 1e6:.2f} us")
    print(f"TFLOPS  : {tflops:.3f}")


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
@pytest.mark.parametrize("NUM_BUFFERS", [1, 2, 3])
def test_runtime_f16_gemm_buffer_load(NUM_BUFFERS):
    run_f16_gemm_buffer_load(NUM_BUFFERS=NUM_BUFFERS)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=512)
    parser.add_argument("-N", type=int, default=512)
    parser.add_argument("-K", type=int, default=512)
    parser.add_argument("-BM", "--block-m", type=int, default=128)
    parser.add_argument("-BN", "--block-n", type=int, default=128)
    parser.add_argument("-BK", "--block-k", type=int, default=64)
    parser.add_argument("--numwarps", "--num-warps", dest="num_warps", type=int, default=8, choices=[4, 8])
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--benchmark", action="store_true", help="Benchmark the kernel and report TFLOPS.")
    mode_group.add_argument("--check", action="store_true", help="Check output against torch.")
    args = parser.parse_args()

    GROUP_SIZE_M = 8
    NUM_BUFFERS = 3
    input_dtype = torch.float16
    print(
        f"(M={args.M}, N={args.N}, K={args.K}), (BLOCK_M={args.block_m}, BLOCK_N={args.block_n}, BLOCK_K={args.block_k}), "
        f"NUM_WARPS={args.num_warps}, NUM_BUFFERS={NUM_BUFFERS}, dtype={input_dtype}")
    if args.benchmark:
        benchmark_f16_gemm_buffer_load(M=args.M, N=args.N, K=args.K, BLOCK_M=args.block_m, BLOCK_N=args.block_n,
                                       BLOCK_K=args.block_k, GROUP_SIZE_M=GROUP_SIZE_M, NUM_BUFFERS=NUM_BUFFERS,
                                       NUM_WARPS=args.num_warps, dtype=input_dtype)
    else:
        run_f16_gemm_buffer_load(M=args.M, N=args.N, K=args.K, BLOCK_M=args.block_m, BLOCK_N=args.block_n,
                                 BLOCK_K=args.block_k, GROUP_SIZE_M=GROUP_SIZE_M, NUM_BUFFERS=NUM_BUFFERS,
                                 NUM_WARPS=args.num_warps, dtype=input_dtype, check=True)
