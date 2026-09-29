# Five-stage MXFP8 current performance — September 29, 2026

Updated after the paired two-buffer output-store experiment. The default now selects two LDS output buffers for nine BK256 shapes and the original store for three BK512 shapes. All variants passed correctness with zero spills; input generation and the five-stage compute schedule are unchanged.

Latest timings below are from paired comparisons on identical rotating inputs, including the 20-pair confirmation for the final shape. Base/APRE remain the supplied historical numbers. Full measurements and implementation details: [two-buffer results](two_buffer/RESULTS.md).

| M×N×K | Base µs | APRE µs | Retained µs | TFLOPS | CTA tile | Cluster | Output buffers |
|---|---:|---:|---:|---:|---|---|---:|
| 512×6144×7168 | 15.94 | 16.31 | 22.616 | 1,994.0 | 256×256×256 | 2×2 | 2 |
| 512×7168×3072 | 8.23 | 8.27 | 7.282 | 3,096.3 | 128×128×512 | 4×4 | 1 |
| 512×8192×1536 | 5.91 | 5.81 | 5.139 | 2,507.5 | 128×128×512 | 4×4 | 1 |
| 512×2048×7168 | 9.69 | 9.47 | 10.190 | 1,475.2 | 64×64×512 | 4×4 | 1 |
| 512×65536×1536 | 20.13 | 20.08 | 19.745 | 5,220.5 | 256×256×256 | 2×2 | 2 |
| 512×7168×16384 | 27.12 | 25.57 | 45.428 | 2,647.2 | 256×256×256 | 2×2 | 2 |
| 16384×6144×7168 | 194.68 | 194.05 | 179.676 | 8,031.7 | 256×256×256 | 4×4 | 2 |
| 16384×7168×3072 | 106.14 | 105.69 | 99.094 | 7,281.5 | 256×256×256 | 4×4 | 2 |
| 16384×8192×1536 | 69.27 | 70.88 | 69.800 | 5,907.1 | 256×256×256 | 4×4 | 2 |
| 16384×2048×7168 | 68.20 | 65.31 | 60.257 | 7,983.1 | 256×256×256 | 4×4 | 2 |
| 16384×65536×1536 | 600.29 | 562.03 | 553.084 | 5,963.9 | 256×256×256 | 4×4 | 2 |
| 16384×7168×16384 | 546.19 | 544.31 | 455.827 | 8,442.4 | 256×256×256 | 4×4 | 2 |

Reproduce with `/usr/local/bin/gpu-lock bash experiments/mxfp8_five_shapes_20260929/run.sh`. Use `--output-store serial` or `--output-store double` to force a variant. `results.json` contains the retained results; the original run is preserved in `results_initial_serial.json`.

---

# Initial serial-output baseline — September 29, 2026

Source: `third_party/amd/python/examples/gluon/gfx1250_gemm/kernels_mxfp8_five_shapes_0929.py`.

All 12 full outputs passed FP32 reference checks (`rtol=0.008`, `atol=1e-5`); maximum relative error was 0.003893, consistent with BF16 output rounding. All configurations compiled with zero spills. No native/compiler files changed.

## Kernel configuration

- 256×256×256 CTA: retained September 27 five-stage, delayed-wait, parent-B kernel; six-tile unroll, two data slots, three scale slots. Generalized B descriptor and rank-3 load layouts to 2×2 clusters; 4×4 remains supported.
- 128×128×512 and 64×64×512 CTA: full-accumulator BK512 path adapted to five stages: load low two K128 chunks → compute low → load high two chunks/arrive → compute high → wait/refill. Six-tile unroll plus cleanup and two prefetched slots. Both use unshuffled scales to support the 64-row CTA layout.
- No assembly override or matrix B reuse in these results.

## Inputs and measurement

- GPU-generated A `[M,K]` and B `[N,K]`: `(torch.rand(..., float32) / 10).to(torch.float8_e4m3fn)`.
- GPU-generated A scales `[M,K/128]` and B scales `[N/128,K/128]`; conversion matches local AITER `f32_to_mx_e8m0_scale(... * 448, dtype=FP8_E4M3)`, default RoundUp. This normalizes by 448 before rounding to an E8M0 power of two.
- Repeat scale codes across four K32 groups; repeat B codes across 128 N rows. Pack scales only for BK256. Input generation/expansion/packing excluded from timing.
- Seed 42, one generated set per shape, then 100 identical copies of A, B, packed/expanded scales, and BF16 output at distinct GPU addresses. Every shape fit all 100 copies.
- 20 warmup launches; graph of 100 launches rotating copies; three warmup graph replays; median of five timed replays using GPU events. Times below are per launch.
- Copy cap is 100, with a fallback budget leaving at least 2 GiB or half the currently free memory (whichever is larger). It did not reduce the count in this run.
- GPU runs serialized with `/usr/local/bin/gpu-lock`; `.rock` Python with explicit `PYTHONPATH=/home/jungpark/mnt/wpindex/triton/python`; default expert scheduling enabled.
- Base and APRE are user-supplied historical results, not remeasured here. Percent change below compares latency to the lower of those two numbers.

| M×N×K | CTA tile | Cluster | Base µs | APRE µs | Five-stage µs | TFLOPS | Latency change |
|---|---|---|---:|---:|---:|---:|---:|
| 512×6144×7168 | 256×256×256 | 2×2 | 15.94 | 16.31 | 23.089 | 1,953.2 | +44.9% |
| 512×7168×3072 | 128×128×512 | 4×4 | 8.23 | 8.27 | 7.250 | 3,110.3 | -11.9% |
| 512×8192×1536 | 128×128×512 | 4×4 | 5.91 | 5.81 | 5.027 | 2,563.1 | -13.5% |
| 512×2048×7168 | 64×64×512 | 4×4 | 9.69 | 9.47 | 10.230 | 1,469.4 | +8.0% |
| 512×65536×1536 | 256×256×256 | 2×2 | 20.13 | 20.08 | 20.521 | 5,023.1 | +2.2% |
| 512×7168×16384 | 256×256×256 | 2×2 | 27.12 | 25.57 | 45.971 | 2,616.0 | +79.8% |
| 16384×6144×7168 | 256×256×256 | 4×4 | 194.68 | 194.05 | 181.802 | 7,937.8 | -6.3% |
| 16384×7168×3072 | 256×256×256 | 4×4 | 106.14 | 105.69 | 100.684 | 7,166.5 | -4.7% |
| 16384×8192×1536 | 256×256×256 | 4×4 | 69.27 | 70.88 | 71.978 | 5,728.4 | +3.9% |
| 16384×2048×7168 | 256×256×256 | 4×4 | 68.20 | 65.31 | 61.188 | 7,861.6 | -6.3% |
| 16384×65536×1536 | 256×256×256 | 4×4 | 600.29 | 562.03 | 566.908 | 5,818.5 | +0.9% |
| 16384×7168×16384 | 256×256×256 | 4×4 | 546.19 | 544.31 | 456.529 | 8,429.5 | -16.1% |

## Reproduction

```bash
/usr/local/bin/gpu-lock bash experiments/mxfp8_five_shapes_20260929/run.sh --output-store serial
```

Use `--shape-index 0 1` for selected rows, `--check-only` for correctness, and `--output <path>` to retain a separate result JSON. The JSON contains individual samples, copy sizes/counts, resources, and maximum reference errors. `benchmark.log` is the raw completed run.

Compiler commit: `00496a329c9b183b86ab207683f2730973ec05db`.
Measured source SHA256: `34c5405f34d38ef50d631f68a537070bbaa41ac13443303b3ac3a373f597702c`.
