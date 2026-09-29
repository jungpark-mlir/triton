# Two-buffer output comparison — September 29, 2026

Retained two-buffer output stores for all nine BK256 shapes. The three BK512 shapes keep the original full-tile store. `--output-store auto` is the default; `serial` and `double` force either variant.

## Validation and method

Both variants passed full-output FP32 reference checks on all 12 shapes. Maximum relative error: 0.003893 (BF16 output rounding). All variants have zero spills and unchanged peak LDS/VGPR usage. Compute helpers, operand layouts, six-tile unrolling, and input generation are unchanged.

Same seed-42 FlyDSL-pattern inputs and same 100 GPU addresses for both variants. Each graph rotates through 100 identical copies of A, B, scales, and output. 20 warmup launches, three graph warmups, then ten paired measurements with alternating execution order. The final shape was confirmed separately with 20 pairs because its gain is small. The table uses the confirmation for that row. B reuse remains disabled; compiler is wpindex.

BK256 stores contiguous N halves from independent LDS buffers. For BK512, the tested two-buffer variant splits alternating 16-column strips, using a five-dimensional descriptor to place them in the output. The strip selector is held in registers, avoiding cross-warp layout conversion. It was not selected for any BK512 shape.

All saved candidate assemblies contain two `tensor_store_from_lds` operations with no `s_wait_tensorcnt` between them. The second buffer is filled after the first transfer is issued. Peak LDS is unchanged because output storage reuses the dead input arena.

| M×N×K | Original store µs | Two buffers µs | Two-buffer latency reduction | Wins | Retained | TFLOPS |
|---|---:|---:|---:|---:|---|---:|
| 512×6144×7168 | 23.109 | 22.616 | +2.13% | 10/10 | double | 1,994.0 |
| 512×7168×3072 | 7.282 | 8.171 | -12.20% | 0/10 | serial | 3,096.3 |
| 512×8192×1536 | 5.139 | 6.106 | -18.82% | 0/10 | serial | 2,507.5 |
| 512×2048×7168 | 10.190 | 10.235 | -0.45% | 4/10 | serial | 1,475.2 |
| 512×65536×1536 | 20.545 | 19.745 | +3.89% | 10/10 | double | 5,220.5 |
| 512×7168×16384 | 45.986 | 45.428 | +1.21% | 10/10 | double | 2,647.2 |
| 16384×6144×7168 | 182.471 | 179.676 | +1.53% | 10/10 | double | 8,031.7 |
| 16384×7168×3072 | 100.974 | 99.094 | +1.86% | 10/10 | double | 7,281.5 |
| 16384×8192×1536 | 72.286 | 69.800 | +3.44% | 10/10 | double | 5,907.1 |
| 16384×2048×7168 | 61.268 | 60.257 | +1.65% | 10/10 | double | 7,983.1 |
| 16384×65536×1536 | 567.318 | 553.084 | +2.51% | 10/10 | double | 5,963.9 |
| 16384×7168×16384 | 457.540 | 455.827 | +0.37% | 17/20 | double | 8,442.4 |

Positive reduction means two buffers are faster. These are paired measurements; use them to assess the output-store change rather than comparing different runs.

## Reproduction

Run the retained per-shape defaults:

```bash
/usr/local/bin/gpu-lock bash experiments/mxfp8_five_shapes_20260929/run.sh --output-store auto
```

Repeat the comparison:

```bash
/usr/local/bin/gpu-lock bash experiments/mxfp8_five_shapes_20260929/two_buffer/run.sh
```

Raw samples: `paired.json` (row 0), `remaining.json` (rows 1–11), `confirm11.json` (final row confirmation), and combined `paired_all.json`. Assemblies are `<index>_<serial|double>.amdgcn`. `baseline.py` preserves the original source; `tested_source.py` preserves the candidate source before updating host dispatch defaults.

Generated `.amdgcn` and `.log` files remain local and are excluded from Git. The comparison runner regenerates assembly; redirect its output to retain a log. JSON timing samples and source snapshots are tracked.
