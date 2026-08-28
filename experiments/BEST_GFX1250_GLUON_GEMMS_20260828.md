# Best GFX1250 Gluon GEMMs

## Canonical snapshot

`experiments/best_gfx1250_gluon_gemms_20260828.py` is a source-contained
snapshot of the best retained Gluon kernels for BF16, MXFP8, and MXFP4. It
defines the kernels, layouts, data preparation, correctness checks, and
benchmark harness locally instead of importing experimental kernel files.

The fixed schedules are:

- BF16: 4x4 clustered KernelC, 1024x1024x128 aggregate tile, two partitioned
  LDS buffers, a refill-side K64 WMMA split into two K32 WMMAs, and BF16
  output.
- MXFP8: 4x4 BF16-style cluster, 1024x1024x256 aggregate tile, leading-four
  TDM, E4M3 inputs with E8M0 block-32 scales, BF16 output, and refill before
  the final register-resident K128 WMMA.
- MXFP4: 4x4 cluster, 1024x1024x512 aggregate tile, K256 warp-pipeline stages,
  cluster synchronization overlapped with the final K128 scaled WMMA, packed
  E2M1 inputs with E8M0 block-32 scales, and BF16 output.

All three use eight warps per CTA and two LDS buffers.

## Correctness

The source-contained selectors JIT-compiled and passed:

- BF16-output random-input check at 1024x1024x4096.
- MXFP8 random-input check at 1024x1024x1024.
- MXFP8 long-K random-input check at 1024x1024x65536 before extraction.
- MXFP4 random-input check at 1024x1024x1024.

## August 28 rebenchmark

The target was M=N=4096 and K=65536 with trigonometric inputs. Each Gluon
result is the median of three runs under `gpu-lock`. The MXFP measurements
were interleaved with hipBLASLt; the BF16-output Gluon result was rerun in
three consecutive trials and compared with the retained hipBLASLt result.
Gluon timing used 10 warmups followed by 1,000 graph-contained launches
(50 launches per graph and 20 replays). hipBLASLt used 1,000 iterations.

- BF16 Gluon with BF16 output: 688.74 us, 3.193 PFLOPS.
- BF16 hipBLASLt with BF16 C/D: 663.37 us, 3.315 PFLOPS, solution 114.
  With matched trigonometric BF16 inputs and BF16 output, Gluon has 3.68%
  lower throughput.
- BF16 hipBLASLt with output-matched FP32 C/D: 14,722.4 us,
  0.149 PFLOPS, fallback solution 121. hipBLASLt currently has no competitive
  FP32-output solution for this case. Use `--bf16-fp32-output` to select the
  non-default Gluon FP32-output path.
- MXFP8 Gluon: 264.23 us, 8.322 PFLOPS.
- MXFP8 hipBLASLt: 267.15 us, 8.232 PFLOPS, solution 222.
  Gluon is 1.10% faster.
- MXFP4 Gluon: 141.25 us, 15.569 PFLOPS.
- MXFP4 hipBLASLt: 140.23 us, 15.681 PFLOPS, solution 153.
  Gluon is 0.72% slower.

The hipBLASLt build reported git version `caca6b6fc9`.

## Gluon commands

```bash
source /home/jungpark/.rock/bin/activate
export PYTHONPATH=/home/jungpark/mnt/wpindex/triton/python

/usr/local/bin/gpu-lock python3 \
  experiments/best_gfx1250_gluon_gemms_20260828.py \
  --kernel all -M 4096 -N 4096 -K 65536 \
  --benchmark --warmup 10 --probe-iters 20 \
  --iters-per-graph 50 --replays 20
```

Run a smaller correctness check by selecting one kernel:

```bash
python3 experiments/best_gfx1250_gluon_gemms_20260828.py \
  --kernel mxfp8 -M 1024 -N 1024 -K 1024 \
  --input-mode random --check
```

## hipBLASLt commands

Run these from the hipBLASLt repository:

```bash
build/release/clients/hipblaslt-bench \
  -r bf16_r --transA T \
  -m 4096 -n 4096 -k 65536 -i 1000 \
  --print_kernel_info --initialization trig_float

build/release/clients/hipblaslt-bench \
  -r f8_r --d_type bf16_r --c_type bf16_r \
  --scaleA 3 --scaleB 3 --transA T \
  -m 4096 -n 4096 -k 65536 -i 1000 \
  --print_kernel_info --initialization trig_float

build/release/clients/hipblaslt-bench \
  -r f4_r --d_type bf16_r --c_type bf16_r \
  --scaleA 3 --scaleB 3 --transA T \
  -m 4096 -n 4096 -k 65536 -i 1000 \
  --print_kernel_info --initialization trig_float
```
