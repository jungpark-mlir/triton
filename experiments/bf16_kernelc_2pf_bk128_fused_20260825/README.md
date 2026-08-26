# Isolated fused-TDM BF16 KernelC

This directory freezes the optimized `256x256x128` BF16 KernelC measured on
2026-08-25. The Python file is self-contained and does not import
`warp_pipelined_gemm_gfx1250.py`, so later tutorial edits cannot change this
reproduction.

## Kernel configuration

- Target: `gfx1250`
- GEMM tile: `BM=256`, `BN=256`, `BK=128`
- Inputs: BF16; accumulation and output: FP32
- Eight waves, two LDS buffers
- Each BK128 tile is consumed as two full-M/N BK64 WMMA halves
- A/B TDM refill is fused into the second-half WMMA stage
- CTA traversal: eight XCDs with `GROUP_SIZE_M=4`
- Optional `--cluster-4x4` mode distributes a logical `1024x1024x128`
  tile across a 4x4 CTA CGA while retaining a 256x256 output tile per CTA

The fused stage is important. A separate TDM stage created 510 periodic
no-WMMA SIMD intervals. Fusing refill after the second-half LDS reads reduced
those intervals to five non-periodic edge/outlier gaps.

## Environment

Run from the Triton repository root:

```bash
source /home/jungpark/.rock/bin/activate
export PYTHONPATH=/home/jungpark/mnt/wpindex/triton/python
```

## Correctness

The odd-tile case exercises the dynamic refill tail:

```bash
/usr/local/bin/gpu-lock python3 \
  experiments/bf16_kernelc_2pf_bk128_fused_20260825/bf16_kernelc_2pf_bk128_fused.py \
  -M 256 -N 256 -K 384 --input-mode random --check
```

Expected final line:

```text
result verified
```

## Target benchmark

```bash
/usr/local/bin/gpu-lock python3 \
  experiments/bf16_kernelc_2pf_bk128_fused_20260825/bf16_kernelc_2pf_bk128_fused.py \
  --benchmark
```

The isolated reproduction measured 705.82 us and 3115.58 TFLOPS. Three runs
from the integrated source measured 705.53, 703.96, and 701.66 us, with a
703.96 us median. The pre-fusion BK128 median was 728.33 us.

## 4x4 CTA cluster experiment

The clustered variants use the same partition-aware local WMMA layout and
two BK128 LDS slots. Because a clustered TDM refill can overwrite LDS owned
by another CTA, the standalone same-slot refill needs an overwrite-safety
rendezvous. The best schedule otherwise remains identical to standalone:
consume and refill one slot, then `tdm.async_wait(2)`.

The ordering of the rendezvous is critical. `cluster.arrive` at the end of
the LDS-load stage runs before the warp-pipeline boundary's local barrier and
is racy at full size. The correct one-slot ordering places both
`cluster.arrive` and `cluster.wait` at the beginning of the refill stage:

```text
final LDS reads -> local stage barrier -> cluster arrive/wait -> same-slot refill
```

This is selected with `--cluster-same-schedule`. The older `--cluster-4x4`
variant consumes and refills slots in pairs and remains available for
comparison.

Three BK128 slots do not fit: compilation requires 417,760 bytes of shared
memory versus the 327,680-byte hardware limit. The paired two-slot schedule
uses 278,496 bytes.

Correctness:

```bash
/usr/local/bin/gpu-lock python3 \
  experiments/bf16_kernelc_2pf_bk128_fused_20260825/bf16_kernelc_2pf_bk128_fused.py \
  --cluster-4x4 -M 1024 -N 1024 -K 4096 --input-mode random --check
```

At the full 4096x4096x65536 target, every FP32 output element matched the
standalone BK128 kernel bitwise. After correcting warp-pipeline lowering so
CTA-cluster operations no longer replace stage barriers, comparable target
tests with `/home/jungpark/mnt/wpindex/triton` measured:

- Standalone, trigonometric inputs: 705.67 us, 3116.21 TFLOPS
- Original 4x4 cluster with full TDM drain: 925.47 us, 2376.11 TFLOPS
- Split-wait 4x4 cluster: 813.56 us, 2702.95 TFLOPS
- Same-schedule 4x4 cluster: 700.75 us median, 3138.10 TFLOPS

Splitting the four-request drain improves cluster time by 12.1% and throughput
by 13.8%. Restoring the standalone refill cadence removes the remaining
regression. Three matched runs measured 708.55, 704.70, and 706.89 us
standalone versus 700.75, 700.97, and 700.00 us clustered. The clustered
median is 0.9% faster.

Reusing the hardware cluster barrier for the next generation before all waves
pass the previous wait deadlocks; the retained schedule keeps an intra-CTA
stage boundary between generations.

## Partitioned cluster-shape sweep

`--cluster-partitioned` partitions both operands into CTA-local logical pieces.
The default is two pieces (two physical partitions times one group);
`--cluster-partition-groups 2` selects the original four-piece experiment.
Both divide KernelC's four TDM producer waves, and CGA dimensions do not count
as additional TDM pieces.

The clustered partition layout required two lowering corrections:

- Partition/group bits must be composed inside the per-CTA tile, with CGA
  block bits added outside them.
- A multicast CGA layout is not square because one block dimension is
  broadcast. Partition selection for a subslice therefore uses a
  pseudoinverse rather than `LinearLayout::invert()`.

The original four-piece 4x2, 2x4, and 2x2 variants pass random
1024x1024x4096 checks and match the standalone kernel at the full
4096x4096x65536 target. Three trigonometric-input runs measured:

- Partitioned 4x2: 700.11, 699.41, 697.23 us; 699.41 us median
- Partitioned 2x4: 708.76, 706.86, 707.84 us; 707.84 us median
- Partitioned 2x2: 728.78, 730.06, 722.81 us; 728.78 us median

Four-piece 4x2 was effectively tied with the plain 4x2 and 4x4 results. The
2x4 shape gained a few microseconds, while 2x2 regressed.

ATT confirms that partitioning removes LDS conflicts but moves the exposed
latency elsewhere. Relative to the plain same-schedule 4x4 trace, partitioned
4x2 reduces `ds_load` stall from 238,498 to 154,164 cycles (-35.4%), while
local-barrier stall rises from 580,405 to 655,847 and WMMA-attributed stall
rises from 777,322 to 880,748. Total attributed stall increases from
1,871,686 to 1,980,001 cycles, and the traced GFX span is nearly unchanged
(1,171,520 versus 1,176,672 cycles). Partition avoidance therefore does not
produce a material end-to-end gain for 4x2.

### Two-piece 4x4 partitioning

Making partition/group composition CTA-local also enables partitioning for a
4x4 CGA without creating eight or sixteen TDM pieces. Both two-piece and
four-piece 4x4 layouts pass the small random check and match standalone at the
full target.

Two-piece partitioning is best:

- Four-piece 4x4: 698.65, 697.96, 698.94 us; 698.65 us median
- Two-piece 4x4: 696.84, 695.25, 694.36 us; 695.25 us median
- Two-piece A / four-piece B: 700.09, 698.29, 697.66 us; 698.29 us median
- Four-piece A / two-piece B: 695.79, 695.83, 694.83 us; 695.79 us median

The symmetric two-piece layout is about 0.8% faster than the plain 4x4 median.
ATT shows `ds_load` stall falling from 238,498 to 121,718 cycles while total
attributed stall remains nearly flat (1,871,686 to 1,879,295) and traced GFX
span rises only 0.3% (1,171,520 to 1,174,856 cycles). Four-piece partitioning
instead adds WMMA dependency stall; two pieces avoid that penalty.

The partitioned traces are retained at:

```text
/home/jungpark/mnt/att-traces/bf16-kernelc-bk128-cluster-partitioned-4x2-20260825
/home/jungpark/mnt/att-traces/bf16-kernelc-bk128-cluster-partitioned-2x4-20260825
/home/jungpark/mnt/att-traces/bf16-kernelc-bk128-cluster-partitioned-2x2-20260825
/home/jungpark/mnt/att-traces/bf16-kernelc-bk128-cluster-partitioned-4x4-20260825
/home/jungpark/mnt/att-traces/bf16-kernelc-bk128-cluster-partitioned-4x4-groups1-20260825
```

### BK32 slicing

`--cluster-bk32` splits each BK128 slot into four independently staged BK32
load/WMMA pairs. It verifies correctly, but loses to the retained BK64 schedule:

- BK64: 692.45, 693.95, 694.63 us; 693.95 us median
- BK32: 701.47, 701.39, 702.45 us; 701.47 us median

The shorter loads work in isolation: ATT `ds_load` stall drops from 121,718 to
47,840 cycles and WMMA-attributed stall drops from 777,328 to 717,140 cycles.
However, local stage-barrier executions double from 4,110 to 8,190 and their
stall rises from 772,471 to 885,679 cycles. Merged WMMA efficiency falls from
90.50% to 88.66%, so BK64 remains the default. The BK32 trace is retained at:

```text
/home/jungpark/mnt/att-traces/bf16-kernelc-bk128-cluster-partitioned-4x4-bk32-20260825
```

### BK64 follow-up

Further changes that preserve the BK64 stage count did not beat the retained
schedule:

- Splitting A/B TDM producers across low/high wave groups: 695.21 us median
- Fusing A/B into one TDM operation: 695.12 us median
- Retained low-four-wave TDM: 694.46 us median
- Removing `s_setprio`, prioritizing LDS, or raising WMMA priority: neutral or
  slower
- Moving the final WMMA between cluster arrive/wait: neutral across eight
  paired samples (695.02 versus 694.78 us median)
- `GROUP_SIZE_M=1/2`: no gain over 4; four-XCD swizzling is about 4 us slower
  than the retained eight-XCD mapping

Increasing operand lookahead is limited by registers rather than LDS:

- Loading all BK128 operands at once uses 510 VGPRs, spills 720 values into
  1,732 bytes of private memory, and takes about 10.6 ms.
- Preloading only the second A half spills 16 values and takes about 1.16 ms.
- Preloading only the second B half spills 168 values and takes about 4.36 ms.

The remaining periodic bubbles therefore cannot be removed by simply moving
more LDS operands into registers. A credible next step requires reducing the
warp-pipeline barrier cost itself, or adding a narrower row/column cluster
rendezvous so A and B multicast refills do not synchronize all 16 CTAs.

## Advanced thread trace

```bash
DIR=/home/jungpark/mnt/att-traces/bf16-kernelc-2pf-bk128-fused-isolated-20260825
mkdir -p "$DIR"

/usr/local/bin/gpu-lock rocprofv3 \
  --advanced-thread-trace \
  --att-target-cu 1 \
  --att-shader-engine-mask 0x1 \
  --kernel-include-regex 'bf16_kernelc_2pf_bk128_fused_gfx1250' \
  --kernel-iteration-range 100 \
  --output-directory "$DIR" \
  --output-file bf16_kernelc_2pf_bk128_fused_isolated_att \
  --output-format csv json \
  -- bash -lc '
    source /home/jungpark/.rock/bin/activate
    export PYTHONPATH=/home/jungpark/mnt/wpindex/triton/python
    python3 experiments/bf16_kernelc_2pf_bk128_fused_20260825/bf16_kernelc_2pf_bk128_fused.py \
      --benchmark
  '
```

The integrated fused-kernel trace is retained at:

```text
/home/jungpark/mnt/att-traces/bf16-kernelc-2pf-bk128-fused-20260825
```

It measured:

- 1,110,565 mean CU cycles
- 1629.90 MHz mean operating frequency
- 1628.98-1630.58 MHz clock range
- Five WMMA issue gaps of at least 129 cycles, down from 510
- No scratch load/store instructions

### Corrected standalone versus 4x4-cluster ATT

Matching traces from the isolated source, built with
`/home/jungpark/mnt/wpindex/triton`, are retained at:

```text
/home/jungpark/mnt/att-traces/bf16-kernelc-bk128-standalone-corrected-baseline-20260825
/home/jungpark/mnt/att-traces/bf16-kernelc-bk128-cluster-4x4-corrected-20260825
```

The profiled graph replays measured 708.52 us / 3103.69 TFLOPS standalone and
931.56 us / 2360.58 TFLOPS clustered. ATT shows that the slowdown is a cycle
regression rather than a frequency regression:

- Traced GFX-clock span: 1,113,888 to 1,891,064 cycles (+69.8%)
- Mean GFX clock: 1613.77 to 2022.10 MHz (+25.3%)
- Total attributed stall: 1,775,556 to 3,152,843 cycles (+77.6%)
- Total attributed idle: 71,353 to 191,429 cycles (+168.3%)

Useful dynamic work is unchanged: both traces execute 131,072 WMMA,
98,304 `ds_load_b128`, and 2,048 `tensor_load_to_lds` instructions. The
dominant added stalls are:

- `s_wait_tensorcnt`: 16,198 to 627,091 cycles (+610,893). The clustered
  loop's `tdm.async_wait(0)` after four paired TDM copies contributes 609,465
  cycles, or 1195 cycles per observed loop hit.
- Local `s_barrier_wait 0xffff`: 688,405 to 1,408,334 cycles (+719,929).
  The hottest local stage boundary contributes 839,992 cycles, or 1647 cycles
  per loop hit, while waves wait for the drained refill phase.
- Hardware cluster `s_barrier_wait 0xfffd`: 50,236 cycles total, or about
  97 cycles per loop hit. This is only 3.6% of the net stall increase.

The TDM drain and the amplified local stage wait account for 96.6% of the net
increase in ATT-attributed stall. TDM-issue idle also rises from 13,827 to
136,499 cycles. WMMA stall decreases by 52,131 cycles, so the matrix pipeline
is not the regression source. This motivated splitting the paired wait.

### Split-wait cluster ATT

The optimized cluster trace is retained at:

```text
/home/jungpark/mnt/att-traces/bf16-kernelc-bk128-cluster-4x4-split-wait-20260825
```

Its profiled replay measured 807.80 us and 2722.23 TFLOPS. Relative to the
full-drain cluster trace:

- GFX-clock span decreases from 1,891,064 to 1,550,512 cycles (-18.0%).
- Total attributed stall decreases from 3,152,843 to 2,436,757 cycles
  (-22.7%).
- `s_wait_tensorcnt 0` falls from 627,091 to 12,177 stalled cycles. The new
  `s_wait_tensorcnt 2` accounts for 247,575 cycles.
- Local barrier wait falls from 1,408,334 to 964,673 stalled cycles.
- The extra slot-1 visibility rendezvous raises hardware cluster-wait stall
  from 50,236 to 143,893 cycles.

The split therefore removes the full-drain hotspot and more than pays for the
additional cluster rendezvous. The remaining cluster gap is primarily the
partial TDM wait plus local and cluster synchronization.

### Same-schedule cluster ATT

The preferred cluster trace is retained at:

```text
/home/jungpark/mnt/att-traces/bf16-kernelc-bk128-cluster-4x4-same-schedule-20260825
```

Its profiled replay measured 698.71 us and 3147.26 TFLOPS. Relative to the
standalone ATT:

- GFX-clock span increases only from 1,113,888 to 1,171,520 cycles (+5.2%).
- Mean GFX clock increases from 1613.77 to 1744.37 MHz (+8.1%).
- Total attributed stall increases from 1,775,556 to 1,871,686 cycles (+5.4%).
- Hardware cluster wait contributes 36,345 stalled cycles.
- Local barrier wait decreases from 688,405 to 580,405 stalled cycles.
- `s_wait_tensorcnt` decreases from 16,198 to 9,070 stalled cycles.

The same-schedule cluster executes the same WMMA, LDS-load, and TDM-load
counts as standalone. Its small cycle overhead is offset by its higher
operating frequency, which explains why clustering itself does not reduce
measured throughput.
