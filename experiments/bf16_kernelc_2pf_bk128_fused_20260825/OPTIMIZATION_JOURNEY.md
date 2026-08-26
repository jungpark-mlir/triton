# Improving BF16 Warp-Pipeline GEMM on GFX1250

## Scope

This document records the optimization journey for the BF16 warp-pipeline
KernelC experiment on AMD GFX1250. It covers the kernel changes, performance
experiments, Advanced Thread Trace (ATT) findings, failed approaches, and
Triton compiler fixes that led to the current best result.

The final isolated implementation is:

```text
experiments/bf16_kernelc_2pf_bk128_fused_20260825/
  bf16_kernelc_2pf_bk128_fused.py
  README.md
  OPTIMIZATION_JOURNEY.md
```

The compiler used for the final measurements is:

```text
/home/jungpark/mnt/wpindex/triton
```

The target benchmark throughout the investigation was:

- GPU architecture: GFX1250
- GEMM shape: `M=4096`, `N=4096`, `K=65536`
- Inputs: BF16
- Accumulation and output: FP32
- Input initialization for reported performance: trigonometric
- CTA-local output tile: `256x256`
- Final K tile: `BK=128`
- Waves per CTA: 8
- XCD mapping: 8 XCDs, `GROUP_SIZE_M=4`

Performance numbers came from repeated graph-replay benchmarks. ATT was used
to separate cycle effects from frequency effects and to classify WMMA issue
gaps, LDS stalls, tensor waits, local CTA barriers, and cluster barriers.

## Final result

The current best kernel uses:

- A 4x4 CTA cluster, forming a logical `1024x1024x128` cluster tile.
- A `256x256` output tile per CTA.
- Two BK128 LDS buffers.
- Two BK64 load/WMMA stages per BK128 tile.
- Two-piece CTA-local partitioned layouts for both A and B.
- Four leading TDM producer waves.
- Fused A/B TDM refill.
- Cluster arrival in the final LDS stage.
- The final WMMA between cluster arrive and cluster wait.
- Same-slot refill after cluster wait.
- Full mask-zero scheduler barriers around cluster synchronization in the
  AMDGPU lowering.

Three final benchmark runs measured:

- 680.80 us, 3230.065 TFLOPS
- 682.70 us, 3221.090 TFLOPS
- 680.35 us, 3232.179 TFLOPS

The median is:

- Latency: **680.80 us**
- Throughput: **3.230 PFLOPS**

The matching ATT capture measured:

- Mean active-CU cycles: **1,129,431**
- Mean GFX frequency: **1696.52 MHz**
- Dynamic WMMA instructions: **131,072**
- Merged WMMA issue span: **1,113,028 cycles**
- Ideal WMMA issue cycles: **1,048,568**
- Merged WMMA efficiency: **94.21%**
- WMMA gaps greater than 32 cycles: **1,020**
- Gaps in the 65-128 cycle range: **2**

The final successful ATT directory is:

```text
/home/jungpark/mnt/att-traces/bf16-kernelc-bk128-cluster-partitioned-4x4-arrive-in-load-stage-sched0-retry-20260825
```

Five full-target comparisons against the standalone kernel and the smaller
random-input checks passed.

## Starting point and architectural constraints

The original KernelC work established several constraints that shaped every
later experiment:

- Eight waves were retained.
- The kernel had to continue using `gl.amd.warp_pipeline_stage`.
- TDM supplied A and B through LDS.
- GFX1250 BF16 WMMA used the exact instruction
  `v_wmma_f32_16x16x32_bf16`.
- Register spills were unacceptable.
- The target shape was large enough that steady-state scheduling dominated
  prologue and epilogue cost.

The early three-buffer BK64 KernelC demonstrated that the basic architecture
could exceed 3 PFLOPS, but it did not provide the best balance of LDS capacity,
refill cadence, and stage overhead for clustering. The later work therefore
focused on a two-buffer BK128 kernel.

## Phase 1: moving to two-buffer BK128

The retained geometry became:

- `BM=256`
- `BN=256`
- `BK=128`
- Two LDS slots
- Eight waves

Each BK128 tile is consumed in two BK64 halves:

```text
load BK64 half 0 -> WMMA half 0
load BK64 half 1 -> WMMA half 1 + refill
```

This keeps the register footprint below the spill threshold while reducing
the number of TDM refills relative to BK64.

Three BK128 slots were tested but do not fit. Compilation required 417,760
bytes of shared memory, exceeding the 327,680-byte hardware limit. The
two-slot clustered schedule uses approximately 278,496 bytes.

## Phase 2: fusing refill into the compute pipeline

An early BK128 version used a separate TDM refill stage. ATT exposed 510
periodic intervals in which neither SIMD issued WMMA.

The refill was moved into the second-half WMMA stage:

```text
stage0: load second BK64 half from LDS
stage1: issue next A/B TDM loads and execute WMMA
```

This reduced the periodic no-WMMA intervals to five edge or outlier gaps.

Performance progression:

- Pre-fusion BK128 median: **728.33 us**
- Integrated fused BK128 runs: 705.53, 703.96, and 701.66 us
- Integrated fused BK128 median: **703.96 us**
- Isolated reproduction: **705.82 us / 3115.58 TFLOPS**

The integrated fused trace measured:

- 1,110,565 mean CU cycles
- 1629.90 MHz mean frequency
- Five WMMA issue gaps of at least 129 cycles
- No scratch load or store instructions

The first major finding was therefore that the refill had to be part of the
compute cadence. Treating it as a standalone producer stage created a
structural WMMA hole.

## Phase 3: introducing a 4x4 CTA cluster

The 4x4 cluster distributes a logical `1024x1024` tile over 16 CTAs while
retaining one `256x256` output tile per CTA.

The first clustered schedule was correct but expensive:

- Standalone reference: 705.67 us, 3116.21 TFLOPS
- Original clustered full-drain schedule: 925.47 us, 2376.11 TFLOPS

ATT showed that clustering itself was not the primary cost. The regression
came from draining four TDM requests and then waiting at a local stage
boundary:

- GFX span: 1,113,888 to 1,891,064 cycles
- Total attributed stall: 1,775,556 to 3,152,843 cycles
- `s_wait_tensorcnt`: 16,198 to 627,091 stalled cycles
- Local barrier wait: 688,405 to 1,408,334 stalled cycles
- Cluster barrier wait: only 50,236 cycles

The full TDM drain and amplified local stage wait accounted for 96.6% of the
net ATT-attributed stall increase. The hardware cluster wait itself was a
minor contributor.

### Split-wait schedule

The first correction split the paired refill wait:

- Full-drain cluster: 925.47 us
- Split-wait cluster: 813.56 us

The split-wait ATT result measured:

- 807.80 us / 2722.23 TFLOPS
- GFX span reduced from 1,891,064 to 1,550,512 cycles
- Total attributed stall reduced from 3,152,843 to 2,436,757 cycles
- `s_wait_tensorcnt 0` reduced from 627,091 to 12,177 stalled cycles
- Partial `s_wait_tensorcnt 2` contributed 247,575 stalled cycles
- Local barrier wait reduced to 964,673 cycles

This proved that TDM wait granularity, rather than cluster synchronization
itself, caused most of the initial cluster regression.

### Restoring the standalone refill cadence

The better solution was to use the standalone one-slot cadence inside the
cluster:

```text
consume slot 0 -> refill slot 0 -> wait(2)
consume slot 1 -> refill slot 1 -> wait(2)
```

Results:

- Same-schedule 4x4 cluster median: **700.75 us**
- Same-schedule throughput: **3138.10 TFLOPS**
- Matched standalone runs: 708.55, 704.70, and 706.89 us
- Matched cluster runs: 700.75, 700.97, and 700.00 us

ATT showed:

- 698.71 us / 3147.26 TFLOPS under profiling
- GFX span: 1,171,520 cycles
- Mean frequency: 1744.37 MHz
- Total attributed stall: 1,871,686 cycles
- Cluster wait stall: 36,345 cycles
- Local barrier stall: 580,405 cycles
- Tensor-wait stall: 9,070 cycles

The same useful work was executed: 131,072 WMMA, 98,304 `ds_load_b128`, and
2,048 `tensor_load_to_lds` instructions. The cluster had only a small cycle
overhead, which was offset by a higher operating frequency.

## Phase 4: partitioning LDS layouts

ATT showed that LDS bank conflicts remained important. Partition-aware layouts
were introduced for both operands.

The final composition is deliberately CTA-local:

1. Build the local padded shared layout.
2. Add partition and group bits inside the per-CTA tile.
3. Compose the CGA block mapping outside the CTA-local partition layout.

This prevents cluster block bits from interleaving partition pieces across
the logical cluster tile.

### Cluster-shape sweep

The original four-piece partitioned layouts measured:

- 4x2: 700.11, 699.41, and 697.23 us; **699.41 us median**
- 2x4: 708.76, 706.86, and 707.84 us; **707.84 us median**
- 2x2: 728.78, 730.06, and 722.81 us; **728.78 us median**

The 4x2 shape remained best among those configurations. ATT showed that it
reduced `ds_load` stall by 35.4%, from 238,498 to 154,164 cycles, but local
barrier and WMMA-attributed stalls rose enough to remove most of the
end-to-end gain.

The lesson was that reducing an instruction's attributed stall does not
necessarily reduce the critical path. Saved LDS cycles can reappear as
waiting at the next synchronization point.

### Two-piece 4x4 partitioning

Making partition/group composition CTA-local enabled 4x4 partitioning without
creating eight or sixteen TDM pieces.

Measured configurations:

- Four-piece 4x4: 698.65 us median
- Two-piece A and four-piece B: 698.29 us median
- Four-piece A and two-piece B: 695.79 us median
- Symmetric two-piece A/B: **695.25 us median**

The symmetric two-piece layout reduced `ds_load` stall from 238,498 to 121,718
cycles. It avoided the WMMA dependency penalty seen with four pieces and
became the partitioned default.

## Phase 5: identifying structural WMMA bubbles

The two-piece 4x4 ATT trace initially measured:

- WMMA efficiency: 90.50%
- 1,021 periodic long bubbles

The bubbles separated into two repeating classes:

- 510 gaps with a median of 127 cycles. One wave slot was in cluster/TDM
  transition work while the other executed approximately 48 LDS loads and a
  local stage barrier.
- 509 gaps with a median of 61 cycles. Tensor/local waits overlapped the
  remaining LDS-load sequence.

These transitions accounted for approximately 86% of lost WMMA issue cycles.
The bottleneck was no longer a single slow instruction. It was the
load-to-compute stage transition.

## Phase 6: slicing BK32

BK128 was sliced into four independently staged BK32 load/WMMA pairs to shorten
each LDS-load train.

The local result looked promising:

- `ds_load` stall: 121,718 to 47,840 cycles
- WMMA-attributed stall: 777,328 to 717,140 cycles

But the additional stage boundaries doubled local barriers:

- Local barrier executions: 4,110 to 8,190
- Local barrier stall: 772,471 to 885,679 cycles
- WMMA efficiency: 90.50% to 88.66%

Performance:

- BK64 median: **693.95 us**
- BK32 median: **701.47 us**

BK32 demonstrated that shorter LDS trains help, but adding synchronization
boundaries costs more than the saved LDS latency. BK64 remained the correct
granularity.

## Phase 7: experiments that did not improve BK64

Several attempts preserved the BK64 stage count but did not beat the retained
schedule:

- Splitting A/B TDM producers across low and high wave groups:
  **695.21 us median**
- Fusing A and B into one TDM operation: **695.12 us median**
- Retained low-four-wave TDM: **694.46 us median**
- Removing `s_setprio`: neutral or slower
- Giving LDS higher priority: neutral or slower
- Raising WMMA priority: neutral or slower
- Moving WMMA between cluster arrive/wait before enforcing backend order:
  695.02 versus 694.78 us median across paired samples
- `GROUP_SIZE_M=1` or 2: no gain over 4
- Four-XCD mapping: approximately 4 us slower than eight-XCD mapping

### Register-pressure limit

More operand lookahead was not viable:

- Loading all BK128 operands at once used 510 VGPRs, spilled 720 values into
  1,732 bytes of private memory, and took approximately 10.6 ms.
- Preloading only the second A half spilled 16 values and took approximately
  1.16 ms.
- Preloading only the second B half spilled 168 values and took approximately
  4.36 ms.

This established a hard boundary: the remaining bubbles could not be removed
by holding substantially more LDS data in registers.

## Phase 8: discovering backend scheduler reordering

Source order was changed to:

```text
cluster arrive -> WMMA -> cluster wait -> TDM refill
```

ATT still showed:

```text
cluster signal -> cluster wait -> tensor load -> WMMA
```

The compilation pipeline was inspected at three levels:

- TTGIR preserved arrive, dot, wait, TDM.
- LLVM IR preserved barrier signal, WMMA intrinsics, barrier wait, tensor load.
- AMDGCN reordered wait and tensor load before WMMA.

The reordering happened in the AMDGPU machine scheduler. Warp-pipeline lowering
only emitted selective scheduling barriers after memory operations. WMMA is
pure compute and has no dependency on the cluster/TDM side-effect chain, so
the backend was free to sink it past cluster wait and refill.

This explained why source-level scheduling experiments did not behave as
expected in ATT.

## Triton compiler fixes

The compiler fixes were implemented and pushed on the `wp-index` branch in:

```text
/home/jungpark/mnt/wpindex/triton
```

### 1. CTA-local partition composition

Commit:

```text
d87b3b0ac [AMD] Fix clustered partitioned WMMA layouts
```

Changes:

- `partitionedSharedToLinearLayout` now removes the CGA block mapping, composes
  partition/group bits within the CTA-local shape, and then recombines the CGA
  layout.
- `MemDescSubsliceOpConversion` uses `pseudoinvert()` rather than `invert()`
  because multicast CGA layouts can be surjective but non-square.
- WMMA repetition layout is derived after removing the block dimension and
  clamping to `shapePerCTA`.
- Scale layouts derive CGA bases from the dot operand, including the required
  B-operand transpose.
- A clustered partition-layout unit test verifies that partition bits select
  CTA-local pieces and block bits select the outer CTA tile.

These changes enabled valid partitioned 4x4 layouts and prevented interleaved
ownership of clustered tiles.

### 2. Cluster operations remain inside warp-pipeline stages

Commit:

```text
358cf23c4 [AMD] Keep cluster synchronization inside pipeline stages
```

Before this fix, cluster arrive/wait could be interpreted as pre-existing
stage-boundary barriers. That was incorrect because cluster synchronization
does not replace the CTA-local barrier needed to order LDS hazards between
warp-pipeline stages.

The fix:

- Treats cluster arrive and wait as ordinary operations in their containing
  pipeline stage.
- Keeps the CTA-local stage boundary even when cluster synchronization is
  present.
- Improves validation and wrapping of existing CTA-local barriers.
- Adds warp-pipeline tests proving that cluster operations do not replace the
  local stage barrier.

### 3. Full scheduler barriers around cluster synchronization

Commit:

```text
d3a1c1d3e [AMD] Preserve cluster barrier scheduling order
```

The lowering now emits a full:

```text
llvm.amdgcn.sched.barrier(i32 0)
```

after cluster arrive and before cluster wait. In AMDGCN this appears as:

```text
; sched_barrier mask(0x00000000)
```

This prevents non-memory compute, including WMMA, from crossing the
arrive/wait interval during machine scheduling.

After the fix, assembly preserved:

```text
cluster signal -> WMMA -> cluster wait -> tensor load
```

instead of:

```text
cluster signal -> cluster wait -> tensor load -> WMMA
```

For the two-piece partitioned 4x4 kernel with arrive in the WMMA stage:

- Active-CU cycles: 1,182,899 to 1,157,037
- Mean frequency: 1719.02 to 1736.89 MHz
- WMMA issue span: 1,166,987 to 1,140,313 cycles
- WMMA efficiency: 89.85% to 91.95%
- Cluster-wait stall: 34,016 to 5,457 cycles

The scheduler fix converted the intended source overlap into real hardware
overlap.

## Phase 9: advancing cluster arrival

The final kernel moves cluster arrival into the final LDS-load stage:

```python
with gl.amd.warp_pipeline_stage("stage0", priority=0):
    a = a_buf.index(slot).slice(half_k, half_k, 1).load(layout=DOT_A)
    b = a_buf.index(slot).slice(half_k, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    gl.amd.gfx1250.cluster.arrive()

with gl.amd.warp_pipeline_stage("stage1", priority=1):
    acc = gl.amd.gfx1250.wmma(a, b, acc)
    gl.amd.gfx1250.cluster.wait()
    a_desc, b_desc = issue_loads(
        a_desc, b_desc, a_buf.index(slot), b_buf.index(slot), BLOCK_K)
```

This placement relies on the GFX1250 hardware ordering guarantee for the
cluster/TDM protocol; a `sched_barrier` orders scheduling but is not itself a
DS completion wait.

The final placement improved:

- Median latency: 688.77 to **680.80 us**
- Throughput: 3.193 to **3.230 PFLOPS**
- Active-CU cycles: 1,157,037 to **1,129,431**
- WMMA issue span: 1,140,313 to **1,113,028 cycles**
- WMMA efficiency: 91.95% to **94.21%**
- Long WMMA gaps: 1,530 to **1,020**
- 65-128-cycle gaps: 507 to **2**

This removed one class of long transition bubble. Most remaining gaps moved
into the 33-64-cycle range.

## How the bottleneck evolved

The optimization did not address one fixed bottleneck. Each successful change
exposed the next one:

1. **Separate refill-stage bubbles**
   - Symptom: 510 periodic no-WMMA intervals.
   - Fix: fuse TDM refill into the second-half compute stage.

2. **Full TDM drain in the first clustered schedule**
   - Symptom: 627,091 tensor-wait stall cycles and 1.4 million local-barrier
     stall cycles.
   - Fix: split the wait, then restore the standalone one-slot refill cadence.

3. **LDS bank conflicts**
   - Symptom: 238,498 `ds_load` stall cycles in the plain clustered layout.
   - Fix: CTA-local two-piece partitioned A/B layouts.

4. **Excessive stage granularity**
   - Symptom: BK32 reduced LDS stall but doubled local barriers.
   - Fix: retain BK64 slicing.

5. **Register pressure**
   - Symptom: lookahead variants spilled and slowed by factors of two to
     fifteen.
   - Fix: keep only one BK64 half of operands live at a time.

6. **Backend scheduler reordering**
   - Symptom: ATT disagreed with source and LLVM IR ordering.
   - Fix: mask-zero scheduling barriers around cluster arrive/wait.

7. **Cluster-arrival transition bubbles**
   - Symptom: 1,530 gaps over 32 cycles after scheduler ordering was fixed.
   - Fix: advance cluster arrival into the final LDS stage.

## Remaining bottleneck

The final kernel reaches 94.21% merged WMMA efficiency. Its remaining issue
loss is dominated by approximately 1,020 stage-transition gaps, almost all in
the 33-64-cycle range.

The important constraints are:

- Cluster wait is no longer the dominant stall.
- BK32 reduces LDS latency but adds too many local boundaries.
- Full-BK128 or asymmetric operand lookahead spills.
- TDM producer splitting and priority changes do not materially help.
- The useful WMMA, LDS-load, and TDM-load counts are already stable.

The remaining bottleneck is therefore short stage-transition skew among LDS
completion, CTA-local boundary synchronization, and tensor readiness. Further
improvement likely requires a compiler or hardware scheduling change that
reduces the cost of an existing boundary without adding another boundary or
increasing operand liveness.

Potential future directions:

- A narrower row/column cluster rendezvous so A and B multicast refills do not
  synchronize all 16 CTAs.
- Compiler scheduling that reduces arrival skew at the existing local stage
  boundary.
- A way to expose shorter LDS-ready chunks without creating additional
  warp-pipeline boundaries.
- Lower-register-pressure operand representations that permit limited
  lookahead without spills.

## Reproduction

Use the wpindex compiler:

```bash
source /home/jungpark/.rock/bin/activate
export PYTHONPATH=/home/jungpark/mnt/wpindex/triton/python
export TRITON_ALWAYS_COMPILE=1
```

Correctness:

```bash
/usr/local/bin/gpu-lock python3 \
  experiments/bf16_kernelc_2pf_bk128_fused_20260825/bf16_kernelc_2pf_bk128_fused.py \
  --cluster-same-schedule \
  --cluster-partitioned \
  -M 1024 -N 1024 -K 4096 \
  --input-mode random \
  --check
```

Target benchmark:

```bash
/usr/local/bin/gpu-lock python3 \
  experiments/bf16_kernelc_2pf_bk128_fused_20260825/bf16_kernelc_2pf_bk128_fused.py \
  --cluster-same-schedule \
  --cluster-partitioned \
  --input-mode trig \
  --benchmark
```

Final ATT:

```bash
DIR=/home/jungpark/mnt/att-traces/bf16-kernelc-bk128-cluster-partitioned-4x4-arrive-in-load-stage-sched0-retry-20260825

/usr/local/bin/gpu-lock rocprofv3 \
  --advanced-thread-trace \
  --att-target-cu 1 \
  --att-shader-engine-mask 0x1 \
  --kernel-include-regex 'bf16_kernelc_2pf_bk128_fused_cluster_4x4_gfx1250' \
  --kernel-iteration-range 100 \
  --output-directory "$DIR" \
  --output-file bf16_kernelc_bk128_cluster_partitioned_4x4_final_att \
  --output-format csv json \
  -- python3 \
    experiments/bf16_kernelc_2pf_bk128_fused_20260825/bf16_kernelc_2pf_bk128_fused.py \
    --cluster-same-schedule \
    --cluster-partitioned \
    --input-mode trig \
    --benchmark
```

## Summary

The largest gains came from changes that corrected pipeline structure rather
than from isolated instruction tuning:

- Fuse refill with useful compute.
- Avoid full TDM drains.
- Reuse the standalone one-slot cadence in the cluster.
- Compose partitioned layouts per CTA.
- Keep BK64 to balance LDS latency and barrier count.
- Prevent the backend from destroying source-level overlap.
- Advance cluster arrival to reduce transition bubbles.

The result progressed from a 728.33-us pre-fusion BK128 kernel and a
925.47-us first clustered implementation to the current **680.80-us,
3.230-PFLOPS** kernel with **94.21% WMMA efficiency**.
