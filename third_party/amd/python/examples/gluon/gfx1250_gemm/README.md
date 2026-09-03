# Canonical GFX1250 Gluon GEMMs

This package is the maintained home of the performance-retained GFX1250 GEMM
kernels. It centralizes code that previously existed only in the dated
`experiments/best_gfx1250_gluon_gemms_20260828.py` snapshot while leaving that
snapshot and the older example entry points intact for comparison.

The promoted device implementation was not structurally rewritten. That is
intentional: Gluon warp-pipeline lowering is sensitive to source organization
and instruction ordering. Cleanup beyond comments and host documentation must
be treated as a code-generation change.

## Package structure

- `kernels.py`: device kernels, layout builders, input preparation, correctness
  references, graph timing, and CLI implementation.
- `bench_gfx1250_gemms.py`: thin executable wrapper; supports direct and
  `python -m` invocation without duplicating behavior.
- `__init__.py`: public exports for the four retained device kernels.

Dependencies are one-way: experiment reproducers may import this package, but
canonical code must never import from `experiments`.

## Kernel selectors and contracts

| Selector | Operands | Scale format | Aggregate tile | Default pipeline |
| --- | --- | --- | --- | --- |
| `bf16` | BF16 x BF16 | None | 1024x1024x128, 4x4 CTAs | Two LDS slots; refill-side K64 split into two K32 WMMAs |
| `mxfp8` | E4M3 x E4M3 | E8M0, block 32 | 1024x1024x256, 4x4 CTAs | Leading-four TDM; warp-pipelined drain; split-N epilogue |
| `fp8_mxfp4` | E4M3 x packed E2M1 | B scale only, E8M0 block 32 | 1024x1024x256, 4x4 CTAs | Three-slot ring, single-slot prologue, pipelined drain |
| `mxfp4` | Packed E2M1 x packed E2M1 | E8M0, block 32 | 1024x1024x512, 4x4 CTAs | Two K256 groups with cluster overlap |

All retained launch paths use eight warps, FP32 accumulation, and BF16 output.
The BF16 selector can optionally write FP32 with `--bf16-fp32-output`.

### Terminology

- **CTA cluster**: cooperating thread blocks that share a distributed tile.
- **CGA layout**: maps logical tensor dimensions across CTAs in the cluster.
- **TDM**: descriptor-driven asynchronous global/LDS tensor transfer.
- **MXFP**: low-precision values paired with one E8M0 scale per K block.
- **Leading-four TDM**: warps 0-3 issue data/scale descriptor operations while
  the remaining warps focus on compute.

## Input and storage conventions

- Matrices are logically `A[M,K]`, `B[K,N]`, and `C[M,N]`.
- Kernels receive B as a contiguous `[N,K]` transpose.
- MXFP4 stores two E2M1 values per byte and is packed along K.
- E8M0 scales are preshuffled in groups of 128 non-K values before transfer.
- `--input-mode trig` uses `sin(i)` for A and `cos(i)` for B.
- `--input-mode random` uses deterministic seed 42 unless `--seed` overrides it.
- Without `--input-mode`, BF16/MXFP8 default to trig while
  FP8xMXFP4/MXFP4 default to random. Specify the option for comparisons.

## Running the code

Use the compiler checkout required by the GFX1250 experiments:

```bash
source /home/jungpark/.rock/bin/activate
export PYTHONPATH=/home/jungpark/mnt/wpindex/triton/python
```

Small correctness check:

```bash
python3 -m third_party.amd.python.examples.gluon.gfx1250_gemm.bench_gfx1250_gemms \
  --kernel mxfp8 -M 1024 -N 1024 -K 1024 \
  --input-mode random --check
```

Target benchmark:

```bash
/usr/local/bin/gpu-lock python3 -m \
  third_party.amd.python.examples.gluon.gfx1250_gemm.bench_gfx1250_gemms \
  --kernel all -M 4096 -N 4096 -K 65536 \
  --input-mode trig --benchmark --warmup 10 --probe-iters 20 \
  --iters-per-graph 50 --replays 20
```

The timed region contains only GEMM launches. Tensor creation, packing, scale
preshuffling, reference computation, and graph construction are excluded.
Nominal throughput is `2*M*N*K / elapsed`.

`gpu-lock` prevents concurrent users but does not pin clocks, reset thermal
state, or disable DVFS. Compare medians from interleaved runs and record input
mode; isolated results can differ by several percent.

## Verification status

Verified on September 3, 2026 with
`PYTHONPATH=/home/jungpark/mnt/wpindex/triton/python`:

- All four centralized kernels passed 1024x1024x1024 random-input correctness.
- Before the MXFP8 optimization below, forced compilation produced the same
  executable AMDGCN instruction sequence and the same VGPR, SGPR, and scratch
  allocation as the original snapshot. Only debug source paths differed.
- Three interleaved 4096x4096x65536 random-input benchmark pairs:
  - BF16: snapshot 3.161 PFLOPS, centralized 3.160 PFLOPS (-0.029%).
  - MXFP8: snapshot 8.209 PFLOPS, centralized 8.233 PFLOPS (+0.289%).
  - FP8 x MXFP4: snapshot 9.865 PFLOPS, centralized 9.888 PFLOPS (+0.234%).
  - MXFP4: snapshot 14.616 PFLOPS, centralized 14.552 PFLOPS (-0.442%).
- Three interleaved explicit-trig pairs differed by no more than 0.41% between
  the snapshot and centralized entry points.

### MXFP8 split-epilogue optimization

A later sweep applied the newer FP8xMXFP4 prologue/epilogue ideas to MXFP8:
single-slot prologue, warp-pipelined drain, and two half-width output stores.
The single-slot prologue was neutral. The split-N epilogue was the measured
performance win. A later interleaved tail recheck was effectively tied:
262.92 us pipelined versus 262.86 us plain for six random-input pairs, while
six warmed trig-input pairs measured 273.69 us pipelined versus 274.00 us
plain. The warp-pipelined drain remains enabled by default so the tail follows
the same explicit stage0/stage1 execution model as steady-state compute.

The split store requires a 128-column accumulator partition and a final cluster
rendezvous before output staging reuses operand LDS. That layout change reduced
VGPR allocation from 504 to 470 and private scratch from 68 bytes/thread to
zero. Enabling the warp-pipelined tail further reduces the current canonical
allocation to 468 VGPRs; dynamic shared memory remains 286,968 bytes.

At 4096x4096x65536:

- Six interleaved random-input pairs: 266.31 us / 8.257 PFLOPS baseline versus
  261.77 us / 8.401 PFLOPS split epilogue, a 1.74% speedup.
- Four interleaved trig-input pairs: 274.63 us / 8.007 PFLOPS baseline versus
  269.89 us / 8.148 PFLOPS split epilogue, a 1.75% speedup.
- All variants passed a 1024x1024x2048 random reference check and matched the
  canonical result at 4096x4096x65536.

The reproducible sweep is
`experiments/mxfp8_prologue_epilogue_20260903.py`.
Without `--variants`, that harness now runs only the canonical winning
configuration; alternative schedules require explicit selection.

The same harness also tested moving the cluster wait around the second K128
WMMA. E4M3 scaled WMMA has a fixed K128 hardware instruction, so it cannot be
split into K64 pieces like BF16. The two legal alternatives were
`wait -> WMMA -> refill` and `WMMA -> wait -> refill`; neither changed the
then-current 470-VGPR, zero-scratch allocation. With the default pipelined tail,
all three wait orders use 468 VGPRs. The first alternative was 0.12% slower
over six random-input pairs (264.71 versus 264.39 us) and 0.13% slower over
four trig-input pairs (273.13 versus 272.78 us). The later wait was slower in
the initial sweep, so the canonical `wait -> refill -> WMMA` order remains.

## Rules for future refactoring

1. Preserve each kernel's `warp_pipeline_stage`, barrier, refill, and
   `tdm.async_wait` order unless the change is explicitly an optimization.
2. Run correctness before benchmarking, including a long-K MXFP8 check.
3. Force compilation and compare executable AMDGCN plus VGPR/SGPR/scratch use.
4. Benchmark under `gpu-lock` with explicit input mode and interleaved samples.
5. Do not accept unexplained correctness, spill, occupancy, or throughput
   regressions.
6. Keep the old snapshot until each subsequent module split passes these gates.
