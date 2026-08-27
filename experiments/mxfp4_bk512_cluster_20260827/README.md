# GFX1250 MXFP4 BK512 Clustered Warp Pipeline

## Target

- Problem: `M=4096, N=4096, K=65536`
- Inputs: packed E2M1 MXFP4 with E8M0 block-32 scales
- Output: BF16
- Per-CTA tile: `256x256x512`
- Eight warps per CTA
- Two LDS buffers
- Two physical LDS partitions for A and B
- Leading-four TDM refill schedule

The implementation is
`mxfp4_bk512_warp_pipeline_gfx1250` in
`third_party/amd/python/examples/gluon/warp_pipelined_gemm_gfx1250.py`.
All schedule variants retain `gl.amd.warp_pipeline_stage`.

## Cluster shapes

`--mxfp4-cluster-shape` accepts `2 2`, `2 4`, `4 2`, and `4 4`.
Each CTA retains the same 256x256x512 work; the aggregate M/N tile follows the
cluster dimensions.

Three-run graph-timing medians for the original K128-stage schedule:

- 4x4: 146.42 us, 15.018 PFLOPS
- 2x4: 147.44 us, 14.915 PFLOPS
- 4x2: 147.74 us, 14.884 PFLOPS
- 2x2: 150.42 us, 14.619 PFLOPS

All shapes passed correctness. The 4x4 cluster remains the best shape.

## Warp-pipeline improvements

The initial BK512 schedule used eight warp-pipeline regions:

```text
load K128 -> WMMA -> load K128 -> WMMA ->
load K128 -> WMMA -> load K128 -> WMMA
```

`--mxfp4-k256-stages` groups two K128 fragments per load/compute pair:

```text
load K256 -> 2 x WMMA -> load K256 -> 2 x WMMA
```

This reduces the tile to four warp-pipeline regions. It increases VGPR usage
from 390 to 490 but retains zero VGPR/SGPR spills and zero scratch.

`--mxfp4-overlap-cluster` moves cluster synchronization around the final
scaled WMMA:

```text
cluster.arrive() -> final scaled WMMA -> cluster.wait()
```

This allows the final register-only matrix operation to overlap the cluster
rendezvous before the consumed LDS slot is refilled.

Interleaved three-run medians:

- Original K128 stages: 146.17 us, 15.045 PFLOPS
- K256 stages: 142.14 us, 15.471 PFLOPS
- K256 stages plus cluster overlap: 139.60 us, 15.752 PFLOPS

The final schedule is 4.7% faster than the original clustered BK512 schedule.
It uses 494 VGPRs, 82 SGPRs, and has no spills or scratch accesses.

The matched hipBLASLt baseline is 134.99 us and 16.291 PFLOPS, leaving a 3.3%
throughput gap.

## Correctness

The final schedule passed:

- Random input at `1024x1024x1024`
- Random input at `1024x1024x2048`, exercising steady-state refills
- Trigonometric input at `1024x1024x1024`

## Final command

```bash
source /home/jungpark/.rock/bin/activate
export PYTHONPATH=/home/jungpark/mnt/wpindex/triton/python

/usr/local/bin/gpu-lock python3 \
  third_party/amd/python/examples/gluon/warp_pipelined_gemm_gfx1250.py \
  --mxfp4-bk512 \
  --mxfp4-cluster-shape 4 4 \
  --mxfp4-k256-stages \
  --mxfp4-overlap-cluster \
  --input-mode random \
  -M 4096 -N 4096 -K 65536 \
  -BM 1024 -BN 1024 -BK 512 \
  --num-buffers 2 --num-warps 8 \
  --benchmark --iters-per-graph 50
```

## Advanced thread trace

The original 4x4 K128-stage trace is retained at:

```text
/home/jungpark/mnt/att-traces/gluon-mxfp4-bk512-cluster-4x4-4096x4096x65536-20260827
```

It measured 223,134.9 mean CU cycles. Barrier waits represented 45.4% of
attributed stall, LDS dependency waits represented 18.0%, and direct LDS-load
stall represented only 1.8%. Exactly four long WMMA issue gaps appeared per
BK512 tile, motivating the K256 stage grouping and cluster-overlap changes.
