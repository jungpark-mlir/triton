# MXFP GEMM LDS Partition Experiment Guide

## Quick Start

Base command (non-WS warp_pipeline, 8-warp pingpong):
```bash
python mxfp_gemm_gfx1250.py \
  --num_warps 8 --scale_preshuffled --with_a_scale \
  --dtype_a float8_e4m3 --dtype_b float8_e4m3 \
  --num_buffers 4 --schedule baseline \
  --pingpong \
  -M 1024 -N 1024 -K 8192
```

Tile size is MNK = 256x256x128 (defaults: `--BM 256 --BN 256 --BK 128`).

## Experiment Configurations

### 1. Baseline (no partitioning)
```bash
python mxfp_gemm_gfx1250.py \
  --num_warps 8 --scale_preshuffled --with_a_scale \
  --dtype_a float8_e4m3 --dtype_b float8_e4m3 \
  --num_buffers 4 --schedule baseline \
  --pingpong \
  -M 1024 -N 1024 -K 8192
```

### 2. Partition A only
```bash
python mxfp_gemm_gfx1250.py \
  --num_warps 8 --scale_preshuffled --with_a_scale \
  --dtype_a float8_e4m3 --dtype_b float8_e4m3 \
  --num_buffers 4 --schedule baseline \
  --pingpong --ws_partition_a \
  -M 1024 -N 1024 -K 8192
```

### 3. Partition B only
```bash
python mxfp_gemm_gfx1250.py \
  --num_warps 8 --scale_preshuffled --with_a_scale \
  --dtype_a float8_e4m3 --dtype_b float8_e4m3 \
  --num_buffers 4 --schedule baseline \
  --pingpong --ws_partition_b \
  -M 1024 -N 1024 -K 8192
```

### 4. Partition A+B
```bash
python mxfp_gemm_gfx1250.py \
  --num_warps 8 --scale_preshuffled --with_a_scale \
  --dtype_a float8_e4m3 --dtype_b float8_e4m3 \
  --num_buffers 4 --schedule baseline \
  --pingpong --ws_partition_a --ws_partition_b \
  -M 1024 -N 1024 -K 8192
```

---

## What Each Option Does

### Core Options

| Flag | Description |
|------|-------------|
| `--num_warps 8` | Launch 8 hardware warps (waves). Required for pingpong. |
| `--pingpong` | Split the 8 warps into 2 groups of 4, alternating M-halves. Each group computes half the output tile. Requires `--num_warps 8`. |
| `--schedule baseline` | Use the baseline software-pipelined schedule. |
| `--num_buffers 4` | Number of LDS multi-buffer stages for pipelining global loads and compute. |
| `--scale_preshuffled` | Pre-shuffle scale factors in memory to match WMMA's expected data layout, avoiding runtime permutation. Sets `tiles_per_warp = 2`. |
| `--with_a_scale` | Include A-side scale factors (both A and B scales are loaded and applied). |
| `--dtype_a / --dtype_b` | Data types for operands. `float8_e4m3` = FP8 E4M3. Also supports `e2m1` (MXFP4). |

### LDS Partition Options

| Flag | Description |
|------|-------------|
| `--ws_partition_a` | Use `PartitionedSharedLayout` for operand A in LDS. Splits A data across 2 LDS partitions (64KB segments) so that SIMD pair 0 and pair 1 read from different segments. |
| `--ws_partition_b` | Use `PartitionedSharedLayout` for operand B in LDS. Same concept applied to B. |

### Debug Options

| Flag | Description |
|------|-------------|
| `--debug_deterministic` | Force deterministic execution for debugging. |
| `--dump_bottom` | Dump the bottom (drain) phase of the pipeline. |

---

## Background: LDS Architecture and Conflicts

### LDS Structure (MI450)
- 384 KB total LDS, divided into **6 segments** of 64 KB each.
- Each segment has **64 banks** (32-bit each).
- Two independent read ports: **Port A** (SIMD pair 0: SIMD0 + SIMD2) and **Port B** (SIMD pair 1: SIMD1 + SIMD3).

### Address Mapping
```
ADDR[18:0] = { Segment[2:0], SRAM_address[7:0], Bank[5:0], ByteInBank[1:0] }
```
- Bits [18:16] select the segment (partition).
- Bits [7:2] select the bank within a segment.

### Warp-to-SIMD Assignment
With 8 warps dispatched:
- **Port A (SIMD pair 0)**: warps 0, 2, 4, 6 (even warps)
- **Port B (SIMD pair 1)**: warps 1, 3, 5, 7 (odd warps)

### Two Types of LDS Conflicts

**Partition conflict** (`CU_CACHE_LDS_PARTITION_READ_CONFLICTS`): Occurs when Port A and Port B try to read from the **same 64KB segment** in the same cycle. This halves effective LDS bandwidth for that cycle.

**Bank conflict** (`DS_READ_BANK_CONFLICTS`): Occurs when two threads within the same wavefront access different addresses that map to the **same bank** within a segment. This serializes the accesses.

### How Partitioning Helps
`PartitionedSharedLayout` physically distributes tensor data across LDS segments so that:
- Even warps (Port A) always read from segment set {0, 2, 4}
- Odd warps (Port B) always read from segment set {1, 3, 5}

This eliminates partition conflicts for that operand, since the two ports never hit the same segment.

The partition index is derived from `warp_id & 1`:
- Partition 0 → even warps (Port A)
- Partition 1 → odd warps (Port B)

---

## Layout Details

### Default Layout (no partitioning)
Both A and B use `PaddedSharedLayout` — a contiguous block in LDS with padding to avoid bank conflicts:
```
A: PaddedSharedLayout([BK_PACKED, 16], [BLOCK_M, BK_PACKED], [1, 0])
B: PaddedSharedLayout([BK_PACKED, 16], [BLOCK_N, BK_PACKED], [1, 0])   # transposed
```

### Partitioned Layout
Wraps the padded layout with `PartitionedSharedLayout(num_partitions=2, num_groups=G, partition_dim=D, partition_layout=inner)`:

**For A** (`partition_dim=0`, along M):
- `num_groups_a = BLOCK_M / (tiles_per_warp * 16 * 2)`
  - With `scale_preshuffled` (tiles_per_warp=2): `256 / (2*16*2) = 4`
- The M dimension is sliced into `num_partitions * num_groups = 8` pieces.
- Pieces are interleaved: even pieces → partition 0 (Port A), odd pieces → partition 1 (Port B).

**For B** (`partition_dim=0`, along N, since B is transposed):
- `num_groups_b = BLOCK_N / (tiles_per_warp * 16 * 2)`
  - With `scale_preshuffled`: `256 / (2*16*2) = 4`
- The N dimension is sliced into 8 pieces, interleaved across 2 partitions.

### WMMA Layout (warp_bases)

The WMMA warp layout changes when partition-A is enabled to align warp-to-tile mapping with the partition structure:

**Default (8-warp, no partition)**:
```
warp_bases = [[0, tpw], [0, tpw*2], [tpw, 0]]
  → b0,b1 move along N, b2 moves along M
  → All warps share the same M rows for A reads
```

**With partition-A (8-warp)**:
```
warp_bases = [[tpw, 0], [0, tpw], [0, tpw*2]]
  → b0 (warp LSB) moves along M, b1,b2 move along N
  → Even warps (Port A) read from M-partition 0
  → Odd warps (Port B) read from M-partition 1
```

This ensures each SIMD pair accesses a different LDS segment for operand A.

## Perf Counters to Watch

In the simulator output (`perf_counters_miperf_absolute.txt` and `perf_counters_miperf.csv`):

| Counter | What it measures |
|---------|-----------------|
| `XDL_UTIL_E2E` | End-to-end XDL (WMMA) utilization, higher is better |
| `CU_CACHE_LDS_PARTITION_READ_CONFLICTS` | Total partition conflicts (two ports hitting same segment) |
| `DS_READ_BANK_CONFLICTS_SUM` | Total bank conflicts (threads hitting same bank) |
| `wave_waitcnt_idle_clocks` | Cycles waves spend waiting on `s_wait_dscnt` (LDS latency) |
