# Membar: Warp-Local Shared Memory Access

## Problem 2-1: Unnecessary Barriers Between Write/Read Op Pairs

Triton's membar analysis treats shared memory as a flat address space shared by
all threads in a CTA. When a **writer** (TDM copy, async_copy, or local_store)
and a **reader** (local_load) touch the same allocation, membar inserts a
CTA-wide barrier (`__syncthreads()` / `s_barrier`) even if the layout
guarantees that each warp only accesses its own partition. In such cases, the
barrier is unnecessary — there is no cross-warp data dependency.

**Target op pairs** for barrier suppression:

| Writer | Reader | Status |
|--------|--------|--------|
| `AsyncTDMCopyGlobalToLocalOp` | `local_load` | **Implemented** (commit [`df6d5be`](https://github.com/triton-lang/triton/commit/df6d5be2206ec6f32cf47116d23f3b6235873bfe)) |
| `AsyncCopyGlobalToLocalOp` | `local_load` | Extensible (same `warpsPerCTA` check) |
| `local_store` | `local_load` | Extensible (same `warpsPerCTA` check) |

```
  Shared Memory (one buffer slot)
  ┌─────────────────────────────────────────────────┐
  │  Warp 0 region      │  Warp 1 region            │
  │  [0x000, 0x100)     │  [0x100, 0x200)           │
  │                     │                           │
  │  W0 writes here     │  W1 writes here           │
  │  W0 reads here      │  W1 reads here            │
  └─────────────────────┴───────────────────────────┘
        ↑ no cross-warp overlap → barrier unnecessary
```

## Solution (2-1): warpsPerCTA Comparison

The check compares `warpsPerCTA` distributions from the writer and reader.
If they match, each warp owns the same tensor-element partition on both
sides, and since shared memory encodings never let two different elements
share the same address, the byte addresses are also disjoint.

### How warpsPerCTA Determines Warp Ownership

Triton's register-side encodings (`BlockedEncodingAttr`, `AMDMfmaEncodingAttr`,
etc.) include a `warpsPerCTA` vector that specifies how warps are distributed
across tensor dimensions. For a `[64, 64]` tile with 4 warps:

```
  warpsPerCTA = [4, 1]              warpsPerCTA = [2, 2]
  4 warps along rows, 1 along cols  2 warps along rows, 2 along cols

  ┌──────── 64 cols ────────┐       ┌──── 32 cols ──┬─── 32 cols ──┐
  │ W0: rows  0-15          │       │ W0: r 0-31    │ W1: r 0-31   │
  ├──────────────────────────┤       │    c 0-31     │    c 32-63   │
  │ W1: rows 16-31          │       ├───────────────┼──────────────┤
  ├──────────────────────────┤       │ W2: r 32-63  │ W3: r 32-63  │
  │ W2: rows 32-47          │       │    c 0-31     │    c 32-63   │
  ├──────────────────────────┤       └───────────────┴──────────────┘
  │ W3: rows 48-63          │       Each warp: 32×32 tile
  └──────────────────────────┘
  Each warp: 16×64 row block
```

Each warp owns a disjoint partition of tensor elements. The `warpsPerCTA`
vector fully determines which elements belong to which warp.

For TDM copies, there is no register encoding — the warp distribution is
computed from the tensor descriptor's block shape via
`tdmGetWarpDistribution(blockShape, numDims, numWarps)`. This function
distributes warps across dimensions in the same way as the register encoding.

### The Check: Step by Step

The implementation extracts `warpsPerCTA` from both the writer and reader:

```
  ┌───────────────────────────────────────────────────────────┐
  │ filterWarpLocalAccesses(op1, op2)                         │
  │                                                           │
  │  1. Is at least one op an AsyncTDMCopyGlobalToLocalOp?    │
  │     No → return false (don't filter non-TDM pairs)        │
  │                                                           │
  │  2. Extract warpsPerCTA for each op:                      │
  │     ┌─────────────────────────────────────────────────┐   │
  │     │ TDM op?                                         │   │
  │     │   Yes → getTDMWarpsPerCTA(tdmOp)                │   │
  │     │          calls tdmGetWarpDistribution            │   │
  │     │   No  → getRegWarpsPerCTA(op)                   │   │
  │     │          reads from distributed encoding        │   │
  │     └─────────────────────────────────────────────────┘   │
  │                                                           │
  │  3. Normalize: strip trailing 1s                          │
  │     [4, 1] → [4]     [4, 1, 1] → [4]     [2, 2] → [2, 2]│
  │                                                           │
  │  4. Compare: normalized vectors equal?                    │
  │     Yes → return true (suppress barrier)                  │
  │     No  → return false (keep barrier)                     │
  └───────────────────────────────────────────────────────────┘
```

Concrete examples:

```
  FA MQA decode (barrier suppressed):

  TDM write:   tdmGetWarpDistribution([64, 64], numWarps=4) → [4, 1]
  local_load:  blocked encoding warpsPerCTA = [4, 1]

  normalize([4, 1]) = [4]
  normalize([4, 1]) = [4]
  [4] == [4] → MATCH → barrier suppressed ✓
```

```
  Standard MMA with K-broadcast (barrier kept):

  TDM write:   tdmGetWarpDistribution([64, 64], numWarps=4) → [4, 1]
  local_load:  MMA operand warpsPerCTA = [2, 2]

  normalize([4, 1]) = [4]
  normalize([2, 2]) = [2, 2]
  [4] ≠ [2, 2] → MISMATCH → barrier kept ✓
```

```
  Rank change from reshape/trans (barrier suppressed):

  TDM writes a 2D tile:  warpsPerCTA = [4, 1]
  After reshape+trans, local_load sees a 3D view:  warpsPerCTA = [4, 1, 1]

  normalize([4, 1])    = [4]
  normalize([4, 1, 1]) = [4]
  [4] == [4] → MATCH → barrier suppressed ✓
```

### Why Matching warpsPerCTA Proves Disjointness

The check operates in **tensor space** — it only asks "which rows/columns
does each warp own?" It does not look at byte addresses at all. So why is
this safe? Because of a simple property of how shared memory works:

> **Every tensor element gets its own unique address in shared memory.
> No two different elements ever land on the same byte offset.**

This is a property called a **bijection** (one-to-one mapping), and it holds
for all Triton shared memory encodings — padded, swizzled, linear, and
rotating. The encoding may rearrange *where* each element goes (swizzling
shuffles addresses, padding adds gaps), but it never puts two elements at the
same place. Each element always occupies its own unique bytes.

This means: **if two warps own different elements, they automatically access
different addresses.** The encoding cannot break this — it can only rearrange
which addresses they use, not cause them to collide.

The full reasoning:

1. Same `warpsPerCTA` → each warp owns the **same set of tensor elements**
   on both the write and read sides.
2. Different warps own different elements (warp partitions don't overlap).
3. The shared encoding gives each element a unique address (no collisions).
4. Therefore, different warps access **different addresses**.
5. No cross-warp address overlap → **no CTA-wide barrier needed**.

```
  Tensor elements             Shared encoding              Byte addresses
  (warpsPerCTA = [4, 1])      (one-to-one mapping)

  ┌──────────────────┐                                ┌──────────────────┐
  │ W0: rows  0-15   │ ──── 1024 elements ──────────▶ │ W0: 1024 addrs   │
  ├──────────────────┤       each element gets         ├──────────────────┤
  │ W1: rows 16-31   │ ──── its own address ─────────▶ │ W1: 1024 addrs   │
  ├──────────────────┤       (never shared)            ├──────────────────┤
  │ W2: rows 32-47   │ ──── 1024 elements ──────────▶ │ W2: 1024 addrs   │
  ├──────────────────┤                                ├──────────────────┤
  │ W3: rows 48-63   │ ──── 1024 elements ──────────▶ │ W3: 1024 addrs   │
  └──────────────────┘                                └──────────────────┘
  Disjoint elements                                   Disjoint addresses
  (by warpsPerCTA)                                    (guaranteed by encoding)
```

The encoding determines the specific byte offsets — contiguous blocks for
non-padded, strided blocks for padded, XOR-scattered for swizzled — but
disjointness holds regardless because no two elements share an address:

```
  Same element partition, different encodings — all disjoint:

  Non-padded (stride = 128 bytes):    Padded (stride = 144 bytes):
  W0: bytes [0x0000, 0x0800)          W0: bytes [0x0000, 0x0900)
  W1: bytes [0x0800, 0x1000)          W1: bytes [0x0900, 0x1200)
  W2: bytes [0x1000, 0x1800)          W2: bytes [0x1200, 0x1B00)
  W3: bytes [0x1800, 0x2000)          W3: bytes [0x1B00, 0x2400)
       all disjoint ✓                      all disjoint ✓

  Swizzled (XOR-based):
  W0: 1024 addrs (XOR-scattered)      ← same elements, shuffled addresses
  W1: 1024 addrs (different set)      ← still unique, still disjoint
  ...                                      all disjoint ✓
```

### Padded Shared Layouts in Practice

This is worth highlighting because it was a critical factor in the design
decision. The primary motivating case — FA MQA decode — uses padded shared
layouts (`PaddedSharedEncodingAttr`, e.g., `padded_shared<[32:+4]>`).

Padded layouts add extra bytes per row to avoid bank conflicts. This changes
the row stride from a power-of-2 to a non-power-of-2 value (e.g., stride
= `(64 + 4) × 2 = 136` bytes instead of `64 × 2 = 128`).

```
  Padded layout: stride = (cols + padding) × elem_size

  W0: rows  0-15 → bytes [0,               16 × stride)
  W1: rows 16-31 → bytes [16 × stride,     32 × stride)
  W2: rows 32-47 → bytes [32 × stride,     48 × stride)
  W3: rows 48-63 → bytes [48 × stride,     64 × stride)

  Regardless of stride value, each warp's byte range is contiguous
  and non-overlapping. The stride determines how large each warp's
  range is — it never causes overlap.
```

The `warpsPerCTA` check does not inspect the stride at all. It operates
entirely in tensor space (which rows does each warp own?), and the one-to-one
address mapping guarantees the rest. This is why it handles padded layouts
natively — unlike the GF(2) approach which fails for non-power-of-2 strides
(see Appendix A).

### Why This Supersedes GF(2)

The original design proposed a GF(2) linear independence check on
`LinearLayout` basis vectors (see Appendix A). The `warpsPerCTA` comparison
is strictly better in practice:

1. **Handles padded layouts.** GF(2) uses XOR arithmetic, which breaks for
   non-power-of-2 strides. The FA MQA decode case uses padded layouts.
2. **No `LinearLayout` computation.** The check reads `warpsPerCTA` from
   encoding metadata — no composed layout needed.
3. **Same practical coverage.** Every case where warp-local access matters
   in practice has warps distributed along a single dimension. GF(2) could
   theoretically handle 2D warp distributions, but those are cross-warp
   (MMA K-broadcast) and don't benefit from this optimization.

## MMA Dot Operand Layouts

MMA operand layouts are **not always cross-warp**. The warp distribution for
dot operands depends on the MMA's `warpsPerCTA`:

```
  MMA output C tiled by warpsPerCTA = {2, 2}:

  ┌─────────┬─────────┐
  │  W0     │  W1     │        Operand A:         Operand B:
  │         │         │        ┌────┬────┐        ┌────┬────┐
  ├─────────┼─────────┤        │W0  │W0  │        │W0  │W1  │
  │  W2     │  W3     │        │W1  │W1  │        │W2  │W3  │
  │         │         │        ├────┼────┤        ├────┼────┤
  └─────────┴─────────┘        │W2  │W2  │        │W0  │W1  │
                               │W3  │W3  │        │W2  │W3  │
  C: warps tile M × N          └────┴────┘        └────┴────┘
                               A: partition M,     B: partition N,
                                  broadcast K         broadcast K
```

Operand A partitions warps along M (rows) and broadcasts along K — all warps
read the same K columns. Operand B partitions along N and broadcasts K.

The `warpsPerCTA` comparison detects this correctly:

| Scenario | Writer warpsPerCTA | Reader warpsPerCTA | Match? | Barrier? |
|----------|:--:|:--:|:--:|:--:|
| TDM `[4,1]` → operand A `[4,1]` | `[4,1]` | `[4,1]` | Yes | **No** |
| TDM `[4,1]` → operand A `[2,2]` | `[4,1]` | `[2,2]` | No | **Yes** |
| TDM `[4,1]` → operand B `[1,4]` | `[4,1]` | `[1,4]` | No | **Yes** |

When the reader has K-broadcast (warps along a different dimension than the
writer), `warpsPerCTA` vectors don't match, and the barrier is correctly kept.

## Batched MMA (Warps Across Batch Dimension)

A distinct pattern from standard MMA operands is **batched MMA**, where warps
are distributed across a batch dimension rather than M/N of a single MMA tile.
Each warp executes an independent MMA on its own data.

### Motivating Example: FA MQA Decode with Split-K

In flash attention MQA decode, batched WMMA is used for split-k where 4
warps are distributed across the batch dimension. TDM loads K and V with
a 2D shape `[4 * BLOCK_N, HEAD_SZ]`, which is then reshaped to
`[4, BLOCK_N, HEAD_SZ]` for WMMA. Each warp independently loads and
consumes its own K/V partition:

```
  TDM load: [4 * BLOCK_N, HEAD_SZ]      Batched WMMA view:
  ┌──────────────────────────┐           [4, BLOCK_N, HEAD_SZ]
  │ W0: rows [0, BN)         │   ──→    W0: MMA on its own BN×HS slice
  ├──────────────────────────┤
  │ W1: rows [BN, 2·BN)      │   ──→    W1: MMA on its own BN×HS slice
  ├──────────────────────────┤
  │ W2: rows [2·BN, 3·BN)    │   ──→    W2: MMA on its own BN×HS slice
  ├──────────────────────────┤
  │ W3: rows [3·BN, 4·BN)    │   ──→    W3: MMA on its own BN×HS slice
  └──────────────────────────┘
  No cross-warp data dependency → CTA barrier unnecessary
```

The pipeline uses triple buffering with `a = i % 3`, `b = (i + 1) % 3`,
`c = (i + 2) % 3`. Currently, membar inserts barriers between TDM writes
and the subsequent `local_load`:

```python
  self.async_wait(6)
  self.issue_global_load_v(i + 2, buf=c)    # TDM write → buffer c
  # <-- barrier inserted (false positive)
  v = self.shared_load_v(buf=a)             # local_load ← buffer a
```

Two independent problems contribute to this false positive:

1. **Buffer index disjointness** — TDM writes to buffer `c = (i+2)%3`
   while `local_load` reads from buffer `a = i%3`. These are different
   buffer slots, but membar cannot distinguish them because the indices
   are dynamic. `BufferIndexExpr` handles this (Problem 1 in the doc set).

2. **Same-buffer write→read from a previous iteration** — The TDM write
   that filled buffer `a` happened 2 iterations ago. Even if buffer index
   disjointness resolves the cross-buffer case, there is still a pending
   RAW dependency on buffer `a` itself. `async_wait` ensures the data is
   visible, but membar inserts a CTA-wide barrier because it cannot tell
   that the access is warp-local.

The warp-local check is the **stronger and more general** fix. If both
the TDM write and the WMMA `local_load` are warp-disjoint with matching
partitions, no cross-warp data dependency exists regardless of buffer
slot — eliminating barriers from both problems simultaneously.

### Why This Differs from Standard MMA Operands

In the standard MMA case, warps are distributed across M and N of a single
MMA tile, with K broadcast across warps. Operand A partitions M
(warp-disjoint along M) but broadcasts K (all warps read the same K data
from shared memory).

In batched MMA, warps are across the batch dimension. There is **no
K-dimension broadcast** — each warp has its own K data entirely. This
makes it a simpler case: the warp dimension selects which batch element
(which `BLOCK_N × HEAD_SZ` block in shared memory), and each warp's
threads address only within that block.

### Detection

The `warpsPerCTA` comparison detects this case directly:

- **TDM write side**: `tdmGetWarpDistribution` returns `[4, 1]` for 4
  warps distributed along rows.
- **Local_load (WMMA operand) side**: the distributed encoding's
  `getWarpsPerCTA` returns `[4, 1]` for batched WMMA with all warps along
  the batch/M dimension.

Both normalize to `[4]` → match → barrier suppressed.

## TDM Copy and Async Wait

### TDM Copy (AMD gfx1250)

TDM (Tensor Data Mover) copies produce a **warp-partitioned write
distribution**. The warp distribution is determined by
`tdmGetWarpDistribution`, which distributes warps across block dimensions:

```
  For a 64×64 block with 4 warps:

  tdmGetWarpDistribution([64, 64], 2, 4)  →  warpsPerCTA = [4, 1]

  TDM DMA targets:
  ┌──────────────────┐
  │ W0: rows  0-15   │ ← warp 0 DMA writes here
  ├──────────────────┤
  │ W1: rows 16-31   │ ← warp 1 DMA writes here
  ├──────────────────┤
  │ W2: rows 32-47   │ ← warp 2 DMA writes here
  ├──────────────────┤
  │ W3: rows 48-63   │ ← warp 3 DMA writes here
  └──────────────────┘
  Write side: always warp-disjoint
```

Whether the barrier can be eliminated depends on whether the **reader**
also has a matching warp distribution:

```
  TDM write → local_load (cross-warp reader):

  Write:                    Read (MMA operand, K-broadcast):
  warpsPerCTA = [4, 1]     warpsPerCTA = [2, 2]
  ┌──────────┐              ┌──────────┐
  │ W0 only  │              │ W0 + W1  │ ← reads from both
  ├──────────┤              │          │    warp 0 and warp 1
  │ W1 only  │              ├──────────┤    regions
  ├──────────┤              │ W2 + W3  │
  │ W2 only  │              │          │
  ├──────────┤              └──────────┘
  │ W3 only  │              [4,1] ≠ [2,2] → CTA barrier needed!
  └──────────┘

  TDM write → local_load (matching partition):

  Write:                    Read:
  warpsPerCTA = [4, 1]     warpsPerCTA = [4, 1]
  ┌──────────┐              ┌──────────┐
  │ W0 only  │              │ W0 only  │
  ├──────────┤              ├──────────┤
  │ W1 only  │              │ W1 only  │
  ├──────────┤              ├──────────┤
  │ W2 only  │              │ W2 only  │
  ├──────────┤              ├──────────┤
  │ W3 only  │              │ W3 only  │
  └──────────┘              └──────────┘
  [4,1] == [4,1] → No CTA barrier needed
```

### `async_wait` Semantics

`async_wait` ensures outstanding DMA operations have completed and their
results are visible in shared memory:

- **AMD**: CTA-wide memory visibility (all warps see the data). Not an
  execution barrier.
- **NVIDIA**: Per-thread `cp.async.wait_group`. Cross-warp visibility
  requires a subsequent `__syncthreads()`.

`async_wait` is **not a conceptual obstacle** to the warp-local
optimization — each warp reads only what it wrote:

```
  Timeline (warp-disjoint case):

  Warp 0:  [TDM write → region A] ... [async_wait] ... [local_load ← region A]
  Warp 1:  [TDM write → region B] ... [async_wait] ... [local_load ← region B]

  Each warp reads only what it wrote → no cross-warp dependency
  → async_wait is sufficient, no CTA barrier needed
```

However, there is a **practical obstacle**: the membar analysis
unconditionally inserts a CTA barrier after every `MemWaitOpTrait` op
(including `async_wait`), bypassing the `MembarFilterFn` entirely. See
[`MemWaitOpTrait`: Unconditional Barrier Problem](#memwaitoptrait-unconditional-barrier-problem)
for details and a proposed solution.

### Interaction Summary

| Operation | Write warpsPerCTA | Read warpsPerCTA | Match? | Barrier? |
|-----------|:-:|:-:|:-:|:-:|
| TDM → local_load (batched MMA) | `[4,1]` | `[4,1]` | Yes | **No** |
| TDM → local_load (standard MMA, K-broadcast) | `[4,1]` | `[2,2]` | No | **Yes** |
| async_copy → local_load (matching) | from src encoding | from dst encoding | If match | **No** |
| async_copy → local_load (mismatched) | from src encoding | from dst encoding | If mismatch | **Yes** |
| local_store → local_load | Encoding | Encoding | If match | Extensible |

## Integration with Membar

**Implemented** as a `MembarFilterFn` clause in the AMD backend.

The warp-local check is a `filterWarpLocalAccesses` function in
`MembarUtility.cpp`, added to the AMD `membarFilter`:

```cpp
bool membarFilter(Operation *op1, Operation *op2, ...) {
  return (filterAsyncLocalLoadsDependencies(op1, op2, allocation) ||
          filterLDSMemoryBarriersDependencies(op1, op2) ||
          filterWarpLocalAccesses(op1, op2));  // NEW
}
```

```
  membar analysis
       │
       ▼
  ┌──────────────────────────────────────────────────────┐
  │ BlockInfo::isIntersected()                           │
  │   for each (slice, ops) pair with overlapping slices │
  │     → MembarFilterFn(op1, op2, ...)                  │
  │       1. filterAsyncLocalLoadsDependencies (existing) │
  │       2. filterLDSMemoryBarriersDependencies          │
  │       3. filterWarpLocalAccesses (NEW)                │
  │          → hasMatchingWarpDistribution(op1, op2)      │
  │            compares normalized warpsPerCTA             │
  └──────────────────────────────────────────────────────┘
```

**Current scope**: Only fires when at least one op is
`AsyncTDMCopyGlobalToLocalOp`. Extensible to `AsyncCopyGlobalToLocalOp`,
`local_store`/`local_load` pairs, and `MemWaitOpTrait` barrier
suppression (see sections below).

### Relationship to `isCvtDimSync` (ConvertLayoutOp)

Triton already has warp-level sync detection for `ConvertLayoutOp` via
`isCvtDimSync` (`lib/Analysis/Utility.cpp`), recently improved by
[PR #9778](https://github.com/triton-lang/triton/pull/9778). These two
solutions address the same conceptual question — "does data cross warp
boundaries?" — but operate on **different codepaths** and handle
**different op types**.

**`isCvtDimSync`** handles `ConvertLayoutOp`, which is a single op that
internally allocates a scratch buffer, writes, syncs, and reads. Membar
treats this as a special case (the `scratchBufferId` path in `Membar.cpp`):

```cpp
if (auto cvt = dyn_cast<ConvertLayoutOp>(op)) {
    auto srcLayout = triton::gpu::toLinearLayout(srcTy);
    auto dstLayout = triton::gpu::toLinearLayout(dstTy);
    isWarpSync = isCvtDimSync(srcLayout, dstLayout, kWarp);
}
```

**`warpsPerCTA` comparison** handles separate write/read op pairs (TDM →
local_load, async_copy → local_load, local_store → local_load). These are
distinct operations in the IR, and the membar analysis checks barriers
between them via the general `isIntersected` path and `MembarFilterFn`.

The two solutions cannot substitute for each other:

| | `isCvtDimSync` | `warpsPerCTA` comparison |
|---|---|---|
| **Op scope** | `ConvertLayoutOp` (single op, internal scratch) | Separate write/read op pairs |
| **Membar codepath** | `scratchBufferId` branch (intra-op) | `isIntersected` + `MembarFilterFn` (inter-op) |
| **Mechanism** | `LinearLayout` algebra (`invertAndCompose`, `isTrivialOver`) | Direct vector comparison |
| **Handles broadcasting** | Yes — `getFreeVariableMasks` detects deduplication | Not needed — async copies and local_store don't broadcast |
| **Barrier outcome** | Downgrades CTA → warp.sync (within the op) | Eliminates inter-op barrier entirely |

**Why `isCvtDimSync` needs `LinearLayout` but `warpsPerCTA` does not:**
`ConvertLayoutOp` can involve **broadcasting** — where the source or
destination layout replicates a value across multiple threads. Broadcasting
causes write deduplication: only one thread writes each unique value. This
can create cross-warp dependencies even when `warpsPerCTA` matches, because
a warp might not write every byte it later needs to read (another warp's
thread handled that write during deduplication). The `getFreeVariableMasks`
check in `isCvtDimSync` detects this. TDM copies, `async_copy`, and
`local_store`/`local_load` don't broadcast — each element is written
exactly once by exactly one warp — so the simpler `warpsPerCTA` check is
sufficient.

### Extension to `async_copy`

The `warpsPerCTA` check naturally extends to `AsyncCopyGlobalToLocalOp`
(NVIDIA `cp.async`, AMD `buffer_load_to_local`). The writer's `warpsPerCTA`
comes from the source pointer tensor's distributed encoding; the reader's
comes from the `local_load` result's distributed encoding. Both are
standard `DistributedEncodingTrait` attributes — no vendor-specific API
needed (unlike TDM which requires `tdmGetWarpDistribution`).

```
  async_copy_global_to_local:
    src = TT_PtrTensor with distributed encoding → warpsPerCTA = [4, 1]
    dst = MemDescType (shared memory)

  local_load:
    src = MemDescType (shared memory)
    result = tensor with distributed encoding   → warpsPerCTA = [4, 1]

  [4, 1] == [4, 1] → MATCH → barrier can be suppressed
```

This makes the `warpsPerCTA` check a strong candidate for common membar
analysis code (not AMD-specific), since `async_copy` is used by both
NVIDIA and AMD backends.

## Problem 2-2: `MemWaitOpTrait` Unconditional Barrier

This is a **separate problem** from the write/read pair barrier (Problem
2-1). Even if the `warpsPerCTA` check successfully suppresses barriers
between writer and reader, the membar analysis has an independent
codepath that inserts a CTA-wide barrier unconditionally after any op
with `MemWaitOpTrait` (`async_wait`, `async_tdm_wait`,
`async_tma_store_wait`):

```cpp
// Membar.cpp — MembarAnalysis::update()
if (op->hasTrait<mlir::OpTrait::MemWaitOpTrait>() &&
    !containsLocalBarrier(op->getNextNode())) {
    builder->setInsertionPointAfter(op);
    insertBarrier(op, builder);   // CTA-wide barrier
    blockInfo->sync();            // clears all tracked dependencies
    return;                       // early return — filter never consulted
}
```

This handler fires **before** the `isIntersected` / `MembarFilterFn` path.
It inserts a CTA barrier and returns — the filter is never consulted.
After `sync()`, all tracked write/read dependencies are cleared, so the
subsequent `local_load` sees a clean `blockInfo` and doesn't trigger any
additional barrier.

In a warp-disjoint pipeline:

```
  async_copy  → blockInfo records shared write
  async_wait  → unconditional CTA barrier inserted, sync()  ← UNNECESSARY
  local_load  → blockInfo is clean, no barrier needed
```

### Why the Original Code Uses the Unconditional Handler

Both `AsyncCopyGlobalToLocalOp` and `AsyncTDMCopyGlobalToLocalOp`
declare `MemWrite<SharedMemory>` in their ODS definitions, so membar
**does** track their writes in `blockInfo` via the `getEffects` path.
And `AsyncWaitOp` has **no** `MemoryEffectsOpInterface` — only
`MemWaitOpTrait` — so it would be fully transparent to `isIntersected`.

One might expect that removing the unconditional handler would let the
normal RAW dependency path handle it:

```
  async_copy   → blockInfo records: WRITE to shared [0, N)
  async_wait   → transparent (no memory effects, passes through)
  local_load   → isIntersected sees: async_copy WRITE vs local_load READ
                 → RAW overlap → barrier
```

On **current `main`** this would produce a barrier — but only because
membar's allocation tracking treats all buffer slots as aliasing (same
`bufferId`, same interval). The barrier fires at `local_load` instead
of at `async_wait`, but is functionally equivalent.

However, the original authors chose the unconditional handler for good
reasons:

**1. Simplicity and robustness.** The rule "wait → barrier, always" is
trivially correct. It doesn't depend on `async_copy` write tracking
surviving across loop iterations, fixed-point convergence dynamics, or
interval aliasing behavior. The barrier is placed at the semantically
correct point — immediately after DMA completion.

**2. The behavioral equivalence is fragile.** The `isIntersected` path
only produces the same barrier because membar currently **cannot**
distinguish buffer slots (all slots alias to the same allocation
interval). Our Problem 1 work (`BufferIndexExpr`, Presburger) aims to
change exactly this — once membar can prove slot disjointness, the
`async_copy` write to slot C and `local_load` read from slot A would
have non-overlapping intervals, and `isIntersected` would see **no
conflict** → **no barrier** → **wrong**.

**3. Interaction with `MemAsyncWriteOpTrait`.** On the `membar_async`
branch, `MemAsyncWriteOpTrait` intentionally clears async DMA writes
from `blockInfo` before `join()` — because between `async_copy` and
`async_wait`, a CTA barrier cannot make in-flight DMA data visible,
so the write shouldn't trigger false RAW barriers with intervening
reads. With async writes cleared, `isIntersected` at `local_load`
would see **nothing** in `blockInfo` → **no barrier at all**. The
unconditional handler becomes the **sole** barrier source.

The two mechanisms work together by design:

```
  MemAsyncWriteOpTrait:
    Prevents false barriers BEFORE async_wait
    (DMA is in-flight, CTA barrier can't make it visible)

  MemWaitOpTrait unconditional handler:
    Provides the real barrier AFTER async_wait
    (DMA is complete, cross-warp visibility needed)
```

### Implications for Refactoring

The naive "Step 1: just pass through" approach is **not** behavior-
preserving in all scenarios:

| Scenario | Remove unconditional handler? | Barrier outcome |
|----------|------|---------|
| Current `main` (no slot distinction, no `MemAsyncWriteOpTrait`) | Barrier at `local_load` (intervals alias → overlap → barrier) | Equivalent |
| After Problem 1 fix (slot distinction via `BufferIndexExpr`) | `async_copy` write to slot C, `local_load` read from slot A → no overlap → **no barrier** | **Wrong** |
| With `MemAsyncWriteOpTrait` (async writes cleared) | `blockInfo` empty at `local_load` → **no barrier** | **Wrong** |

A correct refactoring needs to address these interactions. Several
approaches are possible:

**Option A: Keep unconditional handler, add filter override.**
Keep the handler but allow `MembarFilterFn` to suppress it:

```cpp
if (op->hasTrait<mlir::OpTrait::MemWaitOpTrait>() &&
    !containsLocalBarrier(op->getNextNode())) {
    // Check if the filter says we can skip the barrier.
    // This requires building a curBlockInfo that represents
    // "all pending async writes" and calling isIntersected.
    if (!canFilterMemWaitBarrier(op, blockInfo, filter)) {
        builder->setInsertionPointAfter(op);
        insertBarrier(op, builder);
    }
    blockInfo->sync();
    return;
}
```

This preserves the handler's role as the barrier decision point while
giving the filter a chance to suppress it. The `sync()` always runs
(clearing tracked state regardless of barrier insertion).

**Option B: Tag async writes for deferred barrier.**
Instead of clearing async writes from `blockInfo` (MemAsyncWriteOpTrait)
or letting them trigger premature barriers, tag them so that:
- They don't trigger barriers before `async_wait` (DMA in-flight)
- They DO trigger barriers after `async_wait` (DMA complete)
- The filter can suppress those barriers when warp-local

This requires richer state tracking in `blockInfo` — knowing which
writes are "async-pending" vs "async-completed."

**Option C: Separate async write tracking.**
Track async writes in a dedicated `asyncWriteSlices` map that
`isIntersected` ignores. At `async_wait`, move them into regular
`syncWriteSlices`. Then the normal RAW path handles the rest, and
the filter can suppress as needed.

Each option has trade-offs in complexity and intrusiveness. The choice
depends on how much refactoring of `Membar.cpp` we're willing to do
and whether we want to keep the change AMD-specific or make it generic.

## Applicability

### When This Optimization Applies

1. **Blocked register layout with matching warp distribution**: The
   `warpsPerCTA` on the write and read sides match, and each warp owns
   a disjoint partition of tensor elements. Works with any shared encoding
   (padded, swizzled, linear) because each element always gets a unique address.

2. **MMA dot operand with matching warp distribution**: When `warpsPerCTA`
   partitions the M (or N) dimension identically on both sides. The specific
   shared encoding doesn't matter — unique-address-per-element guarantees
   address disjointness.

3. **Batched MMA with warps across batch dimension**: Each warp executes
   an independent MMA on its own partition (e.g., FA MQA decode with
   split-k). No K-dimension broadcast — strictly simpler than case 2.

4. **`async_copy` pipelines with matching warp distribution**: Same
   check as TDM — writer `warpsPerCTA` from source tensor encoding,
   reader `warpsPerCTA` from `local_load` result encoding. Combined with
   the `MemWaitOpTrait` deferred-barrier approach, eliminates both the
   inter-op barrier and the post-wait barrier.

5. **Warp-specialized Gluon kernels**: Explicit per-warp shared memory
   partitioning.

6. **Per-warp scratch buffers**: Warp-local reductions or shuffles.

### When This Does NOT Apply

1. **Mismatched warp distributions**: When the writer and reader distribute
   warps differently (e.g., TDM writes with `[4, 1]` but MMA reads with
   `[2, 2]` due to K-broadcast). The `warpsPerCTA` comparison correctly
   rejects these.

2. **Cross-warp MMA operand loading**: When the MMA layout requires each warp
   to read data from multiple warp regions (K-broadcast in standard MMA).

3. **Cooperative DMA patterns**: Where the shared encoding distributes writes
   across all warps.

4. **`ConvertLayoutOp` with broadcasting**: Layout conversions where the
   source or destination has free variables (broadcasting). These require
   `LinearLayout`-based `isCvtDimSync` analysis, not `warpsPerCTA`
   comparison, because write deduplication can create cross-warp dependencies.

5. **Non-TDM/non-async_copy code paths** (current scope limitation): The
   implementation currently only fires for `AsyncTDMCopyGlobalToLocalOp`.
   Extension to `async_copy` and `local_store`/`local_load` is
   straightforward — the `warpsPerCTA` comparison logic is the same.

### Interaction with Multi-Buffering

Warp-disjointness and buffer slot disjointness (`BufferIndexExpr`) are
**orthogonal** and compose naturally:

```
  ┌─────────────────────────────────────┐
  │ Multi-buffer allocation             │
  │ ┌───────────┬───────────┬─────────┐ │
  │ │  Slot 0   │  Slot 1   │ Slot 2  │ │  ← BufferIndexExpr
  │ │┌───┬─────┐│           │         │ │     proves slot
  │ ││W0 │ W1  ││           │         │ │     disjointness
  │ │├───┼─────┤│           │         │ │
  │ ││W2 │ W3  ││           │         │ │  ← warpsPerCTA check proves
  │ │└───┴─────┘│           │         │ │     warp disjointness
  │ └───────────┴───────────┴─────────┘ │     within a slot
  └─────────────────────────────────────┘
```

## Summary

### Problem 2-1: Write/Read Op Pair Barriers

| Aspect | Before | Implemented / Proposed |
|--------|--------|-------------|
| **Target op pairs** | None | TDM → `local_load` (implemented); `async_copy` → `local_load`, `local_store` → `local_load` (extensible) |
| **Warp awareness** | `ConvertLayoutOp` scratch only (`isCvtDimSync`) | General shared memory ops via `warpsPerCTA` comparison + one-to-one address mapping |
| **Detection** | Trivial-over-warp check | `warpsPerCTA` comparison with trailing-1 normalization |
| **Async/TDM** | Not considered | TDM `warpsPerCTA` via `tdmGetWarpDistribution`; consumer `warpsPerCTA` from distributed encoding |
| **`async_copy`** | Not considered | Extensible: writer `warpsPerCTA` from source encoding, reader from `local_load` encoding |
| **MMA operands** | Assumed cross-warp | Checkable; batched MMA with matching distributions detected |
| **Batched MMA** | Not distinguished from standard MMA | Detected: warps across batch dimension → matching `warpsPerCTA` → barrier suppressed |
| **Padded layouts** | Not considered | Handled natively (encoding-agnostic: each element always gets a unique address) |
| **Swizzled layouts** | Not considered | Handled natively (swizzle shuffles addresses but never puts two elements at the same address) |

### Problem 2-2: `MemWaitOpTrait` Unconditional Barrier

| Aspect | Current |
|--------|---------|
| **`MemWaitOpTrait` handler** | Unconditional barrier + `sync()` + early return |
| **Barrier decision** | At `async_wait` (always) — filter never consulted |
| **Why not just remove it** | Equivalence with `isIntersected` is fragile: breaks with slot distinction (Problem 1) or `MemAsyncWriteOpTrait` |
| **Refactoring options** | (A) Keep handler + add filter override; (B) Tag async writes for deferred barrier; (C) Separate async write tracking |
| **Goal** | Allow `warpsPerCTA` filter to suppress post-wait barrier for warp-local cases |

---

## Appendix A: GF(2) Linear Independence (Design Alternative)

> This was the original design proposal. The `warpsPerCTA` comparison
> superseded it because it is simpler, handles padded layouts, and covers
> all practical cases. This appendix is retained as background for
> understanding `LinearLayout`'s address model.

### How LinearLayout Computes Addresses

For `local_store` / `local_load`, Triton computes a **composed layout**
that maps thread coordinates `{register, lane, warp}` to shared memory
byte offsets:

```cpp
auto cvt = regLayout.invertAndCompose(sharedLayout);
// cvt: {register, lane, warp, block} → {offset}
```

`LinearLayout` operates over **GF(2)** (Galois Field of order 2), where
addition is XOR and multiplication is AND. Each input bit (register index,
lane ID, warp ID) selects a basis vector, and all selected vectors are
combined via XOR to produce the offset:

```
  offset = Σ(reg_bit_i × reg_base_i) ⊕ Σ(lane_bit_j × lane_base_j)
           ⊕ Σ(warp_bit_k × warp_base_k)
```

### The GF(2) Independence Check

The per-warp address sets are disjoint if and only if the warp basis vectors
are **linearly independent** of the register/lane basis vectors over GF(2):

```
  rank(R ∪ L ∪ W) = rank(R ∪ L) + |W|
```

When warp bases are independent, changing the warp ID flips offset bits that
no register/lane combination can compensate for:

```
  Independent (warp-disjoint):

  register bases:  [ 0 0 0 0 0 0 0 1 ]   ← bit 0
                   [ 0 0 0 0 0 0 1 0 ]   ← bit 1
                   [ 0 0 0 0 0 1 0 0 ]   ← bit 2
  lane bases:      [ 0 0 0 0 1 0 0 0 ]   ← bit 3
                   [ 0 0 0 1 0 0 0 0 ]   ← bit 4
  ─────────────────────────────────────
  warp bases:      [ 0 0 1 0 0 0 0 0 ]   ← bit 5  ← independent!
                   [ 0 1 0 0 0 0 0 0 ]   ← bit 6  ← independent!

  → Warp 0: offsets [0x00, 0x20)
    Warp 1: offsets [0x20, 0x40)
    → all disjoint ✓
```

```
  NOT independent (cross-warp):

  register bases:  [ 0 0 1 0 0 0 0 0 ]   ← bit 5  ← same as warp!
  warp bases:      [ 0 0 1 0 0 0 0 0 ]   ← bit 5  ← NOT independent

  → same address reachable by different (warp, reg) combos
  → barrier needed
```

### Why GF(2) Was Not Implemented

1. **Fails for padded layouts.** GF(2) uses XOR arithmetic, which equals
   integer addition only when there are no carries. Non-power-of-2 strides
   produce carries:

```
  GF(2) vs integer for stride = 3 (0b11):

  row = 1:  GF(2): 3     integer: 3     ✓
  row = 2:  GF(2): 6     integer: 6     ✓
  row = 3:  GF(2): 3⊕6=5 integer: 9     ✗ (carry breaks it)
```

2. **Requires `LinearLayout` computation** during membar analysis.

3. **No additional practical coverage.** The only case GF(2) catches but
   `warpsPerCTA` does not is 2D warp distributions (e.g., `[2, 2]`) where
   warps partition both rows and columns. In practice, such layouts are
   cross-warp (MMA K-broadcast) and don't benefit from the optimization.
