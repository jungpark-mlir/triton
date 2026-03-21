# Membar Analysis: Dynamic Buffer Index Disjointness

## Motivation

Triton's membar analysis determines when shared memory barriers must be inserted
between operations to prevent data hazards (RAW, WAR, WAW). The analysis tracks
which regions of shared memory each operation accesses via `AllocationSlice`, and
inserts a barrier when two slices may overlap.

For multi-buffered shared memory (common in software pipelining), the buffer is
allocated as a single `N x ...` tensor (e.g., `memdesc<3x128x128xf16>`), and
individual buffer slots are accessed via `ttg.memdesc_index`. In a typical
pipelined loop, one slot is being written (prefetched from global memory) while
a different slot is being read (consumed by compute):

```
  memdesc<3x128x128xf16> allocation
  ┌──────────┬──────────┬──────────┐
  │  Slot 0  │  Slot 1  │  Slot 2  │
  └──────────┴──────────┴──────────┘

  Iteration i:
    Write to slot[(i + 2) % 3]   ← prefetch next tile
    Read from slot[i % 3]        ← consume current tile

  i=0:  read slot 0,  write slot 2   ← disjoint
  i=1:  read slot 1,  write slot 0   ← disjoint
  i=2:  read slot 2,  write slot 1   ← disjoint
```

These two slots are always disjoint, so no barrier is needed between them within
the same iteration. However, because `memdesc_index` uses a *dynamic* index,
the prior membar analysis could not distinguish the slots — it conservatively
assumed they might overlap and inserted an unnecessary barrier:

```
  AllocationSlice::intersects() sees:

  Write: memdesc_index %alloc[%w_idx]   →  interval = [0, 3×128×128×2)
  Read:  memdesc_index %alloc[%r_idx]   →  interval = [0, 3×128×128×2)
                                               ↑ same interval, dynamic offset unknown
                                               → conservatively assumes overlap
                                               → inserts barrier (false positive!)
```

The existing `MemDescSubsliceOp` path already resolves static offsets, but
`MemDescIndexOp` was unhandled.

## Design

### Symbolic Buffer Index Analysis

The solution decomposes `memdesc_index` indices into a canonical
`(baseValue, constantOffset, modulus)` representation called `BufferIndexExpr`.
Two accesses are provably disjoint when they share the same SSA base value, the
same modulus, and their constant offsets differ modulo N:

```
  slot[remsi(phase, 3)]       → {base=%phase, offset=0, mod=3}
  slot[remsi(phase + 2, 3)]   → {base=%phase, offset=2, mod=3}
                                    ↑ same     ↑ 0≠2     ↑ same
                                        ∴ provably disjoint
```

A barrier is only suppressed when all of the following hold:

1. Both accesses have a `BufferIndexExpr` (from `MemDescIndexOp`)
2. Neither is loop-carried
3. Both share the same SSA base value
4. Both share the same constant modulus (or neither has one)
5. The offsets are different (modulo N when applicable)

If any condition fails, the analysis conservatively inserts a barrier.

### Modulus Tracking

When `analyzeBufferIndex` encounters a modular operation (`arith.remsi` or
`select/cmpi`), it records the modulus N in `BufferIndexExpr::modulus`.
The disjointness comparison then reduces offsets mod N before comparing. This
prevents false disjointness claims for congruent offsets — e.g., offsets 0 and 3
are the same slot when `numBuffers = 3`.

Modular patterns are only matched when N is a **compile-time constant**. If N is
dynamic, the pattern is not recognized and the index falls through to an opaque
representation. This ensures two unambiguous states:

- **Modulus present** → offsets are compared mod N.
- **Modulus absent** → no modular operation was stripped; raw offsets are exact
  integer values and direct comparison is sound.

### Loop-Carried Dependencies

The membar analysis propagates pending read/write slices through the control
flow graph, including across loop backedges, to detect cross-iteration hazards.

```
  Iteration N                      Iteration N+1
  ┌───────────────────────┐       ┌───────────────────────┐
  │ %phase_N = ...        │       │ %phase_N1 = ...       │
  │                       │       │                       │
  │ write slot[f(%phase)] │──────▶│ read slot[g(%phase)]  │
  │ read  slot[g(%phase)] │  back │ write slot[f(%phase)] │
  │                       │  edge │                       │
  └───────────────────────┘       └───────────────────────┘
         ↑ intra-iteration:              ↑ cross-iteration:
           same %phase SSA                 different %phase SSA
           → can prove disjoint            → cannot compare bases
```

Within a single iteration, if two `memdesc_index` ops share the same SSA base
value `%phase`, their constant offsets directly reflect which buffer slots they
access. But when a slice crosses a backedge, its `%phase` refers to the
*previous* iteration's value — a different SSA definition — so comparing it
against a current-iteration slice on the basis of "same base" would be
incorrect.

To handle this, the `resolve()` method uses MLIR's `DominanceInfo` to identify
backedges and marks all slices propagated across them as loop-carried via
`joinLoopCarried()`. The `BufferIndexExpr` disjointness check is then
**skipped** for any loop-carried slice, falling back to the original
conservative behavior. The optimization only applies to intra-iteration
read/write pairs where the base SSA value is meaningful.

## Implementation

### `BufferIndexExpr` (Membar.h)

```cpp
struct BufferIndexExpr {
  Value baseValue;                  // dynamic SSA base (nullptr for constants)
  int64_t constantOffset = 0;       // compile-time offset from the base
  std::optional<int64_t> modulus;   // modulus N from remsi/select-cmpi
};
```

### `AllocationSlice` Additions (Membar.h)

- `std::optional<BufferIndexExpr> bufferIndexExpr` — set for `MemDescIndexOp`.
- `bool isLoopCarried` — set for slices propagated across loop backedges.

In `intersects()`, the check runs after the allocation interval check but before
the subslice offset check:

```cpp
if (bufferIndexExpr && other.bufferIndexExpr &&
    !isLoopCarried && !other.isLoopCarried) {
  if (bufferIndexExpr->isProvablyDifferentFrom(*other.bufferIndexExpr))
    return false;
}
```

### `analyzeBufferIndex` (Membar.cpp)

Recursive decomposition of an index value into `BufferIndexExpr`:

```
  remsi(addi(%phase, 2), 3)
       │
       ▼
  ┌─ remsi ────────────────────────┐
  │  modulus = 3                   │
  │  ┌─ addi ────────────────────┐ │
  │  │  offset += 2              │ │
  │  │  ┌─ %phase ─────────────┐ │ │
  │  │  │  base = %phase        │ │ │
  │  │  │  offset = 0           │ │ │
  │  │  └───────────────────────┘ │ │
  │  └────────────────────────────┘ │
  └────────────────────────────────┘
  Result: {base=%phase, offset=2, mod=3}
```

| IR Pattern | Decomposition |
|---|---|
| `arith.constant C` | `{nullptr, C, nullopt}` |
| `arith.addi(x, C)` | `{base(x), offset(x) + C, nullopt}` |
| `arith.remsi(x, N)` where N is constant | `{base(x), offset(x), mod=N}` |
| `select(cmpi slt/sge, addi(base, C), N)` where N is constant | `{base, offset + C, mod=N}` |
| dynamic modulus or unrecognized pattern | `{value, 0, nullopt}` (opaque) |

The `arith.addi` case checks both operands for constants (commutative). The
`select/cmpi` pattern (`matchModuloPattern`) supports both `slt` and `sge`
predicate polarities.

### `isProvablyDifferentFrom` (Membar.h)

| Condition | Behavior |
|---|---|
| Different base SSA values | Cannot relate → conservative |
| Both have modulus, same N | Compare `offset % N` |
| Both have modulus, different N | Cannot relate → conservative |
| One has modulus, other does not | Cannot relate → conservative |
| Neither has modulus | Compare raw offsets (exact for constants and `addi`) |

## IR Generation Requirements

The analysis operates on the IR as produced by the pipeliner or user-level code
(e.g., Gluon). For the disjointness proof to apply, the IR must satisfy two
requirements.

### Shared SSA Base

Both the read and write buffer indices must derive from the **same SSA base
value** with different constant offsets. If the IR uses separate iteration
arguments for load and compute phases, the analysis sees different bases and
conservatively inserts a barrier:

```
  ╔═══════════════════════════════════════════════════════╗
  ║  Separate counters → different SSA bases              ║
  ║                                                       ║
  ║    %load_idx ──┐                                      ║
  ║                ├── different bases → can't compare     ║
  ║    %wmma_idx ──┘       → conservative barrier         ║
  ╚═══════════════════════════════════════════════════════╝

  ╔═══════════════════════════════════════════════════════╗
  ║  Unified counter → same SSA base                      ║
  ║                                                       ║
  ║    %phase ─── remsi(%phase, 3) ────── offset=0        ║
  ║         │                                             ║
  ║         └─── remsi(%phase + 2, 3) ── offset=2         ║
  ║                                                       ║
  ║    same base, 0 ≠ 2 (mod 3) → provably disjoint      ║
  ╚═══════════════════════════════════════════════════════╝
```

```python
# Separate counters → different SSA bases → conservative barrier
load_slot    = (load_idx  // K) % NUM_BUFFERS
compute_slot = (wmma_idx // K) % NUM_BUFFERS

# Unified counter → same SSA base → provably disjoint
compute_slot = phase % NUM_BUFFERS
load_slot    = (phase + STAGE_OFFSET) % NUM_BUFFERS
```

### Recognized Modular Patterns

The index must use one of the recognized modular idioms so that the modulus
is recorded and offsets are compared correctly:

**`select/cmpi` modular wrap** — emitted by `createIncrementModulo`:

```mlir
// slt polarity
%sum = arith.addi %phase, %c1 : i32
%cmp = arith.cmpi slt, %sum, %c3 : i32
%idx = arith.select %cmp, %sum, %c0 : i32

// sge polarity
%sum = arith.addi %phase, %c1 : i32
%cmp = arith.cmpi sge, %sum, %c3 : i32
%idx = arith.select %cmp, %c0, %sum : i32
```

**`arith.remsi`** — emitted by Gluon (Python `%` operator):

```mlir
%idx = arith.remsi %phase, %c3 : i32
```

### Common Pipeliner Status

The common pipeliner (`LowerLoops.cpp`) currently creates two independent
`iter_args` — `insertIdx` and `extractIdx` — to reduce register liverange:

```
  ┌── scf.for iter_args ──────────────────────────────────┐
  │                                                       │
  │   %insertIdx  ──── createIncrementModulo ──→ insert   │
  │   %extractIdx ──── createIncrementModulo ──→ extract  │
  │        │                    │                          │
  │        └────── different SSA bases ──────┘             │
  │                     ↓                                  │
  │         BufferIndexExpr sees different bases           │
  │              → conservative barrier                   │
  └───────────────────────────────────────────────────────┘

  In reality, they always differ by a fixed stage offset:
    insertIdx  = (extractIdx + STAGE_OFFSET) % NUM_BUFFERS
```

```cpp
// Create two counters for the insert and extract indices to avoid creating
// long liverange.
loadGroup.insertIdx = createIncrementModulo(builder, loc, insertIdx,
                                            numBuffersVal, zero, one);
loadGroup.extractIdx = createIncrementModulo(builder, loc, extractIdx,
                                             numBuffersVal, zero, one, &cndExt);
```

Because the insert and extract counters are separate SSA values, the analysis
sees different bases and falls back to a conservative barrier. The two counters
always differ by a fixed constant offset and increment in lockstep, so they
could in principle be unified into a single phase counter. However, the impact
of extending the liverange depends on the backend compiler's register allocator
(e.g., NVIDIA's `ptxas`), and should be evaluated carefully before changing.

Gluon-based kernels do not share this limitation — the unified phase approach
is already applied there.

## Limitations

1. **Loop-carried dependencies.** Expression matching is disabled across loop
   iterations. This may miss optimization opportunities when the loop structure
   guarantees disjointness across iterations as well.

2. **Common pipeliner.** Uses separate insert/extract counters, preventing the
   analysis from proving disjointness. See
   [Common Pipeliner Status](#common-pipeliner-status).

## Testing

Two lit tests in `test/Conversion/amd/amdgpu_membar.mlir` use
`amdg.async_tdm_copy_global_to_local` (write) and `ttg.local_load` (read) on a
3-buffer allocation:

**`disjoint_tdm_copy_select_cmpi`** — `select/cmpi slt` pattern:

```mlir
// Write: (phase + 2) % 3       Read: (phase + 1) % 3
%w = arith.addi %phase, %c2     %r = arith.addi %phase, %c1
%wc = arith.cmpi slt, %w, %c3   %rc = arith.cmpi slt, %r, %c3
%wi = arith.select %wc, %w, %c0 %ri = arith.select %rc, %r, %c0
```

**`disjoint_tdm_copy_remsi`** — `arith.remsi` pattern:

```mlir
// Write: (phase + 2) % 3       Read: phase % 3
%w = arith.addi %phase, %c2     %ri = arith.remsi %phase, %c3
%wi = arith.remsi %w, %c3
```

Both assert `CHECK-NOT: ttg.barrier local` between the TDM copy and
`local_load`. Verified against baseline `triton-opt`:

| Test | Old Code | New Code |
|---|---|---|
| `disjoint_tdm_copy_select_cmpi` | `ttg.barrier local` inserted | No barrier |
| `disjoint_tdm_copy_remsi` | `ttg.barrier local` inserted | No barrier |

## Files Changed

| File | Change |
|---|---|
| `include/triton/Analysis/Membar.h` | Added `BufferIndexExpr`, `bufferIndexExpr` and `isLoopCarried` on `AllocationSlice`, `joinLoopCarried()` on `BlockInfo` |
| `lib/Analysis/Membar.cpp` | Added `analyzeBufferIndex()`, `matchModuloPattern()`, `bufferIndexExpr` check in `intersects()`, backedge detection via `DominanceInfo` |
| `test/Conversion/amd/amdgpu_membar.mlir` | Added `disjoint_tdm_copy_select_cmpi` and `disjoint_tdm_copy_remsi` lit tests |
