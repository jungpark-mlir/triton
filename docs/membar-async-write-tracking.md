# Membar: Async Write Tracking

## Motivation

Triton's membar analysis has a special-case handler for `MemWaitOpTrait`
ops (`async_wait`, `async_tdm_wait`, etc.) that unconditionally inserts
a CTA-wide barrier and clears all tracked dependencies:

```cpp
// Current code — Membar.cpp, MembarAnalysis::update()
if (op->hasTrait<mlir::OpTrait::MemWaitOpTrait>() &&
    !containsLocalBarrier(op->getNextNode())) {
    builder->setInsertionPointAfter(op);
    insertBarrier(op, builder);   // CTA-wide barrier, always
    blockInfo->sync();            // clears everything
    return;                       // early return — filter never consulted
}
```

This creates two problems:

1. **The filter is never consulted.** The `MembarFilterFn` (which handles
   warp-local access suppression via `warpsPerCTA` comparison) is
   bypassed entirely. Even when the write and read are provably
   warp-disjoint, the barrier is inserted.

2. **The barrier cannot be removed without breaking correctness.** Simply
   deleting the handler fails when shared memory reads are interleaved
   with in-flight DMA writes (see [Failure Case](#failure-case-interleaved-reads)).

This document proposes **async write tracking** — a separation of DMA
writes into a dedicated `asyncWriteSlices` map that survives CTA
barriers but becomes visible to `isIntersected` only after `async_wait`
promotes them. This unifies the barrier decision into the existing
`isIntersected` + `MembarFilterFn` path with no special-case barrier
insertion.

## Background: Why Async Writes Are Different

Async DMA copies (`async_copy_global_to_local`, `async_tdm_copy_...`)
issue commands to a DMA engine that writes to shared memory independently
of the thread. Two properties distinguish these writes from synchronous
stores:

1. **A CTA barrier cannot make DMA data visible.** `s_barrier` /
   `__syncthreads()` synchronizes threads, not the DMA engine. Only
   an explicit wait (`async_wait`, `async_tdm_wait`) ensures the DMA
   has completed and the data is in shared memory.

2. **The write has a lifecycle.** After `async_copy`, the write is
   *in-flight*. After `async_wait`, it is *completed* (visible). After
   a barrier + `sync()`, it is *consumed* (cleared from tracking).

Current `BlockInfo` has no notion of this lifecycle — all writes go
into `syncWriteSlices` and are treated identically.

## Design: Separate Async Write Map

Add a third map to `BlockInfo`:

```cpp
struct BlockInfo {
  using SliceMapT = std::map<AllocationSlice, std::set<Operation *>>;

  SliceMapT syncReadSlices;
  SliceMapT syncWriteSlices;
  SliceMapT asyncWriteSlices;   // NEW: in-flight DMA writes
};
```

### Lifecycle Rules

```
  async_copy       async_wait       barrier/sync()
      │                │                 │
      ▼                ▼                 ▼
 ┌──────────┐    ┌──────────┐     ┌──────────┐
 │ In-flight │───▶│ Completed│────▶│ Consumed │
 │           │    │          │     │          │
 │ asyncWrite│    │ syncWrite│     │ (cleared)│
 │ Slices    │    │ Slices   │     │          │
 └──────────┘    └──────────┘     └──────────┘
     │                │
     │  invisible to  │  visible to
     │  isIntersected │  isIntersected
     │                │
     │  survives      │  cleared by
     │  sync()        │  sync()
```

- **`async_copy`**: write goes to `asyncWriteSlices` (not `syncWriteSlices`)
- **`async_wait`**: promotes `asyncWriteSlices` → `syncWriteSlices`
- **`sync()` (barrier)**: clears `syncWriteSlices` (and `syncReadSlices`),
  but **NOT** `asyncWriteSlices` — because a thread barrier cannot make
  in-flight DMA visible
- **`isIntersected`**: only checks `syncWriteSlices` (unchanged)

### Key Invariant

> A CTA barrier clears synchronous state but not async state.
> Only `async_wait` transitions async writes into synchronous tracking.

This invariant reflects the hardware semantics: `s_barrier` synchronizes
threads, `async_wait` synchronizes the DMA engine.

## Implementation

### `BlockInfo` Changes

```cpp
struct BlockInfo {
  using SliceMapT = std::map<AllocationSlice, std::set<Operation *>>;

  SliceMapT syncReadSlices;
  SliceMapT syncWriteSlices;
  SliceMapT asyncWriteSlices;

  BlockInfo &join(const BlockInfo &other) {
    for (auto &slice : other.syncReadSlices)
      syncReadSlices[slice.first].insert(slice.second.begin(),
                                         slice.second.end());
    for (auto &slice : other.syncWriteSlices)
      syncWriteSlices[slice.first].insert(slice.second.begin(),
                                          slice.second.end());
    for (auto &slice : other.asyncWriteSlices)
      asyncWriteSlices[slice.first].insert(slice.second.begin(),
                                           slice.second.end());
    return *this;
  }

  void sync() {
    syncReadSlices.clear();
    syncWriteSlices.clear();
    // asyncWriteSlices NOT cleared — DMA writes survive thread barriers
  }

  void promoteAsyncWrites() {
    for (auto &[slice, ops] : asyncWriteSlices)
      syncWriteSlices[slice].insert(ops.begin(), ops.end());
    asyncWriteSlices.clear();
  }

  bool operator==(const BlockInfo &other) const {
    return syncReadSlices == other.syncReadSlices &&
           syncWriteSlices == other.syncWriteSlices &&
           asyncWriteSlices == other.asyncWriteSlices;
  }

  // isIntersected: UNCHANGED — only checks syncReadSlices/syncWriteSlices
};
```

### `MembarAnalysis::update()` Changes

Two changes. First, at `async_copy` time, redirect writes to the async
map. Second, at `async_wait` time, promote and let the normal path run.

```cpp
void MembarAnalysis::update(Operation *op, BlockInfo *blockInfo,
                            FuncBlockInfoMapT *funcBlockInfoMap,
                            OpBuilder *builder) {
  // ... containsLocalBarrier check (unchanged) ...

  // CHANGED: MemWaitOpTrait handler — promote only, no barrier/sync/return
  if (op->hasTrait<mlir::OpTrait::MemWaitOpTrait>())
    blockInfo->promoteAsyncWrites();
  // Fall through — async_wait has no MemoryEffectsOpInterface,
  // so curBlockInfo will be empty, isIntersected returns false,
  // join is a no-op. The promoted writes are now in syncWriteSlices
  // for future ops to check against.

  // ... getEffects populates curBlockInfo (unchanged) ...

  // NEW: redirect async DMA writes to asyncWriteSlices
  if (op->hasTrait<OpTrait::MemAsyncWriteOpTrait>()) {
    curBlockInfo.asyncWriteSlices =
        std::move(curBlockInfo.syncWriteSlices);
    curBlockInfo.syncWriteSlices.clear();
  }

  // ... scratchBufferId / isIntersected / join (unchanged) ...
}
```

### `translateBlockInfoToCallsite` Changes

```cpp
inline BlockInfo translateBlockInfoToCallsite(
    const BlockInfo &calleeBlockInfo, size_t callOffset) {
  BlockInfo translatedBlockInfo;
  auto translateSlices = [&](const BlockInfo::SliceMapT &srcSlices,
                             BlockInfo::SliceMapT &dstSlices) {
    for (const auto &[slice, ops] : srcSlices) {
      auto translatedSlice =
          slice.translated(callOffset, /*invalidateBufferId=*/true);
      auto &dstOps = dstSlices[translatedSlice];
      dstOps.insert(ops.begin(), ops.end());
    }
  };
  translateSlices(calleeBlockInfo.syncReadSlices,
                  translatedBlockInfo.syncReadSlices);
  translateSlices(calleeBlockInfo.syncWriteSlices,
                  translatedBlockInfo.syncWriteSlices);
  translateSlices(calleeBlockInfo.asyncWriteSlices,    // NEW
                  translatedBlockInfo.asyncWriteSlices);
  return translatedBlockInfo;
}
```

### Required Trait

`MemAsyncWriteOpTrait` must be added to all async DMA write ops:

| Op | Dialect | Write effect |
|----|---------|-------------|
| `AsyncCopyGlobalToLocalOp` | TritonGPU | `MemWrite<SharedMemory>` on `$result` |
| `AsyncTDMCopyGlobalToLocalOp` | TritonAMDGPU | `MemWrite<SharedMemory>` on `$result` |
| `AsyncTDMGatherOp` | TritonAMDGPU | `MemWrite<SharedMemory>` on `$result` |
| `BufferLoadToLocalOp` | TritonAMDGPU | `MemWrite<SharedMemory>` on `$result` |

These ops already declare `MemWrite<SharedMemory>` in ODS, so membar
tracks them as writes via `getEffects`. The trait just tells
`update()` to redirect the write to `asyncWriteSlices`.

## Behavior-Preserving Verification

### Step 1: Exact replication of current barriers

To verify correctness, we can first deploy the infrastructure while
keeping the unconditional barrier — proving it produces identical
barrier placement:

```cpp
// Step 1: promote + unconditional barrier (same as current)
if (op->hasTrait<mlir::OpTrait::MemWaitOpTrait>() &&
    !containsLocalBarrier(op->getNextNode())) {
    blockInfo->promoteAsyncWrites();
    builder->setInsertionPointAfter(op);
    insertBarrier(op, builder);    // same barrier position
    blockInfo->sync();             // same clear
    return;
}
```

This is **identical** to the current code plus one additional call
(`promoteAsyncWrites`). Since `promoteAsyncWrites` moves writes to
`syncWriteSlices` which `sync()` immediately clears, the net effect
is: promote + clear = same as the current clear (which clears
`syncWriteSlices` that already had the writes). The only difference
is that `asyncWriteSlices` is now also cleared by the promote.

In a standard pipelined loop, async_copy writes went to
`asyncWriteSlices` instead of `syncWriteSlices`, but the handler
promotes and clears them in the same step. The barrier is at the
same IR position. Lit tests pass unchanged.

### Step 2: Remove unconditional barrier

Once Step 1 is validated, remove the barrier and let `isIntersected`
decide:

```cpp
// Step 2: promote only — barrier decided by isIntersected
if (op->hasTrait<mlir::OpTrait::MemWaitOpTrait>())
    blockInfo->promoteAsyncWrites();
// No barrier, no sync, no early return.
// Fall through to normal path.
```

After promotion, the async_copy write is in `syncWriteSlices`. When
`local_load` comes, `isIntersected` sees the RAW dependency and
inserts a barrier. The filter can suppress it for warp-local cases.

## Tricky Cases

### Case 1: Standard pipelined loop (AMD ordering)

```
loop body:
  async_wait(N-2)
  local_load(slot A)
  compute
  async_copy(slot C)
```

**Current behavior**: barrier at `async_wait`, no barrier at `local_load`.

**With async write tracking (Step 2)**:

```
blockInfo at loop entry: asyncWriteSlices = {WRITE[C]} (from backedge)

async_wait       → promoteAsyncWrites: WRITE[C] → syncWriteSlices
                   (no barrier, fall through)
local_load(A)    → isIntersected: READ[A] vs WRITE[C] → RAW
                   → barrier + sync()
                   Barrier is AFTER async_wait, BEFORE local_load.
compute          → no shared effects
async_copy(C)    → asyncWriteSlices: {WRITE[C]}

Backedge: asyncWriteSlices = {WRITE[C]}
```

Barrier count: **1** (at `local_load`). Same as current (1 at `async_wait`).
Position: functionally equivalent — after DMA completion, before read.

### Case 2: NVIDIA-style ordering

```
loop body:
  async_copy(slot C)
  async_commit
  async_wait(N-2)
  local_load(slot A)
  compute
```

**Current behavior**: barrier at `async_wait`. Potentially a WAR barrier
at `async_copy` (from previous iteration's `local_load` read).

**With async write tracking (Step 2)**:

```
blockInfo at loop entry: syncReadSlices = {READ[A]}, asyncWriteSlices = {WRITE[C]}

async_copy(C)    → asyncWriteSlices: {WRITE[C]}
                   isIntersected: WRITE goes to async, not sync
                   WAR check: syncReadSlices {READ[A]} vs curBlockInfo syncWriteSlices (empty)
                   → no WAR barrier
async_commit     → no effects
async_wait       → promoteAsyncWrites: WRITE[C] → syncWriteSlices
local_load(A)    → isIntersected: READ[A] vs WRITE[C] → RAW → barrier + sync()
compute

Backedge: syncReadSlices = {READ[A]}, asyncWriteSlices = {WRITE[C]}
```

Barrier count: **1** (at `local_load`).
**Current code produces 2**: 1 at `async_wait` (unconditional) + 1 at
`async_copy` (WAR from previous iteration's read). The WAR barrier at
`async_copy` is a false positive — it fires because the async write
goes into `syncWriteSlices`, triggering `isIntersected` against the
previous iteration's `local_load` READ. With async write tracking, the
write goes to `asyncWriteSlices` (invisible to `isIntersected`), so
the false WAR doesn't fire. **Fewer barriers, same correctness.**

### Case 3: Interleaved shared memory reads during in-flight DMA

```
async_copy(slot C)          // start DMA
result = local_load(other)  // read from different shared buffer
async_wait(0)               // DMA complete
data = local_load(slot A)   // read DMA result
```

This is the case that **breaks** if you simply remove the handler
without async write tracking.

**Without async write tracking (handler removed)**:

```
async_copy(C)       → syncWriteSlices: {WRITE[C]}
local_load(other)   → isIntersected: READ[other] vs WRITE[C]
                      intervals overlap (aliasing) → barrier + sync()
                      ↑ FALSE POSITIVE (DMA in-flight, barrier useless)
                      syncWriteSlices cleared!
async_wait          → pass through (no handler)
local_load(A)       → isIntersected: syncWriteSlices empty → NO BARRIER
                      ← WRONG: DMA data needs cross-warp visibility
```

The false-positive barrier's `sync()` cleared the async_copy write.
After `async_wait`, nothing remains to trigger a barrier.

**With async write tracking**:

```
async_copy(C)       → asyncWriteSlices: {WRITE[C]}
local_load(other)   → isIntersected: syncWriteSlices empty → no barrier
                      (asyncWriteSlices invisible — no false positive)
async_wait          → promoteAsyncWrites: WRITE[C] → syncWriteSlices
local_load(A)       → isIntersected: READ[A] vs WRITE[C] → RAW → barrier ✓
```

Both problems solved: no false barrier before `async_wait`, real barrier
after it.

### Case 4: ConvertLayoutOp between async_copy and async_wait

```
async_copy(slot C)      // start DMA
convert_layout(...)     // uses scratch shared memory (write + read)
async_wait(0)           // DMA complete
local_load(slot A)      // read DMA result
```

`ConvertLayoutOp` has a scratch buffer, so it enters the `scratchBufferId`
path in `update()`. That path does its own `isIntersected` + `sync()`.

**Without async write tracking (handler removed)**:

```
async_copy(C)       → syncWriteSlices: {WRITE[C]}
convert_layout      → scratchBufferId path:
                      curBlockInfo: WRITE[scratch] (added for RAW check)
                      isIntersected: WRITE[scratch] vs WRITE[C] → WAW?
                      if overlapping → barrier + sync()
                      syncWriteSlices cleared!
async_wait          → pass through
local_load(A)       → nothing in syncWriteSlices → NO BARRIER ← WRONG
```

**With async write tracking**:

```
async_copy(C)       → asyncWriteSlices: {WRITE[C]}
convert_layout      → scratchBufferId path:
                      isIntersected: syncWriteSlices empty → no WAW
                      (asyncWriteSlices invisible)
                      scratch path does its own sync
async_wait          → promoteAsyncWrites: WRITE[C] → syncWriteSlices
local_load(A)       → isIntersected: READ[A] vs WRITE[C] → barrier ✓
```

### Case 5: Multiple async_copy with interleaved async_wait

```
async_copy(slot 0)    // DMA to slot 0
async_copy(slot 1)    // DMA to slot 1
async_wait(1)         // wait until at most 1 outstanding
local_load(slot 0)    // slot 0 is complete
// ... more work ...
async_wait(0)         // wait for everything
local_load(slot 1)    // slot 1 is complete
```

**With async write tracking**:

```
async_copy(0)       → asyncWriteSlices: {WRITE[0]}
async_copy(1)       → asyncWriteSlices: {WRITE[0], WRITE[1]}
async_wait(1)       → promoteAsyncWrites: {WRITE[0], WRITE[1]} → syncWriteSlices
                      (promotes ALL — conservative; we don't track
                       which specific copies the wait covers)
local_load(0)       → isIntersected: READ[0] vs WRITE[0,1] → RAW → barrier
                      sync() clears syncWriteSlices
async_wait(0)       → promoteAsyncWrites: nothing to promote (already promoted)
local_load(1)       → isIntersected: syncWriteSlices empty → no barrier
```

The first `async_wait` conservatively promotes all pending writes.
This means `local_load(slot 1)` doesn't get a barrier even though
it should (slot 1 was just completed by `async_wait(0)`). However,
the barrier at `local_load(slot 0)` already provides the CTA-wide
synchronization that covers both slots (since `sync()` is CTA-wide).

**Current code** also handles this conservatively — the first
`async_wait(1)` inserts a barrier + `sync()`, then `async_wait(0)`
inserts another barrier + `sync()`. Two barriers. The async write
tracking version produces one. Both are correct — the first barrier
ensures cross-warp visibility for all DMA data that has been waited on.

> **Note**: Tracking per-wait granularity (which specific async_copies
> correspond to which async_wait) would require token-chain analysis.
> This is a potential future refinement but not necessary for
> correctness — conservative promotion is safe.

### Case 6: Manual barrier between async_copy and async_wait

```
async_copy(slot C)
__syncthreads()       // user-placed barrier (e.g., for unrelated sync)
async_wait(0)
local_load(slot A)
```

**With async write tracking**:

```
async_copy(C)       → asyncWriteSlices: {WRITE[C]}
__syncthreads()     → containsLocalBarrier → sync()
                      sync() clears syncReadSlices, syncWriteSlices
                      asyncWriteSlices preserved: {WRITE[C]}
async_wait          → promoteAsyncWrites: WRITE[C] → syncWriteSlices
local_load(A)       → isIntersected: RAW → barrier ✓
```

The manual barrier clears synchronous state but the async write
survives. This is correct — the user's barrier may synchronize
threads for some other reason, but it cannot make in-flight DMA
visible.

## Using the Filter

With async write tracking (Step 2), `isIntersected` is the sole
barrier decision point. The `MembarFilterFn` is consulted for every
hazard pair — including pairs where the writer is an `async_copy`:

```
async_copy(C)    → asyncWriteSlices: {WRITE[C]}
async_wait       → promoteAsyncWrites → syncWriteSlices: {WRITE[C]}
local_load(A)    → isIntersected:
                   lhsOp = async_copy,  rhsOp = local_load
                   lhsIsRead = false,   rhsIsRead = true
                   → filter(async_copy, local_load, false, true, allocation)
                     → filterWarpLocalAccesses: warpsPerCTA match → suppress ✓
```

The filter sees the **original writer** (`async_copy`) because the op
stored in `syncWriteSlices` is the one that was recorded when the write
was first tracked. Promotion moves the slice entries without changing
the ops. This means the filter has full access to the writer's type,
encoding, and layout — exactly what `filterWarpLocalAccesses` needs
for the `warpsPerCTA` comparison.

### AMD Backend Filter (Current Scope)

The existing AMD `membarFilter` adds a clause for warp-local access:

```cpp
bool membarFilter(Operation *op1, Operation *op2, ...) {
  return (filterAsyncLocalLoadsDependencies(op1, op2, allocation) ||
          filterLDSMemoryBarriersDependencies(op1, op2) ||
          filterWarpLocalAccesses(op1, op2));
}
```

With async write tracking, `filterWarpLocalAccesses` now also
suppresses the post-wait barrier (which was previously unreachable
because the `MemWaitOpTrait` handler returned early). No change to
the filter function itself — just the removal of the handler that
blocked it.

## Migrating to Common Logic

The `warpsPerCTA` comparison is not inherently AMD-specific. The
check applies whenever:

- The writer distributes warps via `warpsPerCTA` (all distributed
  encodings do)
- The reader distributes warps via `warpsPerCTA`
- The shared encoding gives each element a unique address (all do)

Currently, the only AMD-specific piece is `tdmGetWarpDistribution`
for TDM ops. For `AsyncCopyGlobalToLocalOp` and `local_store` /
`local_load`, `warpsPerCTA` comes from standard `DistributedEncoding`
attributes available in core Triton.

### Migration Path

**Phase 1: AMD filter (current).**
The `warpsPerCTA` check lives in the AMD `membarFilter`. Only AMD
benefits. This is the lowest-risk starting point.

**Phase 2: Move to common filter or `isIntersected`.**

Option A — Common `MembarFilterFn`:

```cpp
// In core membar setup (not AMD-specific):
auto commonFilter = [](Operation *op1, Operation *op2,
                        bool lhsIsRead, bool rhsIsRead,
                        Allocation *allocation) -> bool {
    return filterWarpLocalAccesses(op1, op2);
};
```

Each backend composes its own filters with the common one:

```cpp
// AMD:
auto amdFilter = [&](Operation *op1, ...) {
    return commonFilter(op1, ...) ||
           filterAsyncLocalLoadsDependencies(op1, op2, allocation) ||
           filterLDSMemoryBarriersDependencies(op1, op2);
};

// NVIDIA (future):
auto nvFilter = [&](Operation *op1, ...) {
    return commonFilter(op1, ...) ||
           filterNvidiaSpecific(op1, op2);
};
```

Option B — Move into `isIntersected` directly:

```cpp
// In BlockInfo::isIntersected (private overload):
for (auto lhsOp : lhs.second)
  for (auto rhsOp : rhs.second) {
    // Common warp-local check (no filter needed):
    if (hasMatchingWarpDistribution(lhsOp, rhsOp))
      continue;  // warp-disjoint, skip this pair
    if (!filter || !filter(lhsOp, rhsOp, ...))
      return true;
  }
```

This makes warp-local detection a **built-in property** of membar
rather than a backend filter. Benefits:

- All backends get it automatically
- No filter composition needed
- The check is simple (vector comparison) — no `LinearLayout` or
  expensive computation
- `filterWarpLocalAccesses` becomes dead code in the AMD backend

The risk is coupling core membar to encoding-specific logic
(`warpsPerCTA`). This can be mitigated by abstracting behind a
`getWarpDistribution(Operation*)` utility that works for any op
with a `DistributedEncodingTrait` result/operand, with backend
hooks for special cases like TDM.

**Phase 3: Full integration.**

```
  ┌──────────────────────────────────────────────────────────┐
  │ isIntersected()                                          │
  │                                                          │
  │  for each (slice, ops) pair with overlapping slices:     │
  │    for each (writer, reader) op pair:                    │
  │      1. Warp-local check (built-in):                     │
  │         getWarpDistribution(writer) ==                   │
  │         getWarpDistribution(reader)                      │
  │         → skip (disjoint)                                │
  │                                                          │
  │      2. Backend filter (optional):                       │
  │         MembarFilterFn(writer, reader, ...)              │
  │         → backend-specific suppression                   │
  │                                                          │
  │      3. Neither → barrier needed                         │
  └──────────────────────────────────────────────────────────┘
```

Async write tracking is orthogonal to where the `warpsPerCTA` check
lives. The tracking ensures the writer op reaches `isIntersected`
at the right time (after DMA completion); the check decides whether
the barrier is needed. Both can evolve independently.

## Summary

| Aspect | Current | Async Write Tracking |
|--------|---------|---------------------|
| **Async writes stored in** | `syncWriteSlices` | `asyncWriteSlices` (separate map) |
| **Barrier at `async_wait`** | Unconditional (handler) | None (promotion only) |
| **Barrier decision** | Handler bypasses `isIntersected` | `isIntersected` at reader op |
| **Filter consulted** | Never (handler returns early) | Always (via `isIntersected`) |
| **False positives before wait** | Yes (async write in syncWriteSlices triggers RAW) | No (async write invisible) |
| **Survives intervening barriers** | No (`sync()` clears) | Yes (`sync()` preserves asyncWriteSlices) |
| **`MemWaitOpTrait` handler** | 5 lines: barrier + sync + return | 2 lines: promote, fall through |
| **`MemAsyncWriteOpTrait`** | Needed separately (clears writes before join) | Subsumed (redirects writes to asyncWriteSlices) |
| **Fixed-point convergence** | Unchanged | `asyncWriteSlices` in `join` and `operator==` |

### Incremental Deployment

1. **Add `asyncWriteSlices` to `BlockInfo`** — update `join`, `sync`,
   `operator==`, `translateBlockInfoToCallsite`, `dump`.
2. **Add `MemAsyncWriteOpTrait` to async DMA ops** — redirect writes
   to `asyncWriteSlices` in `update()`.
3. **Step 1 (behavior-preserving)**: promote + unconditional barrier +
   sync at `MemWaitOpTrait` handler. Validates infrastructure with
   identical barrier placement.
4. **Step 2 (optimization)**: replace handler with promote-only.
   Barriers decided by `isIntersected` + filter. Validate with lit
   tests + kernel benchmarks.
5. **Step 3 (common logic)**: move `warpsPerCTA` check from AMD
   filter into common `isIntersected` or common filter. All backends
   benefit.
