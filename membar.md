# Membar Analysis: False Positive Barriers for Async Shared Memory Writes

## Problem

The membar analysis (`lib/Analysis/Membar.cpp`) inserts unnecessary `ttg.barrier local` ops between async DMA copies (e.g., `amdg.async_tdm_copy_global_to_local`) and subsequent `ttg.local_load` ops within the same loop iteration.

### Concrete example

In a warp-pipelined GEMM kernel with triple-buffered shared memory, the loop body has two stages:

```
scf.for ... {
    scf.execute_region {                       // stage0
        async_tdm_copy → a_buffer[prod%3]      // async Write to shared memory
        async_tdm_copy → b_buffer[prod%3]      // async Write to shared memory
        local_load     ← a_buffer[cons%3]      // Read from shared memory
        local_load     ← b_buffer[cons%3]      // Read from shared memory
    }

    amdg.async_tdm_wait {num = 2}             // wait for DMA completion
    // MemWaitOpTrait auto-inserts ttg.barrier local here

    scf.execute_region {                       // stage1
        tt.dot ...                             // WMMA compute
    }
}
```

The `MemWaitOpTrait` on `async_tdm_wait` already causes the membar analysis to auto-insert a `ttg.barrier local` after the wait, which provides cross-iteration synchronization. Yet the analysis **also** inserts a spurious `ttg.barrier local` between the `async_tdm_copy` ops and the `local_load` ops inside stage0. This barrier is unnecessary because:

1. The async DMA write has **not completed yet** — the data written by `async_tdm_copy` is not visible until after `async_tdm_wait` + barrier. There is no RAW (Read After Write) hazard.
2. The producer and consumer access **different slots** of the triple buffer (`prod%3 != cons%3`). They never alias.

### Root cause in the analysis

The `update()` function in `Membar.cpp` processes memory effects at lines 292-311:

```cpp
if (isa<MemoryEffects::Write>(effectInstance.getEffect()))
    curBlockInfo.syncWriteSlices[slice].insert(op);
```

The `async_tdm_copy`'s `MemWrite<SharedMemory>` is recorded as a synchronous write. When the subsequent `local_load` is processed, `isIntersected()` finds the write and the read on the same allocation with unknown subslice offsets (because `memdesc_index` uses dynamic indices), and conservatively inserts a barrier.

Two independent analysis limitations contribute:

1. **No async write model**: All writes are treated as immediately visible. The analysis has no concept of deferred/async writes.
2. **No dynamic offset tracking**: `AllocationSlice` only tracks static offsets from `MemDescSubsliceOp`. `MemDescIndexOp` with dynamic indices produces unknown offsets, causing conservative intersection.

---

## Proposed Solutions

### Near-term fix: Clear async writes before join

**Concept**: Keep `MemWrite<SharedMemory>` on the async op so that `isIntersected` still detects hazards where the async write is the *current* op (WAR and WAW against prior state). But prevent the write from propagating into `blockInfo.syncWriteSlices`, so it doesn't trigger false hazards where the async write is a *prior* op (RAW or WAW from subsequent ops).

This is correct because a `ttg.barrier local` is a thread-level shared memory fence. It synchronizes accesses made by threads (`local_store`/`local_load`), not in-flight DMA operations. An async DMA write that hasn't completed cannot be made visible by a barrier — only `async_tdm_wait` can do that. Therefore, an async write that is still in-flight should not appear as a prior write in `blockInfo`, since no barrier the analysis could insert would resolve a hazard against it.

**Implementation**:

1. Add a trait (e.g., `MemAsyncWriteOpTrait`) to `AsyncTDMCopyGlobalToLocalOp` and similar async-to-shared-memory ops.

2. In `Membar.cpp:update()`, after `isIntersected` but before `blockInfo->join(curBlockInfo)`:

```cpp
// Async writes aren't visible yet and cannot be made visible by a
// barrier — only by an explicit wait. Keep the write in curBlockInfo
// for hazard detection against prior state (WAR, WAW checked by
// isIntersected above), but don't propagate it into blockInfo where
// it would cause false positives against subsequent ops.
if (op->hasTrait<OpTrait::MemAsyncWriteOpTrait>())
    curBlockInfo.syncWriteSlices.clear();

blockInfo->join(curBlockInfo);
```

**Hazard analysis**:

The `isIntersected` check covers three hazards. The async write's `curBlockInfo.syncWriteSlices` is populated when `isIntersected` runs (before the clear), so all three checks fire correctly for the async write as a *current* op:

| Hazard | Check | Detected? | Correct? |
|--------|-------|-----------|----------|
| **WAR** — async write vs prior read | `curBlockInfo.syncWriteSlices` vs `blockInfo.syncReadSlices` | Yes (before clear) | Yes — prior reads must complete before DMA overwrites the location |
| **WAW** — async write vs prior sync write | `curBlockInfo.syncWriteSlices` vs `blockInfo.syncWriteSlices` | Yes (before clear) | Yes — prior sync write must be visible before DMA starts |
| **RAW** — async write as prior, subsequent read | `blockInfo.syncWriteSlices` (would contain async write) vs future `curBlockInfo.syncReadSlices` | No (cleared) | Correct — DMA write is not visible yet; barrier can't help |
| **WAW** — async write as prior, subsequent write | `blockInfo.syncWriteSlices` (would contain async write) vs future `curBlockInfo.syncWriteSlices` | No (cleared) | Correct — barrier can't order a thread write against an in-flight DMA write; `async_tdm_wait` is needed |

After the clear, the async write does not enter `blockInfo.syncWriteSlices`. Subsequent ops see no pending write from it. The eventual visibility of the async write is handled by `MemWaitOpTrait` on `async_tdm_wait`, which auto-inserts a barrier and syncs all state.

**Cross-iteration correctness**:

The `MemWaitOpTrait` auto-barrier after `async_tdm_wait` syncs all state, so the back-edge carries empty `blockInfo` to the next iteration. The WAR between a prior iteration's `local_load` and the current iteration's `async_tdm_copy` is already fenced by that auto-barrier.

If the auto-barrier were absent (e.g., no wait in the loop), the `local_load` reads would accumulate in `blockInfo.syncReadSlices`, flow via the back-edge, and `isIntersected` would correctly detect the WAR when the next `async_tdm_copy` fires — auto-inserting a barrier exactly where needed.

### Long-term: Proper async dependency tracking in BlockInfo

**Concept**: Extend `BlockInfo` with a third container for in-flight async writes, and model write visibility transitions explicitly.

```
BlockInfo:
    syncWriteSlices   — committed writes (trigger RAW and WAW with subsequent ops)
    syncReadSlices    — reads (trigger WAR with subsequent writes)
    asyncWriteSlices  — in-flight writes (trigger WAR and WAW as current op only)
```

**Rules**:

| Current op      | Prior state                    | Hazard | Barrier? |
|-----------------|--------------------------------|--------|----------|
| Read            | `syncWriteSlices` (committed)  | RAW    | Yes      |
| Read            | `asyncWriteSlices` (in-flight) | —      | No       |
| Write (sync)    | `syncReadSlices`               | WAR    | Yes      |
| Write (sync)    | `syncWriteSlices` (committed)  | WAW    | Yes      |
| Write (sync)    | `asyncWriteSlices` (in-flight) | —      | No (barrier can't help) |
| Write (async)   | `syncReadSlices`               | WAR    | Yes      |
| Write (async)   | `syncWriteSlices` (committed)  | WAW    | Yes      |
| Write (async)   | `asyncWriteSlices` (in-flight) | —      | No (DMA ordering) |

**Transitions**:
- **Async copy** → adds entry to `asyncWriteSlices`
- **Wait op** → promotes entries from `asyncWriteSlices` to `syncWriteSlices` (writes are now architecturally visible)
- **Barrier** → clears all three containers

For count-based waits (`async_tdm_wait {num = N}`), promotion rule: all entries except the N most recent are promoted. This requires `asyncWriteSlices` to be ordered.

**Advantages over the near-term fix**:
- Correct by construction; no post-hoc clearing
- Eliminates the `MemWaitOpTrait` auto-barrier heuristic entirely; the wait promotes async writes and normal RAW/WAR/WAW logic decides if a barrier is needed
- Composes with multiple async streams and partial waits
- General: works for AMD TDM, NVIDIA TMA, `ttg.async_copy_global_to_local`, future async ops

---

## Relationship to Dynamic Offset Tracking

The false positive has two independent causes. The async write fix and better offset tracking are **orthogonal** improvements:

| Improvement              | What it solves                                          | Limitation                                         |
|--------------------------|---------------------------------------------------------|----------------------------------------------------|
| **Async write model**    | Eliminates false RAW for all async-to-shared-memory ops | Does not help synchronous writes to different slots |
| **Dynamic offset tracking** | Eliminates false conflicts for provably disjoint accesses | Cannot prove `prod%3 != cons%3` without symbolic reasoning about loop invariants |

For **this specific kernel**, the async fix is sufficient and more precise — it captures exactly why the barrier is unnecessary (the write hasn't happened yet) without requiring the compiler to reason about dynamic index relationships.

Better offset tracking would independently help a **different class** of false positives: synchronous `local_store` / `local_load` pairs accessing different slots of the same buffer. Today `AllocationSlice` handles static offsets from `MemDescSubsliceOp` but not dynamic indices from `MemDescIndexOp`. Extending it to handle constant-index `MemDescIndexOp` would be a small improvement, but the triple-buffer pattern uses dynamic `remsi` indices that require symbolic analysis beyond what `AllocationSlice::intersects` can practically do.

---

## Affected Ops

Async-to-shared-memory ops that would benefit from the `MemAsyncWriteOpTrait`:

| Op | Dialect | Currently has |
|----|---------|---------------|
| `AsyncTDMCopyGlobalToLocalOp` | TritonAMDGPU | `MemWrite<SharedMemory>` on `$result` |
| `AsyncTDMGatherOp` | TritonAMDGPU | `MemWrite<SharedMemory>` on `$dst` |
| `AsyncCopyGlobalToLocalOp` | TritonGPU | `MemWrite<SharedMemory>` on `$result` |
| `AsyncTMACopyGlobalToLocalOp` | TritonNvidiaGPU | `MemWrite<SharedMemory>` on `$result` |
| `AsyncTMAGatherOp` | TritonNvidiaGPU | `MemWrite<SharedMemory>` on `$result` |
