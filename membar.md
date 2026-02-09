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

**Concept**: Keep `MemWrite<SharedMemory>` on the async op (so WAR is still detected), but prevent the write from propagating into `blockInfo.syncWriteSlices` (so it doesn't trigger false RAW with subsequent reads).

**Implementation**:

1. Add a trait (e.g., `MemAsyncWriteOpTrait`) to `AsyncTDMCopyGlobalToLocalOp` and similar async-to-shared-memory ops.

2. In `Membar.cpp:update()`, after `isIntersected` but before `blockInfo->join(curBlockInfo)`:

```cpp
// Async writes aren't visible yet — keep for WAR detection (already
// checked by isIntersected above) but don't propagate into blockInfo
// where they would cause false RAW with subsequent reads.
if (op->hasTrait<OpTrait::MemAsyncWriteOpTrait>())
    curBlockInfo.syncWriteSlices.clear();

blockInfo->join(curBlockInfo);
```

**Why this is correct**:

- **WAR preserved**: `isIntersected(blockInfo, curBlockInfo)` runs *before* the clear. If a prior `local_load` read from the same buffer, the async write's entry in `curBlockInfo.syncWriteSlices` intersects with `blockInfo.syncReadSlices` → WAR detected → barrier inserted. This is correct: we must ensure prior reads complete before overwriting.

- **False RAW eliminated**: After clearing, the async write doesn't enter `blockInfo.syncWriteSlices`. When a subsequent `local_load` is processed, there's no pending write to intersect with → no barrier.

- **Eventual RAW handled by MemWaitOpTrait**: The `async_tdm_wait` op has `MemWaitOpTrait`, which causes the membar analysis to auto-insert a `ttg.barrier local` after it and sync all state. This ensures the async write is visible before anything downstream of the wait reads the data.

- **Cross-iteration WAR is correctly handled**: The `MemWaitOpTrait` auto-barrier after `async_tdm_wait` syncs all state, so the back-edge carries empty `blockInfo` to the next iteration. The WAR between a prior iteration's `local_load` and the current iteration's `async_tdm_copy` is already fenced by that auto-barrier. If the auto-barrier were absent (e.g., no wait in the loop), the `local_load` reads would accumulate in `blockInfo.syncReadSlices`, flow via the back-edge, and `isIntersected` would correctly detect the WAR when the next `async_tdm_copy` fires — auto-inserting a barrier exactly where needed.

### Long-term: Proper async dependency tracking in BlockInfo

**Concept**: Extend `BlockInfo` with a third container for in-flight async writes, and model write visibility transitions explicitly.

```
BlockInfo:
    syncWriteSlices   — committed writes (trigger RAW with subsequent reads)
    syncReadSlices    — reads (trigger WAR with subsequent writes)
    asyncWriteSlices  — in-flight writes (trigger WAR only, NOT RAW)
```

**Rules**:

| Current op      | Prior state                    | Hazard | Barrier? |
|-----------------|--------------------------------|--------|----------|
| Read            | `syncWriteSlices` (committed)  | RAW    | Yes      |
| Read            | `asyncWriteSlices` (in-flight) | —      | No       |
| Write (sync)    | `syncReadSlices`               | WAR    | Yes      |
| Write (async)   | `syncReadSlices`               | WAR    | Yes      |

**Transitions**:
- **Async copy** → adds entry to `asyncWriteSlices`
- **Wait op** → promotes entries from `asyncWriteSlices` to `syncWriteSlices` (writes are now architecturally visible)
- **Barrier** → clears all three containers

For count-based waits (`async_tdm_wait {num = N}`), promotion rule: all entries except the N most recent are promoted. This requires `asyncWriteSlices` to be ordered.

**Advantages over the near-term fix**:
- Correct by construction; no post-hoc clearing
- Eliminates the `MemWaitOpTrait` auto-barrier heuristic entirely; the wait promotes async writes and normal RAW/WAR logic decides if a barrier is needed
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
