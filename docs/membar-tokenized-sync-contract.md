# Tokenized Sync Contract for Membar

## Motivation

Membar currently treats every synchronous shared-memory read as a pending
dependency against every later shared-memory write. For a pipelined kernel
where a `ttg.local_load` is followed by an `async_copy` writing into shared
memory, this produces a CTA-wide barrier on the WAR edge:

```mlir
%val = ttg.local_load %read_view
%tok = ttg.async_copy_global_to_local %src into %write_view
//  ^ membar inserts a CTA barrier before this op
```

The barrier is unnecessary when the IR already carries an explicit ordering
protocol that guarantees the read is complete before the async write
overwrites the source.

The intended contract is:

> If a sync op returns a transfer token, membar stops protecting that sync op
> against later async ops. The IR producer is responsible for inserting the
> proper token wait before any async op that could conflict.

This moves sync-to-async ordering from implicit membar alias analysis to an
explicit IR protocol expressed by a new SSA token type and a new wait op.

## The Contract

The contract is deliberately narrow.

1. It only applies to specific tokenized sync ops with well-defined semantics,
   initially `ttg.local_load` extended to optionally produce
   `!ttg.local_token`.
2. It only suppresses sync-to-async dependencies (a tokenized sync read
   followed by a later async write into shared memory).
3. It does not suppress sync-to-sync, async-to-sync, or async-to-async
   dependencies.
4. The token's only legal uses are `ttg.sync_transfer_wait` consumers, enforced
   by an MLIR op verifier. The producer of the IR places the wait before any
   aliasing async write.

A tokenized sync read is still a real shared-memory read. It is only excluded
from the subset of pending-read state that is used to block later async writes.

## Proposed IR Additions

This proposal adds two new IR constructs and a result on an existing op.

### `!ttg.local_token` (new SSA type)

A token type representing an in-flight synchronous shared-memory read. It is
distinct from `!ttg.async_token`:

| Token | Producer | Meaning |
|---|---|---|
| `!ttg.async_token` | `ttg.async_copy_global_to_local`, `async_wait`, ... | An async DMA is in flight; consumed by `async_wait`. |
| `!ttg.local_token` (new) | Tokenized `ttg.local_load` | A synchronous shared-memory read is in progress; consumed by `ttg.sync_transfer_wait`. |

A separate type prevents accidental cross-use. `async_wait` does not consume
`local_token`, and `sync_transfer_wait` does not consume `async_token`.

### `ttg.sync_transfer_wait` (new op)

```mlir
ttg.sync_transfer_wait %tok : !ttg.local_token
```

A no-op-lowering marker that consumes one or more `local_token` values. Its
job is to be visible to membar as the point where a tokenized read is
considered drained.

Lowering: emits no machine instruction. The wait is a compile-time IR marker;
the actual hardware ordering is provided by ordinary instruction order on the
warps that issued the load.

Membar effect: when membar processes this op, it removes the tokenized read
slice from async-facing pending-read state.

### `ttg.local_load` result token

`ttg.local_load` is extended with an optional result token. The new shape is:

```mlir
// Existing: result is just the loaded tensor.
%val = ttg.local_load %src : ... -> tensor<...>

// New: result is also an !ttg.local_token.
%val, %tok = ttg.local_load %src : ... -> tensor<...>, !ttg.local_token
```

The token result is opt-in. Only producers that follow the contract emit it.
Existing users of `ttg.local_load` are unchanged.

## Why a Result Token Is Distinct From The Existing Input Token

`TTG_LocalLoadOp` already declares an optional input of `TTG_AsyncToken`,
used by the prefetch and pipeliner paths to thread an `async_wait` token into
the load (so the load is ordered after a specific async producer). The two
token slots have opposite roles:

```mlir
//                consumer-side input          producer-side result
//                (existing)                   (new)
%val_a = ttg.local_load %src token %wait                                 // existing
%val_b, %tok = ttg.local_load %src                                        // new
```

| Slot | Direction | Type | Role |
|---|---|---|---|
| Input `token` (existing) | Operand | `!ttg.async_token` | "Wait for the async producer before this load runs." |
| Result token (new) | Result | `!ttg.local_token` | "Downstream consumers must `sync_transfer_wait` before any aliasing async write." |

The two are independent. A tokenized `local_load` may use both: an input
`async_token` to order against a prior `async_wait`, and a result `local_token`
to order against a later async write.

## MLIR Op Verifier

The contract is enforced as a local MLIR op verifier. Two rules are sufficient.

Rule 1: `local_token` only flows into `sync_transfer_wait`.
Verifier on the producing op (tokenized `ttg.local_load`):

```text
For each result of type !ttg.local_token:
  every use must be a ttg.sync_transfer_wait operand.
```

This guarantees that every `local_token` value is consumed by a
`sync_transfer_wait` somewhere in the function. Dropping the wait is a verifier
error, not a silent miscompile.

Rule 2: `sync_transfer_wait` operands are well-typed.
Verifier on `ttg.sync_transfer_wait`:

```text
Each operand must be of type !ttg.local_token.
Each operand must be defined by a tokenized read op
  (currently only ttg.local_load with a result token).
```

Together, the two rules pin down the producer/consumer pairing of the token
flow.

What the verifier does not check: whether `sync_transfer_wait` is placed
before the conflicting async write along every CFG path. That is a CFG-level
property and is the producer's responsibility. The verifier ensures the wait
exists in the function; placement is the contract that membar relies on. This
split is intentional — it keeps the verifier local while making "the wait is
missing" structurally impossible.

## Membar Model

Pending dependencies split by the kind of future operation they protect against:

```text
sync-facing dependencies:
  include ordinary reads and tokenized sync reads
async-facing dependencies:
  include ordinary reads
  exclude tokenized sync reads
```

Consequences:

- A later sync write still sees the tokenized `local_load` as a dependency
  (sync-to-sync unchanged).
- A later async write does not see the tokenized `local_load` as a reason to
  insert a membar (sync-to-async suppressed by contract).
- `ttg.sync_transfer_wait` is a sync point that flushes the tokenized read
  out of any remaining async-facing state.
- Async-to-sync is unchanged; async writes still go through the existing
  wait/barrier path before a later sync read can consume their data.

This keeps membar out of the business of evaluating async/sync aliasing for
that one specific dependency direction.

## Why This Is Different From General Alias Suppression

This approach does not claim that the two shared-memory slices are disjoint.
They may alias. It claims that aliasing, if present, is ordered by an explicit
IR protocol outside membar's dependency inference.

- Buffer index analysis and buffer coloring suppress barriers by proving or
  declaring that two buffer slots are different.
- Warp-local analysis suppresses barriers by proving the access is
  warp-partitioned with no cross-warp dependency.
- The tokenized sync contract suppresses one specific ordering edge because
  another IR mechanism is responsible for that ordering.

This is not a replacement for disjointness analysis. It is a separate contract
for sync-to-async edges where the compiler has chosen explicit token ordering.

## Correctness Boundaries

### Sync-to-Async (affected)

```mlir
%val, %tok = ttg.local_load %read_view
ttg.sync_transfer_wait %tok
%next_tok = ttg.async_copy_global_to_local %src into %write_view
```

For a tokenized `local_load`, membar does not insert a barrier before the async
write solely to protect the prior read. The producer is responsible for placing
`sync_transfer_wait %tok` before the async write.

### Sync-to-Sync (unchanged)

```mlir
%val, %tok = ttg.local_load %read_view
ttg.local_store %new_value, %write_view
```

A later synchronous write still sees the prior tokenized read as a normal
pending dependency. Tokenization does not weaken sync-to-sync ordering.

### Async-to-Sync (unchanged)

```mlir
%tok = ttg.async_copy_global_to_local %src into %write_view
ttg.async_wait %tok
%val = ttg.local_load %read_view
```

The existing async wait/barrier model still owns visibility from async writes
to later sync reads.

### Async-to-Async (unchanged)

This document only describes sync ops that produce `!ttg.local_token`. Async-
to-async ordering is governed by `!ttg.async_token` and `async_wait`.

### Does `local_store` Need a Result Token?

Not for the initial `local_load` contract.

The motivating dependency is a WAR edge: `local_load` reads shared memory, then
a later async copy writes shared memory. The token on `local_load` says "do not
overwrite this shared-memory source until the load has been safely consumed."

Adding the same result token to `local_store` would not be symmetric.
`local_store` is a write, so later async ops create different hazards:

| Prior sync op | Later async op | Hazard | Needs `local_store` token? |
|---|---|---|---|
| `local_load` | async write to shared | WAR | Yes, this is the proposed contract |
| `local_store` | async read from shared | RAW | Only if the async op consumes the stored data and no existing fence/wait protocol covers it |
| `local_store` | async write to shared | WAW | Usually no; only matters if the first write's ordering is semantically observable |

The RAW case is the only plausible reason. For example, a local-to-global
async transfer reading data just staged by `local_store`:

```mlir
ttg.local_store %value, %shared
ttng.fence_async_shared
ttng.async_tma_copy_local_to_global %shared, %desc
```

That ordering is already expressed with target-specific fences such as
`fence_async_shared`. A `local_store` token would only be justified if we
choose to replace those fences with a common token wait.

The conservative design:

1. Add a result token only to the read op that needs the contract:
   `ttg.local_load`.
2. Do not add a symmetric token result to `ttg.local_store` unless a concrete
   async-read consumer path appears that existing fence/wait semantics do not
   express well.
3. If such a path appears, define a separate `local_store` token contract with
   store-visibility semantics, not by symmetry with `local_load`.

## Implementation Sketches

Two implementation paths are viable. They are not exclusive: (a) is a
lower-risk landing point, (b) is the longer-term form. Both rely on the same
IR additions and verifier.

### Sketch (a): AMD `MembarFilterFn` clause

The check lives entirely in the AMD backend filter, alongside
`filterAsyncLocalLoadsDependencies`, `filterLDSMemoryBarriersDependencies`, and
`filterWarpLocalAccesses`.

```cpp
bool filterTokenizedLocalLoadAgainstAsyncWrite(Operation *lhs, Operation *rhs,
                                               bool lhsIsRead, bool rhsIsRead) {
  // Direction: prior sync read vs later async write only.
  if (!lhsIsRead || rhsIsRead)
    return false;
  auto load = dyn_cast<ttg::LocalLoadOp>(lhs);
  if (!load || !load.getResultToken())
    return false;
  if (!rhs->hasTrait<OpTrait::MemAsyncWriteOpTrait>())
    return false;
  return true; // suppress the WAR barrier
}

bool membarFilter(Operation *op1, Operation *op2,
                  bool lhsIsRead, bool rhsIsRead, ...) {
  return filterAsyncLocalLoadsDependencies(op1, op2, allocation) ||
         filterLDSMemoryBarriersDependencies(op1, op2) ||
         filterWarpLocalAccesses(op1, op2) ||
         filterTokenizedLocalLoadAgainstAsyncWrite(op1, op2,
                                                   lhsIsRead, rhsIsRead);
}
```

The filter sees the LHS (tokenized `local_load`) and RHS (async write) at the
same `isIntersected` callback. Per-pair structural check; no `BlockInfo`
changes. The verifier guarantees the load's token is consumed by a
`sync_transfer_wait` somewhere; placement before the async write is the
producer's contract.

Pros: scoped to AMD, no core membar surface change, follows the existing
filter pattern.
Cons: AMD-specific. Other backends do not benefit. The filter cannot easily
distinguish "wait was placed before this async write" from "wait exists
somewhere in the function." The verifier and the producer contract together
cover that gap.

### Sketch (b): Common membar via async-facing read state

A second pending-read map in `BlockInfo` exposes the dependency-direction
split directly:

```cpp
struct BlockInfo {
  using SliceMapT = std::map<AllocationSlice, std::set<Operation *>>;

  SliceMapT syncReadSlices;        // visible to all later writes
  SliceMapT syncWriteSlices;
  SliceMapT asyncFacingReadSlices; // visible only to later async writes
};
```

Population rules in `MembarAnalysis::update`:

- Ordinary `local_load`: insert into `syncReadSlices` and
  `asyncFacingReadSlices`.
- Tokenized `local_load`: insert into `syncReadSlices` only.
- `sync_transfer_wait`: erase the corresponding slice entries from
  `asyncFacingReadSlices` (or simply clear the map; the verifier guarantees
  every token reaches a wait).

`isIntersected` against a later op:

- Sync write: check `syncReadSlices` for WAR (unchanged).
- Async write: check `asyncFacingReadSlices` for WAR.
- WAW and RAW unchanged.

`join`, `operator==`, `sync()`, and `translateBlockInfoToCallsite` mirror the
existing maps. `sync()` clears all three (a CTA barrier flushes everything,
including async-facing reads).

Pros: all backends benefit. The dependency-direction split is explicit in the
state, not in a per-pair filter. Cleaner semantics for future extensions (for
example, tokenized variants of other sync ops).
Cons: larger surface change — touches `BlockInfo`, `update`, fixed-point join
logic, and the call-site translator. Higher initial risk.

### Recommended order

Land the IR additions and verifier first. Wire (a) on AMD to validate the
end-to-end contract on real kernels. Move to (b) once the contract is stable
and another backend wants the same suppression.

## Orthogonality With `MemWaitOpTrait` Handler

Membar today inserts an unconditional CTA barrier after every op with
`MemWaitOpTrait` (`async_wait`, `async_tdm_wait`, `async_tma_store_wait`) and
calls `sync()`. The tokenized contract is orthogonal to this handler:

- If a `MemWaitOpTrait` op sits between the tokenized `local_load` and the
  later async write, the unconditional barrier already provides ordering and
  `sync()` clears every pending slice, including the tokenized read. The
  later `sync_transfer_wait` is then a no-op as far as membar state is
  concerned, because there is nothing left to flush.
- If no `MemWaitOpTrait` op sits between them, the tokenized contract is
  responsible for the suppression and `sync_transfer_wait` is the only
  ordering edge that matters.

The two mechanisms cover non-overlapping cases. The proposed contract does
not require changing the `MemWaitOpTrait` handler. The async write tracking
proposal in `membar-async-write-tracking.md` is a separate, independently
motivated change.

## Difference From PR #9418

PR #9418 added an AMD-specific annotation pass that stamped `local_load` ops
whose token chains back to an `async_wait`, and a backend filter suppressed
barriers for those annotated ops. It was rejected because correctness relied
on an implicit assumption about how the pipeliner generates IR (always
multi-buffered, `numBuffers >= 2`), which violated pass independence.

This proposal differs structurally:

| | PR #9418 | Tokenized Sync Contract |
|---|---|---|
| Where the contract lives | Hidden in pass-pipeline assumption (annotation pass + filter) | Visible in IR: `!ttg.local_token` SSA value, `ttg.sync_transfer_wait` consumer op |
| What guarantees correctness | "The pipeliner always emits multi-buffered IR" | MLIR op verifier ensures the token is consumed by a wait |
| Failure mode if a transform changes the IR | Silent miscompile (annotation lost or invalidated) | Verifier error or barrier reinserted |
| Reviewer-visible scope | Implicit | Local op signatures and verifier rules |

The reviewer concern that sank #9418 — that a barrier is suppressed based on
an unstated IR-generation invariant — does not apply here. The invariant is
expressed as IR.

## Summary

| Aspect | Tokenized Sync Contract |
|---|---|
| Mechanism | New `!ttg.local_token` result on `ttg.local_load`, consumed by new `ttg.sync_transfer_wait` |
| Distinct from existing | Existing input `!ttg.async_token` on `local_load` is unchanged; result token is new |
| Membar behavior | Tokenized sync reads are excluded from async-facing pending-read state; `sync_transfer_wait` is the flush point |
| Suppresses | Sync read before later async write, when the read is tokenized |
| Does not suppress | Sync-to-sync, async-to-sync, async-to-async |
| Correctness source | MLIR op verifier ensures the token reaches a `sync_transfer_wait`; producer places it before any aliasing async write |
| Implementation paths | (a) AMD `MembarFilterFn` clause; (b) common membar via `asyncFacingReadSlices` |
| Orthogonality | Independent of the `MemWaitOpTrait` unconditional barrier handler and of async write tracking |
| Difference from PR #9418 | Contract is explicit IR (token + wait + verifier), not an implicit pipeliner-generation assumption |
