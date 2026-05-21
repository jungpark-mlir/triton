# Tokenized Sync Contract for Membar

## Main Idea

Membar currently treats a synchronous shared-memory read as a pending WAR
hazard against later shared-memory writes. In pipelined code, that can force a
`local_barrier` before a later async write even when the compiler already knows
where the read lifetime ends.

This proposal makes that lifetime explicit:

```text
tokenized local_load -> !ttg.local_token -> ttg.local_read_ack -> async write phase
```

A tokenized `ttg.local_load` opts out of membar's inferred sync-to-async WAR
tracking. Instead, it creates a `!ttg.local_token` obligation. The producer must
acknowledge that obligation with `ttg.local_read_ack` before entering any async
write phase that may recycle the shared memory read by the load.

`ttg.local_read_ack` is not an alias proof between a specific `local_load` and a
specific async copy. It acknowledges the obligation created by the tokenized
load. If no previous `local_barrier` has already discharged that obligation,
membar inserts a `local_barrier` at the ack. If the obligation was already
discharged, the ack is a no-op.

## How It Works

Without this contract, membar sees this shape as a WAR hazard:

```mlir
%val = ttg.local_load %read_view
%tok = ttg.async_copy_global_to_local %src into %write_view
//  ^ membar inserts a local_barrier before this op
```

With this contract, the producer writes the phase boundary explicitly:

```mlir
%val, %ltok = ttg.local_load %read_view

// Work that uses %val.

ttg.local_read_ack %ltok
%atok = ttg.async_copy_global_to_local %src into %write_view
```

The state model is:

```text
after tokenized local_load:
  sync-facing reads:        read_view
  async-facing reads:       <none for this local_load>
  local-token obligations:  %ltok unresolved

after local_read_ack:
  sync-facing reads:        read_view
  local-token obligations:  %ltok discharged
  local_barrier inserted only if no earlier local_barrier discharged it

at async_copy:
  no inferred sync-to-async WAR against this local_load
  correctness is through the preceding local_read_ack
```

If the ack is placed after the async copy, the IR violates the token contract:
the async phase may recycle shared memory before the obligation is discharged.
Membar intentionally does not recover by rediscovering the old sync-to-async
WAR dependency, because avoiding that inference is the purpose of the contract.

## Why It Is Correct

Correctness comes from making the end of the local-read lifetime explicit in
IR. A tokenized `ttg.local_load` does not rely on membar to prove that a later
async write is safe by alias analysis. Instead, it creates this obligation:

```text
before an async-write phase may recycle this shared memory,
the !ttg.local_token must be acknowledged
```

`ttg.local_read_ack` discharges that exact obligation:

```text
local_read_ack(token):
  if a local_barrier since the tokenized load already discharged the obligation:
    no-op
  else:
    insert a local_barrier here
  mark the token obligation discharged
```

This keeps responsibilities clear:

- Membar no longer asks whether a later async write aliases the tokenized
  `local_load`.
- The producer identifies the block-level phase boundary where the local-read
  obligation must end.
- `local_read_ack` makes that boundary correct by inserting a `local_barrier`
  if needed.
- The producer owns performance by placing the ack where it is usually benign,
  for example near an existing `local_barrier`.

The tokenized load is still a real shared-memory read. It remains visible to
sync-facing dependency tracking, so sync-to-sync hazards are unchanged.

## Design

### IR Additions

This proposal adds:

1. A new `!ttg.local_token` type.
2. A new `ttg.local_read_ack` op.
3. A neutral `ttg.local_token_init` op for loop seeds.
4. An optional result token on `ttg.local_load`.

#### `!ttg.local_token`

`!ttg.local_token` is a TritonGPU block/CTA-scoped compiler token representing
a pending local-read obligation. It is not a runtime hardware token, and it is
not per-thread or per-element.

| Token | Producer | Meaning |
|---|---|---|
| `!ttg.async_token` | `ttg.async_copy_global_to_local`, `async_wait`, ... | Async DMA is in flight; consumed by `async_wait`. |
| `!ttg.local_token` | Tokenized `ttg.local_load` | Local-read obligation is pending; acknowledged by `ttg.local_read_ack`. |
| `!ttg.local_token` | `ttg.local_token_init` | Neutral loop seed with no local-read obligation. |

The separate type prevents accidental cross-use: `async_wait` does not consume
`!ttg.local_token`, and `local_read_ack` does not consume `!ttg.async_token`.

#### `ttg.local_token_init`

```mlir
%tok = ttg.local_token_init : !ttg.local_token
```

`ttg.local_token_init` creates a neutral local token with no associated
local-read obligation. It exists to seed loop-carried token variables without
using `None`, peeling only for token setup, or introducing a runtime branch
around `ttg.local_read_ack`.

The op lowers to no machine instruction and carries no shared-memory effect.

Acknowledging this token is always a no-op:

```text
local_read_ack(local_token_init):
  no local_barrier
  no obligation to discharge
```

The initializer does not suppress any membar dependency by itself. Only a
tokenized `ttg.local_load` creates a local-read obligation.

#### `ttg.local_read_ack`

```mlir
ttg.local_read_ack %tok : !ttg.local_token
```

`ttg.local_read_ack` consumes one or more `!ttg.local_token` values. For tokens
produced by tokenized reads, it acknowledges their local-read obligations. For
neutral `ttg.local_token_init` seeds, it simply consumes the token. The op
itself lowers to no machine instruction. During membar insertion, it may
materialize a `local_barrier` at that point if an acknowledged token has not
already been discharged.

The op carries no-operand shared-memory effects:

```text
MemoryEffects<[MemRead<SharedMemory>, MemWrite<SharedMemory>]>
```

#### Tokenized `ttg.local_load`

Existing form:

```mlir
%val = ttg.local_load %src : ... -> tensor<...>
```

New opt-in form:

```mlir
%val, %ltok = ttg.local_load %src : ... -> tensor<...>, !ttg.local_token
```

The result token is distinct from the existing optional input
`!ttg.async_token` on `ttg.local_load`:

| Slot | Direction | Type | Role |
|---|---|---|---|
| Input `token` | Operand | `!ttg.async_token` | Wait for an async producer before this load runs. |
| Result token | Result | `!ttg.local_token` | Acknowledge this local read before a later async phase may recycle its memory. |

A `ttg.local_load` may use both forms at once: an input async token orders the
load after an async producer; the result local token creates an obligation for a
later async phase.

### Block-Level Semantics

`!ttg.local_token` is block-scoped and matches membar's op-level shared-memory
slice abstraction. The token type itself is compile-time bookkeeping.

A `local_barrier` is Triton's local-memory fence followed by a CTA-scoped
execution barrier. Because `ttg.local_read_ack` may cause membar to insert one,
the ack op must only be placed where a `local_barrier` would be legal. It must
not be placed in divergent per-lane control. Producers should treat it as a
block-level phase-boundary op.

This proposal is intentionally CTA-scoped. Warp-local refinements can be layered
later, but the initial contract uses the synchronization granularity that membar
already uses for shared-memory hazards.

### Verifier Rules

The verifier enforces local structural rules. It does not try to prove the
producer made a profitable placement.

Rule 1: `!ttg.local_token` has restricted uses.

```text
For each tokenized local_load result of type !ttg.local_token:
  each use must be either:
    - a ttg.local_read_ack operand, or
    - an scf.yield operand that feeds the immediately-enclosing scf.for
      iter_arg accepted by Rule 3.

For each ttg.local_token_init result:
  each use must be either:
    - a ttg.local_read_ack operand, or
    - the initial value of an scf.for iter_arg accepted by Rule 3.

For each block argument of type !ttg.local_token:
  each use must be a ttg.local_read_ack operand.

For each scf.for result of type !ttg.local_token:
  each use must be a ttg.local_read_ack operand.
```

Rule 2: `ttg.local_read_ack` operands are well-typed.

```text
Each operand must be of type !ttg.local_token.
```

Rule 2 intentionally does not require the operand to be defined directly by a
tokenized read. In the pipelined case, the ack consumes a loop-carried block
argument whose value came from the previous iteration's tokenized read. Rule 3
owns that origin and placement check.

Rule 3: the token may be acknowledged in the same region, or carried through at
most one `scf.for` iteration.

The operand of `local_read_ack` must satisfy one of three cases:

- Same-region: defined by a tokenized read or `ttg.local_token_init` whose
  immediate parent is the same as the ack's immediate parent.
- One-iteration carry: a block argument of the ack's immediately-enclosing
  `scf.for` body, where the matching yield operand is defined in the for body,
  not itself a block argument, and is produced by a tokenized read or
  `ttg.local_token_init`.
- Final loop result: an `scf.for` result defined in the same region as the ack,
  where the matching yield operand is defined in the for body, not itself a
  block argument, and is produced by a tokenized read or `ttg.local_token_init`.
  This permits an epilogue ack for the final token produced by the loop body.

The one-iteration limit is a compiler policy, not a hardware claim. A producer
could register-buffer local-load values across more iterations, but that grows
loop-carried state and register pressure and is not useful for the intended
pipeline shape.

If a producer writes a chain where a token reaches `local_read_ack` only after
multiple loop iterations, the IR is invalid under this contract. For example:

```mlir
%loop = scf.for ... iter_args(%older = %init0, %newer = %init1) {
  ttg.local_read_ack %older

  %val, %ltok = ttg.local_load %read_view
  scf.yield %newer, %ltok
}
```

This is a two-iteration carry. The ack operand `%older` is a loop block
argument whose matching yielded value is `%newer`, itself a loop block
argument, so Rule 3 rejects it. If this were allowed, membar would have to track
the dynamic age of local-token obligations through the loop fixed point. This
proposal intentionally avoids that model.

### Placement Guidance

Default placement:

```mlir
%val, %ltok = ttg.local_load %read_view
// use %val
ttg.local_read_ack %ltok
%atok = ttg.async_copy_global_to_local %src into %write_view
```

Pipelined placement:

```mlir
%init_ltok = ttg.local_token_init : !ttg.local_token
%final_ltok = scf.for ... iter_args(%carried_ltok = %init_ltok)
    -> (!ttg.local_token) {
  ttg.local_read_ack %carried_ltok
  %atok = ttg.async_copy_global_to_local %src into %write_view

  %val, %ltok = ttg.local_load %read_view
  scf.yield %ltok
}

// Discharge the token produced by the final loop iteration. If the final read
// does not need a token obligation, the producer should avoid tokenizing that
// tail load instead of leaving the token unacknowledged.
ttg.local_read_ack %final_ltok
```

In the pipelined shape, the ack at iteration `i + 1` discharges the obligation
created by the load in iteration `i`, then the async phase in iteration `i + 1`
can proceed without membar trying to infer a sync-to-async WAR dependency
against that earlier load.

This matches the common Gluon/TDM steady-state pipeline shape. A neutral token
initializer lets the frontend write a uniform loop while keeping the IR simple:
every `local_read_ack` is unconditional, and the first iteration acknowledges a
token with no obligation.

```python
# Proposed frontend spelling; exact API names can change.
prev_ltok = ttgl.init_local_token()

for phase in range(0, num_main_iters):
    ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * COPIES_PER_TILE)
    ttgl.local_read_ack(prev_ltok)

    payload, next_ltok = buffer.index(read_slot(phase)).load_with_token(
        layout=OPERAND_LAYOUT)
    ttgl.amd.gfx1250.tdm.async_load(
        desc, [0, producer_offset(phase)], buffer.index(write_slot(phase)))
    acc = do_compute(payload, acc)

    prev_ltok = next_ltok

ttgl.local_read_ack(prev_ltok)
```

The first `local_read_ack` consumes the neutral token and inserts no
`local_barrier`. After the first `load_with_token`, `prev_ltok` is the previous
iteration's local-read token. The ack is then before the async copy that may
recycle that previous read buffer, so the real token chain is one iteration long
and is covered by this contract. The final `local_read_ack` consumes the token
produced by the final loop iteration.

This should lower to an unconditional block-level ack, not to a runtime branch.
For example, the following shape is not the intended contract:

```mlir
scf.if %has_token {
  ttg.local_read_ack %tok
}
```

With `ttg.local_token_init`, no `%has_token` condition is needed. Since
`local_read_ack` may insert a `local_barrier`, it should appear
unconditionally in the block-level loop body.

If the final read has no later async-recycle phase to guard and an epilogue ack
would only add a barrier, the producer should avoid creating a token for that
tail load rather than producing an unacknowledged token.

Invalid placement:

```mlir
%val, %ltok = ttg.local_load %read_view
%atok = ttg.async_copy_global_to_local %src into %write_view
ttg.local_read_ack %ltok
```

The ack is too late. The async phase can recycle shared memory before the token
obligation is discharged. Membar intentionally does not recover by inserting the
old inferred WAR barrier.

## Implementation Sketches

### Sketch A: AMD-Scoped Implementation

The AMD implementation has two pieces:

1. The AMD membar filter suppresses the old inferred sync-to-async WAR edge for
   tokenized `local_load` ops.
2. AMD-owned handling for `ttg.local_read_ack` conditionally inserts a
   `local_barrier` if the acknowledged token obligation has not already been
   discharged.

Filter clause:

```cpp
bool filterTokenizedLocalLoadAgainstAsyncWrite(Operation *lhs, Operation *rhs,
                                               bool lhsIsRead, bool rhsIsRead) {
  if (!lhsIsRead || rhsIsRead)
    return false;
  auto load = dyn_cast<ttg::LocalLoadOp>(lhs);
  if (!load || !load.getResultToken())
    return false;
  if (!rhs->hasTrait<OpTrait::MemAsyncWriteOpTrait>())
    return false;
  return true; // sync-to-async ordering is owned by local_read_ack
}
```

Ack handling:

```text
on tokenized local_load:
  create local-token obligation in AMD-side token state

on local_token_init:
  create a discharged/no-obligation token state

on local_barrier / membar sync:
  mark outstanding local-token obligations discharged

on ttg.local_read_ack(%tok):
  if %tok obligation is unresolved:
    insert local_barrier at the ack
    mark outstanding local-token obligations discharged
  else:
    no-op
  mark %tok obligation discharged
```

Pros: scoped to AMD, follows the existing filter pattern, and avoids changing
core membar state.

Cons: AMD-specific; other backends do not benefit. The filter alone is not
enough; token-state tracking for `local_read_ack` is required.

### Sketch B: Common Membar Token Obligations

Common membar can model the contract directly in `BlockInfo`:

```cpp
struct BlockInfo {
  using SliceMapT = std::map<AllocationSlice, std::set<Operation *>>;

  SliceMapT syncReadSlices;        // visible to later sync writes
  SliceMapT syncWriteSlices;
  SliceMapT asyncReadSlices;       // ordinary reads only; excludes tokenized local_load
  DenseMap<Value, LocalTokenState> localTokenObligations;
};
```

Population rules:

- Ordinary `local_load`: insert into `syncReadSlices` and `asyncReadSlices`.
- Tokenized `local_load`: insert into `syncReadSlices` only, and create
  `localTokenObligations[token] = unresolved`.
- `local_token_init`: create `localTokenObligations[token] = discharged`, or
  otherwise mark the token as a no-obligation seed.
- `local_barrier` / `sync()`: mark outstanding token obligations discharged.
- `local_read_ack`: insert a `local_barrier` if any acknowledged token is
  unresolved, then mark acknowledged tokens discharged.

`isIntersected` behavior:

- Sync write: check `syncReadSlices` for WAR (unchanged).
- Async write: check `asyncReadSlices` for WAR. Tokenized local loads are not
  present there.
- WAW and RAW unchanged.

`join`, `operator==`, `sync()`, and `translateBlockInfoToCallsite` must account
for `localTokenObligations`. A conservative merge treats unresolved as winning
over discharged when paths join.

Pros: all backends benefit and the token obligation is represented directly in
membar state.

Cons: larger surface change touching `BlockInfo`, `update`, fixed-point join
logic, and call-site translation.

### Recommended Order

Land the IR additions and verifier first. Use Sketch A to validate the contract
on AMD. Move to Sketch B once the contract is stable or another backend needs
the same suppression.

## Supplementary Details

### Verifier Pseudocode

The verifier rules above can be implemented with a local structural check:

```text
For each local_read_ack operand:
  require type !ttg.local_token

  if defined by tokenized local_load or local_token_init:
    require same parent op as the ack

  else if it is a block argument:
    require the ack parent is the immediately-enclosing scf.for
    require the argument is not the induction variable
    require the matching yield value is defined in the loop body
    require the matching yield value is produced by tokenized local_load
      or local_token_init

  else if defined by scf.for:
    require the scf.for has the same parent op as the ack
    require the matching yield value is defined in the loop body
    require the matching yield value is produced by tokenized local_load
      or local_token_init

  else:
    reject
```

### Memory Effects On `local_read_ack`

`ttg.local_read_ack` carries no-operand shared-memory effects:

```text
MemoryEffects<[MemRead<SharedMemory>, MemWrite<SharedMemory>]>
```

This is intentionally fence-like. It preserves the op and prevents shared-memory
ops from being moved across it. It does not create a tracked memory slice in the
current membar implementation, which only records effects tied to a concrete
`Value` operand:

```text
for (auto effectInstance : effectInstances)
  if (auto value = effectInstance.getValue())
    ... insert into syncReadSlices / syncWriteSlices ...
```

### Is A Core Membar Trait Needed?

| Implementation | Core trait needed? | Why |
|---|---|---|
| AMD-scoped | No | AMD code can directly match `ttg.local_read_ack` and maintain AMD-side token state. |
| Common membar | Yes | Core membar needs a stable signal such as `LocalReadAckOpTrait` to dispatch token-obligation logic without dialect-specific `isa<LocalReadAckOp>`. |

### `local_store` Result Token

No `local_store` result token is needed for the initial contract.

The motivating hazard is a WAR edge: `local_load` reads shared memory, then a
later async copy may write/recycle that memory. `local_store` is a write, so
later async ops create different hazards:

| Prior sync op | Later async op | Hazard | Needs `local_store` token? |
|---|---|---|---|
| `local_load` | async write to shared | WAR | Yes, this proposal. |
| `local_store` | async read from shared | RAW | Only if existing fence/wait protocol cannot express the async read. |
| `local_store` | async write to shared | WAW | Usually no; only observable if the first write is consumed before overwrite. |

For local-to-global async transfers that read staged shared memory, ordering is
already expressed with target-specific fences such as `ttng.fence_async_shared`.
A future `local_store` token should be a separate store-visibility contract, not
an automatic symmetry with `local_load`.

### Orthogonality With `MemWaitOpTrait`

Membar today inserts an unconditional `local_barrier` after every
`MemWaitOpTrait` op (`async_wait`, `async_tdm_wait`, `async_tma_store_wait`) and
calls `sync()`.

If such a wait sits between a tokenized `local_load` and the later async write,
that unconditional `local_barrier` already discharges the local-token
obligation. A later `local_read_ack` is benign: it consumes the token and
inserts no barrier.

This is the common Gluon/TDM case: `tdm.async_wait` is a `MemWaitOpTrait` op, so
its `local_barrier` usually discharges the previous iteration's local-read
token before the following `local_read_ack`.

If no `MemWaitOpTrait` op sits between them, `local_read_ack` is the explicit
place where a `local_barrier` may be inserted if needed. This proposal does not
require changing the `MemWaitOpTrait` handler. Async write tracking remains a
separate proposal.

### Difference From PR #9418

PR #9418 used an AMD-specific annotation pass to stamp `local_load` ops whose
token chains reached an `async_wait`, then suppressed barriers in the backend
filter. It was rejected because correctness relied on an implicit assumption
about how the pipeliner generated IR.

This proposal makes the contract explicit in IR:

| | PR #9418 | Tokenized Sync Contract |
|---|---|---|
| Contract location | Hidden in pass-pipeline assumption | `!ttg.local_token`, `ttg.local_token_init`, and `ttg.local_read_ack` |
| Correctness source | Assumption about pipeliner shape | Verifier plus explicit token obligation |
| Failure mode | Silent invalid annotation/assumption | Verifier error or `local_read_ack` inserts `local_barrier` |
| Reviewer-visible scope | Implicit | Op signatures and verifier rules |

### Summary

| Aspect | Tokenized Sync Contract |
|---|---|
| Mechanism | Tokenized `ttg.local_load` returns `!ttg.local_token`; `ttg.local_token_init` seeds loop-carried tokens; `ttg.local_read_ack` consumes them. |
| Suppresses | Inferred sync-to-async WAR dependency from tokenized local loads to later async writes. |
| Correctness source | Explicit local-token obligation discharged by `local_read_ack`; ack inserts `local_barrier` if needed. |
| Does not suppress | Sync-to-sync, async-to-sync, async-to-async. |
| Placement | Producer places ack before async phase that may recycle the loaded shared memory. |
| Scope | CTA/block-level; `local_read_ack` is legal only where `local_barrier` is legal. |
| Performance | Producer places ack where it is usually benign, often near an existing `local_barrier`. |
