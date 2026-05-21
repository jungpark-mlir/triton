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
tracking. Instead, it creates a pending local-read barrier requirement. The
producer must clear that requirement with `ttg.local_read_ack` before entering
any async write phase that may recycle the shared memory read by the load.

`ttg.local_read_ack` is not an alias proof between a specific `local_load` and a
specific async copy. It clears the barrier requirement represented by the token.
If no previous `local_barrier` has already cleared that requirement, token-aware
membar handling inserts a `local_barrier` at the ack. If the requirement was
already cleared, the ack is a no-op.

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
  local-token requirement:  %ltok pending

after local_read_ack:
  sync-facing reads:        read_view
  local-token requirement:  %ltok cleared
  local_barrier inserted only if no earlier local_barrier cleared it

at async_copy:
  no inferred sync-to-async WAR against this local_load
  correctness is through the preceding local_read_ack
```

If the ack is placed after the async copy, the IR violates the token contract:
the async phase may recycle shared memory before the barrier requirement is
cleared.
Membar intentionally does not recover by rediscovering the old sync-to-async
WAR dependency, because avoiding that inference is the purpose of the contract.

## Why It Is Correct

Correctness comes from making the end of the local-read lifetime explicit in
IR. A tokenized `ttg.local_load` does not rely on membar to prove that a later
async write is safe by alias analysis. Instead, it creates this barrier
requirement:

```text
before an async-write phase may recycle this shared memory,
the !ttg.local_token's barrier requirement must be cleared
```

`ttg.local_read_ack` clears that exact requirement:

```text
local_read_ack(token):
  if a local_barrier since the tokenized load already cleared the requirement:
    no-op
  else:
    insert a local_barrier here
  mark the token requirement cleared
```

This keeps responsibilities clear:

- Membar no longer asks whether a later async write aliases the tokenized
  `local_load`.
- The producer identifies the block-level phase boundary where the local-read
  barrier requirement must be cleared.
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
a local-read barrier requirement. It is not a runtime hardware token, and it is
not per-thread or per-element.

| Token | Producer | Meaning |
|---|---|---|
| `!ttg.async_token` | `ttg.async_copy_global_to_local`, `async_wait`, ... | Async DMA is in flight; consumed by `async_wait`. |
| `!ttg.local_token` | Tokenized `ttg.local_load` | Local-read barrier requirement is pending; cleared by `ttg.local_read_ack`. |
| `!ttg.local_token` | `ttg.local_token_init` | Neutral loop seed with no local-read barrier requirement. |

The separate type prevents accidental cross-use: `async_wait` does not consume
`!ttg.local_token`, and `local_read_ack` does not consume `!ttg.async_token`.

#### `ttg.local_token_init`

```mlir
%tok = ttg.local_token_init : !ttg.local_token
```

`ttg.local_token_init` creates a neutral local token with no associated
local-read barrier requirement. It exists to seed loop-carried token variables without
using `None`, peeling only for token setup, or introducing a runtime branch
around `ttg.local_read_ack`.

The op lowers to no machine instruction and carries no shared-memory effect.

Acknowledging this token is always a no-op:

```text
local_read_ack(local_token_init):
  no local_barrier
  no requirement to clear
```

The initializer does not suppress any membar dependency by itself. Only a
tokenized `ttg.local_load` creates a local-read barrier requirement.

#### `ttg.local_read_ack`

```mlir
ttg.local_read_ack %tok : !ttg.local_token
```

`ttg.local_read_ack` consumes one or more `!ttg.local_token` values. For tokens
produced by tokenized reads, it clears their local-read barrier requirements. For
neutral `ttg.local_token_init` seeds, it simply consumes the token. The op
itself lowers to no machine instruction. During membar insertion, it may
materialize a `local_barrier` at that point if an acknowledged token has not
already been cleared.

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
| Result token | Result | `!ttg.local_token` | Clear this local read's barrier requirement before a later async phase may recycle its memory. |

A `ttg.local_load` may use both forms at once: an input async token orders the
load after an async producer; the result local token creates a barrier
requirement for a later async phase.

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

The verifier should stay intentionally simple. It should enforce type
correctness, not try to prove token origin or placement:

```text
ttg.local_read_ack:
  each operand must have type !ttg.local_token

ttg.local_token_init:
  result type is !ttg.local_token

tokenized ttg.local_load:
  optional result type is !ttg.local_token
```

The verifier does not need to reject tokens from `scf.for`, `scf.if`,
`scf.execute_region`, function arguments, or other control-flow plumbing. If the
AMD token-aware filter cannot prove a token is a known neutral seed or a known
cleared tokenized-read requirement, it treats the token as an unknown pending
barrier requirement. A prior `local_barrier` in the dataflow state can still
clear that unknown requirement. Otherwise, `local_read_ack` inserts a
`local_barrier`.

This keeps arbitrary token sources conservative: they may lose the optimization
by forcing a barrier at the ack, but they do not become a correctness hole.

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

// Clear the token produced by the final loop iteration. If the final read does
// not need a barrier requirement, the producer should avoid tokenizing that
// tail load instead of leaving the token unused.
ttg.local_read_ack %final_ltok
```

In the pipelined shape, the ack at iteration `i + 1` clears the barrier
requirement created by the load in iteration `i`, then the async phase in
iteration `i + 1` can proceed without membar trying to infer a sync-to-async WAR
dependency against that earlier load.

This matches the common Gluon/TDM steady-state pipeline shape. A neutral token
initializer lets the frontend write a uniform loop while keeping the IR simple:
every `local_read_ack` is unconditional, and the first iteration clears a token
with no barrier requirement.

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
tail load rather than producing an unused token.

Invalid placement:

```mlir
%val, %ltok = ttg.local_load %read_view
%atok = ttg.async_copy_global_to_local %src into %write_view
ttg.local_read_ack %ltok
```

The ack is too late. The async phase can recycle shared memory before the token
barrier requirement is cleared. The token-aware AMD filter intentionally does
not recover by inserting the old inferred WAR barrier, because the tokenized
load asked that filter to suppress that edge.

## Implementation Sketches

### Sketch A: AMD-Scoped Implementation

The AMD implementation is owned by the AMD membar filter path. Without that
path, core membar ignores the local-token contract: tokenized `local_load`
remains a normal shared-memory read, core membar resolves sync-to-async
dependencies as it does today, and `ttg.local_read_ack` lowers away as a no-op.
In that mode, adding tokens changes no membar behavior.

With the AMD token-aware filter enabled, the implementation has two pieces:

1. The AMD membar filter suppresses the old inferred sync-to-async WAR edge for
   tokenized `local_load` ops.
2. AMD-owned handling for `ttg.local_read_ack` conditionally inserts a
   `local_barrier` if the acknowledged barrier requirement has not already been
   cleared.

Filter clause:

```cpp
bool filterTokenizedLocalLoadAgainstAsyncWrite(Operation *lhs, Operation *rhs,
                                               bool lhsIsRead, bool rhsIsRead) {
  if (!lhsIsRead || rhsIsRead)
    return false;
  auto load = dyn_cast<ttg::LocalLoadOp>(lhs);
  if (!load || !load.getResultToken())
    return false;
  if (!isa<ttg::AsyncCopyGlobalToLocalOp,
           amdgpu::AsyncTDMCopyGlobalToLocalOp,
           amdgpu::BufferLoadToLocalOp>(rhs))
    return false;
  return true; // sync-to-async ordering is owned by local_read_ack
}
```

Ack handling:

```text
on tokenized local_load:
  create a pending local-read barrier requirement in AMD-side token state

on local_token_init:
  create a cleared/no-requirement token state

on local_barrier / membar sync:
  mark outstanding local-token requirements cleared

on ttg.local_read_ack(%tok):
  if %tok is unknown and no prior local_barrier covers it:
    insert local_barrier at the ack
    mark outstanding local-token requirements cleared
  else if %tok requirement is pending:
    insert local_barrier at the ack
    mark outstanding local-token requirements cleared
  else:
    no-op
  mark %tok requirement cleared
```

Pros: scoped to AMD, follows the existing filter pattern, and avoids changing
core membar state.

Cons: AMD-specific; other backends do not benefit. The filter alone is not
enough; token-state tracking for `local_read_ack` is required.

### Optional Future: Common Membar Token State

This is not part of the initial AMD-scoped plan. If another backend needs the
same optimization, common membar could model the contract directly in
`BlockInfo`:

```cpp
struct BlockInfo {
  using SliceMapT = std::map<AllocationSlice, std::set<Operation *>>;

  SliceMapT syncReadSlices;        // visible to later sync writes
  SliceMapT syncWriteSlices;
  SliceMapT asyncReadSlices;       // ordinary reads only; excludes tokenized local_load
  DenseMap<Value, LocalTokenState> localTokenRequirements;
};
```

Population rules:

- Ordinary `local_load`: insert into `syncReadSlices` and `asyncReadSlices`.
- Tokenized `local_load`: insert into `syncReadSlices` only, and create
  `localTokenRequirements[token] = pending`.
- `local_token_init`: create `localTokenRequirements[token] = cleared`, or
  otherwise mark the token as a no-requirement seed.
- Unknown token source: treat as pending unless the current dataflow state has a
  prior `local_barrier` that clears local-token requirements on all paths.
- `local_barrier` / `sync()`: mark outstanding token requirements cleared.
- `local_read_ack`: insert a `local_barrier` if any acknowledged token is
  pending or unknown, then mark acknowledged tokens cleared.

`isIntersected` behavior:

- Sync write: check `syncReadSlices` for WAR (unchanged).
- Async write: check `asyncReadSlices` for WAR. Tokenized local loads are not
  present there.
- WAW and RAW unchanged.

`join`, `operator==`, `sync()`, and `translateBlockInfoToCallsite` would need to
account for `localTokenRequirements`. A conservative merge treats pending as
winning over cleared when paths join.

Pros: all backends benefit and the token requirement is represented directly in
membar state.

Cons: larger surface change touching `BlockInfo`, `update`, fixed-point join
logic, and call-site translation.

### AMD Pipeliner Producer Plan

The first producer should be the AMD pipeliner path that creates the problematic
sync-read followed by async-write schedule. In particular, the Gluon/TDM
`kernelB` shape loads from LDS and then issues the next TDM async copy in the
same warp-pipeline stage:

```text
stage0:
  read phase:
    local_load buffer[phase % NUM_BUFFERS]
  write phase:
    async_tdm_copy_global_to_local buffer[(phase + NUM_BUFFERS - 1) % NUM_BUFFERS]

stage1:
  compute using the loaded values
```

That stage should use tokenized LDS loads and acknowledge the previous
iteration's token before issuing the next async write phase:

```python
prev_a_ltok = ttgl.init_local_token()
prev_b_ltok = ttgl.init_local_token()

for ...:
    with ttgl.amd.warp_pipeline_stage("stage0"):
        phase, a, b, next_a_ltok, next_b_ltok = lds_load_with_tokens(...)

        ttgl.local_read_ack(prev_a_ltok)
        ttgl.local_read_ack(prev_b_ltok)

        issue_loads(write_phase, ...)

        prev_a_ltok = next_a_ltok
        prev_b_ltok = next_b_ltok

ttgl.local_read_ack(prev_a_ltok)
ttgl.local_read_ack(prev_b_ltok)
```

The important placement rule is that `local_read_ack` sits after the LDS read
that produces the next token but before the async write phase that may recycle
the previous token's buffer. The ack clears the previous token's requirement,
not the requirement for the token just produced. The first ack consumes a
neutral init token, so the loop does not need a runtime branch. The epilogue ack
clears the final token, unless the pipeliner chooses to avoid tokenizing
tail-only loads with no later async recycle phase.

This producer change is intentionally separate from dynamic-index disjointness:
buffer slot proofs may still remove barriers in some cases, but the token/ack
contract is the explicit correctness mechanism for the sync-to-async phase
boundary.

### Recommended Order

Land in this order:

1. Add `!ttg.local_token`, tokenized `ttg.local_load`,
   `ttg.local_token_init`, `ttg.local_read_ack`, and simple type verifier
   coverage.
2. Implement AMD-scoped membar handling using Sketch A, including token-state
   tracking for `local_read_ack`.
3. Wire tokenized loads and acks into the AMD pipeliner producer, starting with
   the Gluon/TDM `kernelB` schedule.
4. Add lit coverage for parsing/verifier behavior, membar barrier placement,
   and AMD warp-pipeline IR shape.
5. Validate with runtime GEMM coverage: Triton GEMM and the existing Gluon f16
   warp-pipeline GEMM using `--kernelB`.

Move to common membar token state once the contract is stable or another backend
needs the same suppression.

### Validation Plan

Lit tests should cover both the IR contract and the AMD behavior:

- Verifier tests for well-typed and ill-typed `local_read_ack` operands.
- Membar tests showing tokenized `local_load` no longer creates an inferred
  sync-to-async WAR barrier before the async write, while `local_read_ack`
  inserts a `local_barrier` if no earlier barrier cleared the token requirement.
- AMD warp-pipeline tests showing the `kernelB`-style stage contains tokenized
  LDS loads, unconditional `local_read_ack` ops, and no extra false-positive
  barrier before the TDM async write.

Runtime validation should include:

```bash
lit -v test/Conversion/amd/amdgpu_membar.mlir
lit -v test/TritonGPU/amd/amd-convert-warp-pipeline.mlir

python third_party/amd/python/examples/gluon/f16_gemm_warp_pipeline_gfx1250.py --kernelB
```

Also run the existing Triton GEMM coverage that exercises the AMD pipeliner, so
the token path is checked in both ordinary Triton-generated pipelines and the
Gluon/TDM `kernelB` producer.

## Supplementary Details

### Verifier Pseudocode

The verifier rules above can be implemented with a type-only structural check:

```text
For each local_read_ack operand:
  require type !ttg.local_token

For local_token_init:
  require result type !ttg.local_token

For tokenized local_load:
  require optional token result type !ttg.local_token
```

The verifier intentionally does not inspect the defining op for a token passed
to `local_read_ack`.

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

### Core Membar Trait

No core membar trait is needed for the initial AMD-scoped implementation. The
AMD filter path can directly match `ttg.local_read_ack` and maintain its
filter-owned token state. If a future common implementation is needed, it can
add a trait or direct op handling then; this proposal does not require that as
part of the first implementation.

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
that unconditional `local_barrier` already clears the local-token barrier
requirement. A later `local_read_ack` is benign: it consumes the token and
inserts no barrier.

This is the common Gluon/TDM case: `tdm.async_wait` is a `MemWaitOpTrait` op, so
its `local_barrier` usually clears the previous iteration's local-read
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
| Correctness source | Assumption about pipeliner shape | Explicit barrier requirement carried by token |
| Failure mode | Silent invalid annotation/assumption | Type verifier error, conservative barrier for unknown token source, or `local_read_ack` inserts `local_barrier` |
| Reviewer-visible scope | Implicit | Op signatures and verifier rules |

### Summary

| Aspect | Tokenized Sync Contract |
|---|---|
| Mechanism | Tokenized `ttg.local_load` returns `!ttg.local_token`; `ttg.local_token_init` seeds loop-carried tokens; `ttg.local_read_ack` consumes them. |
| Suppresses | Inferred sync-to-async WAR dependency from tokenized local loads to later async writes. |
| Correctness source | Explicit local-token barrier requirement cleared by `local_read_ack`; ack inserts `local_barrier` if needed. |
| Does not suppress | Sync-to-sync, async-to-sync, async-to-async. |
| Placement | Producer places ack before async phase that may recycle the loaded shared memory. |
| Scope | CTA/block-level; `local_read_ack` is legal only where `local_barrier` is legal. |
| Performance | Producer places ack where it is usually benign, often near an existing `local_barrier`. |
