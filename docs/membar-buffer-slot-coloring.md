# Buffer Slot Coloring

## Concept

Each buffer slot access is tagged with a **buffer color** — a compile-time integer that identifies its logical buffer slot role. The AMD membar filter treats accesses with different colors as disjoint.

| | color=0 | color=1 | no color |
|---|---|---|---|
| **color=0** | may alias | **disjoint** | may alias |
| **color=1** | **disjoint** | may alias | may alias |
| **no color** | may alias | may alias | may alias |

- Different colors → provably disjoint → no barrier
- Same color or uncolored → conservative (normal hazard analysis)
- Uncolored ops alias everything → existing code is unaffected

## Where to Place the Color

Three placement options were evaluated against the AMD lowering chain, where `ConvertToBufferOps` replaces `ttg.async_copy_global_to_local` with a new `amdgpu.BufferLoadToLocalOp` (destroying the original op but leaving memdesc operands untouched):

| Placement | Survives dialect conversions? | Survives `scf-to-cf`? | Complexity |
|---|---|---|---|
| On memory ops (`local_load`, `async_copy`) | **NO** — op replaced, attr lost | N/A | Must forward attr in every conversion pattern |
| On `MemDescIndexOp` | **YES** — `MemDescIndexOp` is never replaced | Recomputed each iteration, no block arg issue | Minimal |
| On `MemDescType` | **YES** — type flows with value | **YES** — type is part of block arg | Type system change, heavy |

**`MemDescIndexOp` is the best placement.** Dialect conversions replace memory ops but leave memdesc values untouched. `MemDescIndexOp` is a Pure op that is never replaced or erased by any AMD pass — it's only created (by the pipeliner via `createSingleBufferView`). In the pipeliner's pattern, `MemDescIndexOp` is recomputed each iteration from the allocation and the phase index, so it stays in the loop body and is not loop-carried — no block argument propagation needed.

## IR Representation

The color is an attribute on `MemDescIndexOp`:

```mlir
%alloc = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #shared, #smem, mutable>

scf.for ... iter_args(%phase = %c0, %tok = %wait0) {
    %read  = ttg.memdesc_index %alloc[%phase]      {buffer_color = 0}
        : !ttg.memdesc<2x16x16xf16, ...> -> !ttg.memdesc<16x16xf16, ...>
    %next  = arith.xori %phase, %c1
    %write = ttg.memdesc_index %alloc[%next]        {buffer_color = 1}
        : !ttg.memdesc<2x16x16xf16, ...> -> !ttg.memdesc<16x16xf16, ...>

    // No barrier — AMD filter sees color=0 vs color=1 → disjoint
    %data = ttg.local_load %read ...
    %tok1 = amdg.async_tdm_copy_global_to_local %desc into %write ...

    %wait1 = amdg.async_tdm_wait %tok1
    scf.yield %next, %wait1
}
```

After `scf-to-cf`, the `MemDescIndexOp`s remain in the loop body with their `buffer_color` attributes intact. After `ConvertToBufferOps` replaces the async copy op, `%write` (the memdesc operand) still carries its color on the defining `MemDescIndexOp`.

## AMD Filter Integration

The AMD membar filter resolves the color by tracing each op's memdesc operand back to its defining `MemDescIndexOp` (via `getMemdescValue` or similar). The check is minimal:

```cpp
auto op1Color = getBufferColor(op1);
auto op2Color = getBufferColor(op2);
if (op1Color.has_value() && op2Color.has_value() &&
    *op1Color != *op2Color)
  return false;  // provably disjoint
```

## Gluon API (AMD-specific)

A single new API — `colored_memdesc_index` — is the only addition needed. Existing `ttgl.load` and `amd.async_copy` remain unchanged:

```python
alloc = ttgl.alloc_shared((2, BLOCK_M, BLOCK_K), dtype=tl.float16, layout=shared_layout)

# Prologue
write_slot = amd.colored_memdesc_index(alloc, 0, color=0)
tok = amd.async_copy(src_ptr, write_slot)
tok = amd.async_wait(tok)

phase = 0
for i in range(num_iters):
    read_slot  = amd.colored_memdesc_index(alloc, phase, color=0)
    write_slot = amd.colored_memdesc_index(alloc, 1 - phase, color=1)

    A = ttgl.load(read_slot, reg_layout)        # reads from color=0 slot
    tok = amd.async_copy(next_ptr, write_slot)   # writes to color=1 slot

    acc = tl.dot(A, B, acc)
    tok = amd.async_wait(tok)
    phase = 1 - phase
```

`amd.colored_memdesc_index` lowers to a standard `ttg.memdesc_index` with a `buffer_color` attribute. The color lives on the memdesc, not on the memory ops — so it naturally survives all downstream op conversions.

## Pipeliner Integration

The current AMD pipeliner creates a **single** `MemDescIndexOp` shared by both the async copy (producer) and `local_load` (consumer). When the `PipelineExpander` places them in different stages, the memdesc result crosses stages and becomes a **loop-carried block argument**, which loses all op attributes — including `buffer_color`.

To support coloring, the AMD pipeliner must be adjusted to create **separate** `MemDescIndexOp`s for the producer and consumer, each recomputed from the loop-carried integer phase index within its own stage. The NVIDIA pipeliner already follows this pattern — it uses separate `insertIdx`/`extractIdx` counters and calls `createSingleBufferView` twice:

```cpp
// NVIDIA pipeliner (LowerLoops.cpp) — already creates separate MemDescIndexOps
Value view     = createSingleBufferView(builder, alloc, insertIdx);  // producer stage
auto viewLoad  = createSingleBufferView(builder, alloc, extractIdx); // consumer stage
```

The AMD pipeliner currently uses a single `extractIdx` for both:

```cpp
// AMD pipeliner (LowerLoops.cpp) — shares one MemDescIndexOp
auto viewLoad = createSingleBufferView(builder, alloc, extractIdx);
auto copyOp = AsyncTDMCopyGlobalToLocalOp::create(..., viewLoad, ...);  // producer
auto maybeSharedLoad = replaceUsesWithLocalLoad(..., viewLoad, ...);     // consumer
```

The required change: split into two `createSingleBufferView` calls with separate indices, each stamped with a different color:
- Consumer-stage `MemDescIndexOp` → `buffer_color = 0`
- Producer-stage `MemDescIndexOp` → `buffer_color = 1`

Since each `MemDescIndexOp` is freshly computed within its own stage (from the loop-carried integer index), it is never itself loop-carried — the attribute survives pipelining. This generalizes to N-way buffering (triple buffering uses 3 colors).

## Advantages

- **Decouples pattern recognition from decision making** — the producer (pipeliner or Gluon user) declares disjointness at `MemDescIndexOp` creation time; the filter just compares integers. New index patterns or pipeliner strategies require no changes to the filter. With symbolic index analysis, every new index idiom (e.g., `arith.andi` for power-of-2 buffers, multi-level indexing) requires extending the pattern matcher.
- **Gluon-friendly** — Gluon developers are free to compute buffer indices however they want. Symbolic analysis implicitly constrains users to write recognized patterns (`remsi`, `select/cmpi`); coloring imposes no such constraint.
- **Works with separate counters** — producer and consumer can use different index computations (different SSA bases). Coloring is orthogonal to how the index is computed.
- **Explicit contract** — both sides of the hazard declare their color, making the non-aliasing assertion visible in the IR.
- **Safe fallback** — if any transform drops the attribute, the access becomes uncolored and the filter falls back to conservative behavior, letting membar insert the barrier (performance regression, not miscompile).

## Limitations

- **AMD-specific** — lives in the AMD membar filter, not in core membar. Does not benefit other backends.
- **Attribute fragility** — relies on `MemDescIndexOp` attributes surviving all passes between the pipeliner and membar. In practice, no AMD pass replaces `MemDescIndexOp`, but the dependency exists.
- **Correctness depends on correct assignment** — if a user or the pipeliner assigns the same color to accesses on the same buffer slot, the barrier is suppressed on a real hazard.
- **Requires new API surface** — Gluon needs `colored_memdesc_index`; the pipeliner needs explicit color stamping logic.
