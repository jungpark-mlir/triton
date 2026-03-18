# Membar False Positive Suppression for Multi-Buffered Pipelines

## Problem

In pipelined kernels with multi-buffering, `local_load` and `async_copy` (or TDM copy/gather) access different buffer slots of the same shared memory allocation. The buffer slot is selected via `MemDescIndexOp` with a dynamic loop-carried index.

Membar analysis (`AllocationSlice::intersects`) can only resolve static offsets from `MemDescSubsliceOp`. Since `MemDescIndexOp` uses dynamic indices, membar cannot distinguish disjoint buffer slots and conservatively reports a hazard, inserting an unnecessary barrier.

This false barrier exists on both the Triton pipeliner path and the Gluon path when users write multi-buffered kernels by hand.

## Prior Attempt (PR #9418 — rejected)

An AMD-specific annotation pass stamps `local_load` ops whose token chains back to an `async_wait`, and a backend filter suppresses barriers for those annotated ops. This proves data visibility but does not structurally prove multi-buffering — a single-buffer case would be incorrectly filtered. The filter's correctness relies on the assumption that the pipeliner always generates `numBuffers >= 2`, violating pass independence.

## Solution: Symbolic Buffer Index Analysis

The direct fix is to teach membar how to reason about `MemDescIndexOp` indices. Extend `AllocationSlice::intersects` in core `Membar.cpp` to decompose each index into a canonical form `{baseValue, constantOffset, modulus}`. Two accesses are provably disjoint when they share the same SSA base, the same modulus, and different constant offsets:

```
slot[(phase + 1) % 3]  → {base=%phase, offset=1, mod=3}
slot[(phase + 2) % 3]  → {base=%phase, offset=2, mod=3}
                              same       1 ≠ 2     same  → disjoint
```

Recognized patterns: `arith.remsi` (Gluon `%` operator) and `select/cmpi` (pipeliner's `createIncrementModulo`). Unrecognized patterns fall back to conservative. Loop-carried slices (across backedges) are detected via `DominanceInfo` and excluded from expression matching.

This approach is clean and self-contained:
- Lives in core membar — benefits all backends, not AMD-specific.
- No annotations or attributes — nothing to lose during lowering.
- Single-buffer safe by construction — same offsets are never reported as disjoint.
- Replaces the `syncedViaAsyncWait` machinery entirely.

Requires the pipeliner to create separate `MemDescIndexOp`s for producer and consumer, both derived from the same SSA phase counter with different constant offsets.

See [membar-dynamic-index-disjointness.md](membar-dynamic-index-disjointness.md) for full design and implementation.

## Considered Alternative: Buffer Slot Coloring

We also evaluated an attribute-based approach where each `MemDescIndexOp` is tagged with a `buffer_color` integer, and the AMD membar filter treats different colors as disjoint:

```mlir
%read  = ttg.memdesc_index %alloc[%phase] {buffer_color = 0}
%write = ttg.memdesc_index %alloc[%next]  {buffer_color = 1}
```

This decouples pattern recognition from the disjointness decision — the producer declares the contract, and the filter just compares integers. This has practical value in two areas:

- **Extensibility** — new index idioms or pipeliner strategies require no changes to the filter. With symbolic analysis, each new pattern needs a new case in `analyzeBufferIndex`.
- **Gluon flexibility** — Gluon users can compute indices however they want without being constrained to patterns the compiler recognizes.

However, it introduces AMD-specific machinery, depends on attributes surviving lowering, and requires a new Gluon API (`colored_memdesc_index`). It also requires the same pipeliner change as symbolic analysis (separate `MemDescIndexOp`s per stage).

See [membar-buffer-slot-coloring.md](membar-buffer-slot-coloring.md) for full design.

## Comparison

| | Symbolic Index Analysis | Buffer Coloring |
|---|---|---|
| **Where** | Core `Membar.cpp` | AMD filter |
| **Mechanism** | Compiler infers disjointness from index arithmetic | Code generator declares disjointness via attribute |
| **Single-buffer safe** | Yes (by construction) | Depends on correct assignment |
| **Benefits all backends** | Yes | No |
| **Pattern-dependent** | Yes (`remsi`, `select/cmpi`) | No |
| **Attribute needed** | No | Yes (`buffer_color`) |
| **Gluon support** | Automatic (if index patterns match) | Explicit (`colored_memdesc_index`) |
| **Extensibility** | New idioms need new pattern matchers | Producer stamps color; filter unchanged |
| **Pipeliner change** | Separate `MemDescIndexOp`s, unified counter | Separate `MemDescIndexOp`s |

## Where We Stand

Symbolic index analysis is the straightforward solution — it fixes the gap in membar's reasoning directly, works across backends, and requires no annotations or new API surface. The recognized patterns (`remsi`, `select/cmpi`) already cover what the pipeliner and Gluon produce today.

Buffer coloring remains a viable complement if we encounter index patterns that fall outside the recognized set, or if Gluon users need explicit control that doesn't depend on the compiler's pattern matching. The two operate at different layers and could coexist without conflict.

Both share the same prerequisite: the AMD pipeliner must create separate `MemDescIndexOp`s for producer and consumer stages. Either approach replaces the existing `syncedViaAsyncWait` annotation pass and `filterAsyncWriteDependencies` filter.
