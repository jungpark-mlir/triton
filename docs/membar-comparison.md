# Membar False Positive Suppression: Comparison

## Problem

In pipelined kernels with multi-buffering, `local_load` and `async_copy` (or TDM copy/gather) access different buffer slots of the same shared memory allocation. The buffer slot is selected via `MemDescIndexOp` with a dynamic loop-carried index.

Membar analysis (`AllocationSlice::intersects`) can only resolve static offsets from `MemDescSubsliceOp`. Since `MemDescIndexOp` uses dynamic indices, membar cannot distinguish disjoint buffer slots and conservatively reports a hazard, inserting an unnecessary barrier.

This false barrier exists on both the Triton pipeliner path and the Gluon path when users write multi-buffered kernels by hand.

## Prior Attempt (PR #9418 — rejected)

An AMD-specific annotation pass stamps `local_load` ops whose token chains back to an `async_wait`, and a backend filter suppresses barriers for those annotated ops. This proves data visibility but does not structurally prove multi-buffering — a single-buffer case would be incorrectly filtered. The filter's correctness relies on the assumption that the pipeliner always generates `numBuffers >= 2`, violating pass independence.

## Interaction with Other `local_load` Sources

Other passes also generate `local_load` ops (e.g., `ReduceDataDuplication` for layout conversions via shared memory). These create their own separate `local_alloc`, which is assigned a distinct buffer ID with a non-overlapping allocation interval. Membar's first check in `AllocationSlice::intersects` compares allocation intervals — since different `local_alloc`s have disjoint intervals, membar already knows they access separate memory and never reports a hazard between them. Neither alternative affects these cases.

## Proposed Alternative A: Symbolic Buffer Index Analysis

Extend `AllocationSlice::intersects` in core `Membar.cpp` to decompose `MemDescIndexOp` indices into a canonical form `{baseValue, constantOffset, modulus}`. Two accesses are provably disjoint when they share the same SSA base value, the same modulus, and different constant offsets:

```
slot[(phase + 1) % 3]  → {base=%phase, offset=1, mod=3}
slot[(phase + 2) % 3]  → {base=%phase, offset=2, mod=3}
                              same       1 ≠ 2     same  → disjoint
```

Recognized patterns: `arith.remsi` (Gluon `%` operator) and `select/cmpi` (pipeliner's `createIncrementModulo`). Unrecognized patterns fall back to conservative. Loop-carried slices (across backedges) are detected via `DominanceInfo` and excluded from expression matching.

Requires the pipeliner to create separate `MemDescIndexOp`s for producer and consumer, both derived from the same SSA phase counter with different constant offsets.

See [membar-dynamic-index-disjointness.md](docs/membar-dynamic-index-disjointness.md) for full design.

## Proposed Alternative B: Buffer Slot Coloring

Tag each `MemDescIndexOp` with a `buffer_color` integer attribute. The AMD membar filter treats accesses with different colors as provably disjoint:

```mlir
%read  = ttg.memdesc_index %alloc[%phase] {buffer_color = 0}
%write = ttg.memdesc_index %alloc[%next]  {buffer_color = 1}
// color 0 ≠ color 1 → filter suppresses barrier
```

No pattern recognition needed — works with any index computation. Gluon users set colors via `amd.colored_memdesc_index`. The pipeliner stamps colors at `MemDescIndexOp` creation time. If any pass drops the attribute, membar falls back to conservative (safe).

Requires the same pipeliner change (separate `MemDescIndexOp`s per stage) so the attribute is not lost to loop-carrying.

See [membar-buffer-slot-coloring.md](membar-buffer-slot-coloring.md) for full design.

## Comparison

| | Current Solution | Symbolic Index Analysis | Buffer Coloring |
|---|---|---|---|
| **Where** | AMD filter + annotation pass | Core `Membar.cpp` | AMD filter |
| **Mechanism** | Token chain → `syncedViaAsyncWait` | Symbolic index decomposition | `buffer_color` attribute |
| **Single-buffer safe** | No (relies on pipeliner) | Yes (by construction) | Depends on correct assignment |
| **Pipeliner change needed** | No | Yes (separate `MemDescIndexOp`s, unified counter) | Yes (separate `MemDescIndexOp`s) |
| **Benefits all backends** | No | Yes | No |
| **Pattern-dependent** | No (token chain) | Yes (`remsi`, `select/cmpi`) | No |
| **Attribute/annotation needed** | Yes (`syncedViaAsyncWait`) | No | Yes (`buffer_color`) |
| **Gluon support** | `load_shared_relaxed` | Automatic (if index patterns match) | `colored_memdesc_index` |
| **Handles separate counters** | N/A | No (different SSA bases) | Yes |
| **Extensibility** | N/A | New index idioms need new pattern matchers | Producer stamps color; filter unchanged |

## Recommendation

**Symbolic index analysis is the primary solution for the pipeliner path.** It is more principled — operates in core membar, requires no annotations or attributes, handles the single-buffer case correctly by construction, and benefits all backends. The AMD pipeliner must be adjusted to create separate `MemDescIndexOp`s from a unified phase counter, which is the same structural requirement as coloring.

**Buffer slot coloring is complementary for the Gluon path.** It provides an explicit escape hatch for Gluon users whose index patterns may not match the recognized set (e.g., complex multi-stage pipelines with non-standard modular arithmetic). The two approaches can coexist: symbolic index analysis handles the automatic pipeliner case, while coloring handles explicit user-directed suppression.

If both are implemented, the `syncedViaAsyncWait` annotation pass and `filterAsyncWriteDependencies` filter become unnecessary and can be removed.
