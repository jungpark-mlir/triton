# Evaluating Integer Range Analysis for Dynamic Buffer Index Disjointness

## Context

Triton's membar analysis inserts shared memory barriers between operations that
may access overlapping regions. In multi-buffered pipelines, a producer
(async copy) and a consumer (local load) access different buffer slots within
the same iteration, but membar cannot distinguish them because the slot index
is dynamic. This causes **false positive** barrier insertions that hurt
performance.

The proposed solution (`BufferIndexExpr`, a symbolic index decomposition)
uses symbolic decomposition of the index arithmetic to prove slot disjointness.
This document evaluates whether Triton's **integer range analysis** could serve
as an alternative or complement.

## Triton's Integer Range Analysis

### Overview

The AMD backend extends MLIR's upstream `IntegerRangeAnalysis`, a sparse
forward dataflow analysis that infers lower/upper bounds on integer SSA values.
The implementation lives in:

- `third_party/amd/include/Analysis/RangeAnalysis.h`
- `third_party/amd/lib/Analysis/RangeAnalysis.cpp`

`TritonIntegerRangeAnalysis` subclasses `mlir::dataflow::IntegerRangeAnalysis`
and adds:

1. **Triton-specific op support**: `GetProgramIdOp`, `MakeRangeOp`, `SplatOp`,
   `ExpandDimsOp`, `GatherOp`, etc.
2. **Loop abstract interpretation**: Instead of conservatively widening
   loop-carried values to `[INT_MIN, INT_MAX]`, it tracks loop trip counts and
   propagates lattice states up to N iterations, producing tighter ranges for
   `scf.for` iter args.
3. **Assumption-based narrowing**: `llvm.intr.assume` patterns from `arith.cmpi`
   are collected and used to narrow entry ranges.

### Data Structures

| Type | Description |
|------|-------------|
| `ConstantIntRanges` | Concrete bounds: `{smin, smax, umin, umax}` |
| `IntegerValueRange` | Lattice payload: uninitialized or a `ConstantIntRanges` |
| `IntegerValueRangeLattice` | Per-SSA-value lattice element |

### Transfer Functions for Key Ops

**Arithmetic ops** (`addi`, `remsi`, `select`, etc.) are handled by MLIR's
`InferIntRangeInterface`, not hand-written in Triton's extension. The key
behaviors:

- **`arith.addi`**: `[lhs.min + rhs.min, lhs.max + rhs.max]` (with overflow
  flag handling).
- **`arith.remsi`**: For `x % N` where N is a positive constant, the result
  range is `[0, N-1]` when the dividend is non-negative. A special case
  tightens this when the dividend range spans fewer than N values (i.e., the
  remainder doesn't wrap).
- **`arith.select`**: Union of true/false operand ranges, unless the condition
  is a known constant (then only the taken branch).

### Loop Handling

The analysis computes trip counts for `scf.for` loops and "abstractly
interprets" the loop body N times, joining lattice states at each back-edge
propagation. For loops with unknown or large trip counts (> 1024), it falls
back to the conservative `[INT_MIN, INT_MAX]` range for carried values.

### Current Consumers

| Pass | Purpose |
|------|---------|
| `TritonAMDFoldTrueCmpIOp` | Fold provably-true comparisons |
| `TritonAMDGPUOptimizeBufferOpPtr` | Check offset overflow for buffer ops |
| `TritonAMDGPUConvertToBufferOps` | Buffer lowering decisions |

**No existing integration with membar or allocation analysis.**

## The Problem: Why Range Analysis Cannot Prove Disjointness

### The Core Limitation: Non-Relational Domain

Integer range analysis is a **non-relational** abstract domain. It tracks the
possible values of each SSA value *independently*, as an interval. It cannot
express or reason about relationships between two values.

In a pipelined loop body, the producer and consumer indices are:

```mlir
// Producer: write to slot (phase + 2) % 3
%w_off = arith.addi %phase, %c2 : i32
%w_idx = arith.remsi %w_off, %c3 : i32

// Consumer: read from slot phase % 3
%r_idx = arith.remsi %phase, %c3 : i32
```

Range analysis computes:

| Value | Inferred Range |
|-------|----------------|
| `%phase` | `[0, trip_count - 1]` |
| `%w_off` | `[2, trip_count + 1]` |
| `%w_idx` | `[0, 2]` |
| `%r_idx` | `[0, 2]` |

```
  Range analysis view (non-relational):

  %r_idx:  ├─────────────────────┤    range = [0, 2]
  %w_idx:  ├─────────────────────┤    range = [0, 2]
           0         1         2

           ^^^^^^^^^^^^^^^^^^^^^^^^
           Complete overlap — "may alias"

  Reality (relational — what actually happens per iteration):

  iter 0:  %r_idx=0   %w_idx=2   → disjoint
  iter 1:  %r_idx=1   %w_idx=0   → disjoint
  iter 2:  %r_idx=2   %w_idx=1   → disjoint
           ↑ never equal, but range analysis can't see this
```

Both `%w_idx` and `%r_idx` have range `[0, 2]`. These ranges **overlap
completely**. Range analysis has no way to express the constraint that
`%w_idx ≠ %r_idx` at any given program point. The information that they are
**derived from the same base with different offsets** is lost once the ranges
are computed.

### Concrete Walkthrough

Consider a triple-buffered loop with 6 iterations (`phase` ∈ `[0, 5]`):

| Iteration | `phase` | `%r_idx` = `phase % 3` | `%w_idx` = `(phase+2) % 3` | Disjoint? |
|-----------|---------|------------------------|----------------------------|-----------|
| 0 | 0 | 0 | 2 | Yes |
| 1 | 1 | 1 | 0 | Yes |
| 2 | 2 | 2 | 1 | Yes |
| 3 | 3 | 0 | 2 | Yes |
| 4 | 4 | 1 | 0 | Yes |
| 5 | 5 | 2 | 1 | Yes |

At **every** iteration, the indices are different. But range analysis merges
all iterations into a single interval per value, losing the per-iteration
correlation.

### The `inferRemS` Special Case

MLIR's `inferRemS` has a special case for contiguous dividend ranges shorter
than the modulus:

```
if (lhsMax - lhsMin < N) and (lhsMin % N ≤ lhsMax % N):
    result range = [lhsMin % N, lhsMax % N]
```

This would help if we could evaluate ranges **per iteration** (e.g., when
`phase = 3`: `%r_idx` range = `[0, 0]`, `%w_idx` range = `[2, 2]` → disjoint).
But the analysis produces a single **union** over all iterations, so the final
ranges are still `[0, 2]` for both.

### What Would Be Needed

To prove `%w_idx ≠ %r_idx`, we would need a **relational** abstract domain
that can express constraints between two values:

- **Difference constraints**: `%w_idx - %r_idx ≡ 2 (mod 3)`
- **Octagon domain**: `±x ± y ≤ c`
- **Polyhedral domain**: General linear constraints

These are well-studied in abstract interpretation theory, but:

- They are orders of magnitude more expensive than interval analysis.
- MLIR has no built-in relational integer domain.
- Implementing one for this single use case would be massive overengineering.

## Comparison with BufferIndexExpr (Symbolic Index Analysis)

### What BufferIndexExpr Does Differently

`BufferIndexExpr` is a **lightweight symbolic domain** that captures exactly
the information needed for the disjointness proof:

```
BufferIndexExpr = { baseValue: SSA Value, constantOffset: int, modulus: optional<int> }
```

For the example above:

| Value | BufferIndexExpr |
|-------|-----------------|
| `%r_idx` = `remsi(%phase, 3)` | `{base=%phase, offset=0, mod=3}` |
| `%w_idx` = `remsi(addi(%phase, 2), 3)` | `{base=%phase, offset=2, mod=3}` |

```
  BufferIndexExpr preserves the structural relationship:

  %phase ───┬── remsi(%phase, 3)     → {base=%phase, offset=0, mod=3}
            │
            └── remsi(%phase+2, 3)   → {base=%phase, offset=2, mod=3}
                                             │           │         │
                                         same base   0 ≠ 2    same mod
                                                  ↓
                                          provably disjoint
```

The disjointness check is trivial:
- Same base `%phase`
- Same modulus `3`
- Offsets `0 ≠ 2` (mod 3)
- **Provably disjoint.**

### Why This Works and Range Analysis Does Not

| Property | Integer Range Analysis | BufferIndexExpr |
|----------|----------------------|-----------------|
| **Domain type** | Non-relational intervals | Symbolic expression |
| **Per-value info** | `[min, max]` bounds | `{base, offset, modulus}` |
| **Can prove `A ≠ B`?** | Only if ranges are disjoint | Yes, via structural comparison |
| **Loop handling** | Union over all iterations | Intra-iteration SSA identity |
| **Implementation** | General-purpose dataflow solver | Pattern-matched decomposition |
| **Scope** | All integer values in the function | Only `MemDescIndexOp` indices |

The fundamental difference: range analysis answers "**what values can X
take?**" while `BufferIndexExpr` answers "**how is X computed relative to
Y?**" The disjointness problem requires the relational question, not the
bounding question.

### Can Range Analysis Complement BufferIndexExpr?

There are a few places where range analysis could supplement `BufferIndexExpr`:

1. **Validating modulus bounds**: Range analysis could confirm that the
   modulus `N` in `remsi` is positive, avoiding edge cases in signed remainder.
   However, the pipeliner always emits constant positive moduli, so this adds
   little practical value.

2. **Narrowing opaque indices**: If `BufferIndexExpr` fails to decompose an
   index (returning an opaque representation), range analysis could still
   provide bounds. But bounds alone cannot prove disjointness.

3. **Constant folding**: If range analysis can fold a `cmpi` to a constant
   (via `cmpIIsStaticallyTrue`), it could simplify the select/cmpi modular
   pattern before `BufferIndexExpr` sees it. This is marginal since
   `BufferIndexExpr` already handles both polarities of the pattern.

None of these would replace `BufferIndexExpr`; they would only provide minor
validation or preprocessing.

## Design Suitability Evaluation

### Why Integer Range Analysis is Not Suitable

| Criterion | Assessment |
|-----------|------------|
| **Proves disjointness** | No. Overlapping ranges for congruent modular indices. |
| **Correctness** | Sound but too imprecise — always says "may overlap." |
| **Complexity** | Already implemented, but would need relational extension. |
| **Maintenance** | General-purpose infrastructure; modifying it for one use case is inappropriate. |
| **Integration** | No existing connection to membar; would require new plumbing. |

### Why BufferIndexExpr is the Right Approach

| Criterion | Assessment |
|-----------|------------|
| **Proves disjointness** | Yes, for recognized patterns. |
| **Correctness** | Sound: conservative fallback for unrecognized patterns. |
| **Complexity** | ~100 lines of pattern matching + ~10 lines in `intersects`. |
| **Maintenance** | Self-contained in membar; no external dependencies. |
| **Integration** | Fits naturally into `AllocationSlice::intersects`. |

### When to Consider Range Analysis for Membar

Integer range analysis would become relevant if the problem were different:

- **Proving an index is in-bounds**: "Is this `MemDescIndexOp` index within
  `[0, numBuffers)`?" — range analysis can answer this.
- **Proving a single index is constant**: "Is this index always 0?" — range
  analysis can fold this.
- **Simplifying control flow around buffer accesses**: Range-based constant
  folding of `cmpi` guards could simplify IR before membar runs.

These are orthogonal to the multi-buffer disjointness problem.

## Conclusion

Integer range analysis and symbolic index analysis solve fundamentally
different problems:

- **Range analysis** bounds each value independently: "X ∈ [0, 2]." It is a
  **non-relational** domain that cannot express or verify `X ≠ Y`.

- **Symbolic index analysis** (`BufferIndexExpr`) captures the structural
  relationship between two computations: "X and Y share the same base but
  differ by a constant offset modulo N." This is a **relational** property,
  precisely what is needed to prove buffer slot disjointness.

Attempting to use range analysis for this problem would require extending it
into a relational domain (difference constraints or polyhedra), which would be
disproportionately complex for the narrow pattern we need to handle.
`BufferIndexExpr` is the correct tool: minimal, sound, and precisely scoped
to the pipeliner's multi-buffer idiom.

## Summary Table

| Approach | Can Prove Disjointness | Implementation Effort | Maintenance Risk |
|----------|----------------------|----------------------|-----------------|
| Integer Range Analysis (as-is) | **No** | Zero (exists) | N/A — doesn't solve the problem |
| Integer Range Analysis (relational extension) | Yes | Very High | High — modifying general infrastructure |
| BufferIndexExpr (symbolic decomposition) | **Yes** | Low (~110 lines) | Low — self-contained in membar |
