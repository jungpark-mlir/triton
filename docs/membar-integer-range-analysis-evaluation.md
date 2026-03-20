# Evaluating Integer Range Analysis for Dynamic Buffer Index Disjointness

## Context

Triton's membar analysis inserts shared memory barriers between operations that
may access overlapping regions. In multi-buffered pipelines, a producer
(async copy) and a consumer (local load) access different buffer slots within
the same iteration, but membar cannot distinguish them because the slot index
is dynamic. This causes **false positive** barrier insertions that hurt
performance.

The current solution in development (`BufferIndexExpr` in the `mbar` branch)
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

The disjointness check is trivial:
- Same base `%phase` ✓
- Same modulus `3` ✓
- Offsets `0 ≠ 2` (mod 3) ✓
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

## Other Approaches in the Wider Ecosystem

To ensure we are not missing a better technique, this section surveys how
other compilers and frameworks solve similar problems.

### LLVM ScalarEvolution Alias Analysis (SCEV-AA)

LLVM's `SCEVAAResult::alias()` proves `NoAlias` by computing the symbolic
difference between two pointer expressions (`getMinusSCEV(A, B)`) and checking
whether the unsigned range of the difference exceeds the access sizes.

For our problem, the difference `(phase % 3) - ((phase + 2) % 3)` would
require SCEV to reason symbolically about `srem`. SCEV does not have a
first-class `srem` recurrence representation — it would model each index as an
opaque expression, making `getMinusSCEV` return `SCEVCouldNotCompute`. Even if
it could compute the difference, the result set `{-2, -1, 1, 2}` is
non-contiguous, so the interval-based range check would lose precision and
fall back to `MayAlias`.

**Verdict**: SCEV-AA's difference-range technique cannot handle modular buffer
indices. It is designed for linear pointer arithmetic, not modular slot
selection.

### MLIR Affine Dependence Analysis (Presburger Arithmetic)

MLIR's affine dialect includes `checkMemrefAccessDependence`, which formulates
dependence queries as integer linear programs over `FlatAffineValueConstraints`
(built on MLIR's Presburger library). The Presburger library supports modulo
and floor division via existential (local) variables — `addLocalModulo` encodes
`x mod N` as an existential `q` with constraints `x = N*q + r, 0 ≤ r < N`.

For our problem, the query "can `phase % 3 == (phase + 2) % 3`?" becomes:

```
r1 = phase - 3*q1,   0 ≤ r1 < 3
r2 = phase + 2 - 3*q2, 0 ≤ r2 < 3
r1 = r2
```

Substituting the equality: `3*(q2 - q1) = 2`. The GCD test (embedded in
`IntegerRelation::isIntegerEmpty`) immediately sees `gcd(3) = 3` does not
divide 2 — **no integer solution exists, accesses are provably disjoint**.

This is mathematically the most powerful available approach. It handles
arbitrary affine index expressions, not just recognized patterns.

| Property | Assessment |
|----------|------------|
| **Proves disjointness** | Yes, for any affine + modulo index expression. |
| **Runtime cost** | GCD test + Simplex + Fourier-Motzkin on ~5 variables, ~10 constraints. Negligible for our problem size. |
| **Engineering effort** | **Medium-high.** Requires: (1) extracting index expressions from `arith` ops into affine form, (2) building an `IntegerRelation` with existential variables for modulo, (3) calling `isIntegerEmpty()`, (4) integrating the Presburger library into the membar build. |
| **Maintenance risk** | Medium. Depends on MLIR's Presburger library (stable, upstream). The constraint-building code is the fragile part. |

### Polyhedral Model / ISL

The Integer Set Library (ISL), used by LLVM Polly and GCC Graphite, performs
exact dependence analysis over quasi-affine expressions (including modulo and
floor division). ISL could model our buffer index problem and solve it with
exact integer arithmetic.

ISL is strictly more powerful than MLIR's Presburger library but adds an
external C library dependency. Since MLIR already provides equivalent
Presburger functionality, using ISL directly would be redundant.

**Verdict**: Equivalent capability to MLIR Presburger, but unnecessary given
that the Presburger library is already linked into Triton.

### Classical Dependence Tests (GCD, Banerjee)

The **GCD test** checks whether `gcd(a1, a2, ..., an)` divides a constant `c`
in a linear Diophantine equation. For our modular buffer problem, the GCD test
is exactly what proves disjointness (as shown in the Presburger formulation
above). It is a necessary condition for integer solutions — if it fails, no
dependence exists.

The **Banerjee test** provides tighter bounds using direction vectors but
is designed for loop-carried dependences across iterations, not
intra-iteration slot selection.

**Verdict**: The GCD test is the core mathematical tool that makes disjointness
proofs work. Both `BufferIndexExpr` and the Presburger approach ultimately
rely on the same arithmetic property: `N ∤ k` implies `(x mod N) ≠ ((x+k) mod N)`.

### MLIR GPU Barrier Elimination (`EliminateBarriers`)

MLIR's upstream `GPUEliminateBarriers` pass (based on Moses et al., PPoPP 2023)
removes `gpu.barrier` ops that do not enforce any conflicting memory effect
pair. It works at the **memory effects interface** level — checking whether
reads and writes to shared memory exist between barriers.

This is a coarser analysis that does not reason about which sub-regions of
shared memory are accessed. It can remove barriers when there are no
conflicting effects at all, but cannot distinguish accesses to different slots
of the same buffer.

**Verdict**: Complementary to our problem but cannot solve it. It operates
at the buffer level, not the slot level.

### Relational Abstract Domains (Octagons, Polyhedra)

In abstract interpretation theory, **relational domains** can express
constraints between multiple variables:

- **Difference constraints**: `x - y ≤ c` (O(n³) closure)
- **Octagon domain**: `±x ± y ≤ c` (O(n³) closure)
- **Polyhedral domain**: General linear constraints (exponential worst case)

Any of these could express `idx_write - idx_read ≡ 2 (mod 3)` and prove
disjointness. However:

- MLIR has no built-in relational integer abstract domain.
- Implementing one is a major undertaking (thousands of lines for octagons,
  tens of thousands for polyhedra).
- The runtime cost is at least O(n³) per join/transfer, where n is the number
  of tracked variables.
- Massively overengineered for our specific two-variable, one-modulus pattern.

**Verdict**: Theoretically capable but impractical for this problem.

## A New Idea: Lightweight Presburger Check

The survey reveals a middle ground between `BufferIndexExpr` and full affine
dependence analysis:

Instead of pattern-matching specific index forms, we could:

1. When `AllocationSlice::intersects()` sees two `MemDescIndexOp` accesses,
   extract each index's definition as a small expression tree.
2. Recursively build an `IntegerRelation` from the `arith` ops:
   - `arith.constant C` → add equality `var = C`
   - `arith.addi(x, y)` → add equality `var = x + y`
   - `arith.remsi(x, N)` → use `addLocalModulo` (existential variable)
   - Unknown op → leave as unconstrained variable (conservative)
3. Add the equality constraint `idx1 = idx2`.
4. Call `isIntegerEmpty()`. If empty → provably disjoint.

This would handle **any** affine + modulo index expression, not just the
`remsi` and `select/cmpi` patterns that `BufferIndexExpr` recognizes. It would
also naturally handle nested expressions like `((phase + 1) % 3 + 1) % 3`.

| Property | BufferIndexExpr | Lightweight Presburger |
|----------|----------------|----------------------|
| **Patterns handled** | `remsi`, `select/cmpi`, `addi` | Any affine + modulo expression |
| **Implementation** | ~110 lines, zero dependencies | ~200-300 lines + Presburger library dependency |
| **Runtime cost** | O(expression depth), ~ns | GCD + Simplex, ~µs (still negligible) |
| **Extensibility** | Must add new patterns manually | Automatically handles new arith ops |
| **Debugging** | Easy: inspect `{base, offset, mod}` | Harder: inspect constraint matrix |
| **Build dependency** | None (self-contained in Membar) | MLIR Presburger library (already linked in the build) |

The tradeoff is clear: `BufferIndexExpr` is simpler and covers the patterns
we actually see today; the Presburger approach is more general but adds
complexity. Whether the generality is worth the cost depends on how many
unrecognized patterns we expect to encounter in practice.

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

The wider ecosystem survey confirms that approaches capable of proving
modular disjointness (Presburger arithmetic, polyhedral analysis, SCEV) all
rely on the same core insight: reasoning about the *structure* of the index
computation, not just its *range*. `BufferIndexExpr` is a minimal
instantiation of this principle — it hand-rolls the GCD test for the specific
modular patterns emitted by the pipeliner.

The Presburger-based approach is a viable alternative if more generality is
needed in the future, as it handles arbitrary affine + modulo expressions
with existing MLIR infrastructure. For the current scope of patterns,
`BufferIndexExpr` remains the best fit: minimal, sound, and precisely scoped
to the pipeliner's multi-buffer idiom.

## Summary Table

| Approach | Can Prove Disjointness | Engineering Effort | Runtime Cost | Maintenance Risk |
|----------|----------------------|-------------------|-------------|-----------------|
| Integer Range Analysis (as-is) | **No** | Zero (exists) | N/A | N/A — doesn't solve the problem |
| LLVM SCEV-AA | **No** | N/A | N/A | N/A — wrong abstraction level |
| Relational Abstract Domain | Yes | Very High (~10k LOC) | O(n³) per join | High — new infrastructure |
| Presburger / Affine Dependence | **Yes** | Medium (~200-300 LOC) | ~µs (GCD + Simplex) | Medium — depends on MLIR Presburger |
| BufferIndexExpr (symbolic) | **Yes** | Low (~110 LOC) | ~ns (pattern match) | Low — self-contained |
