# Approaches to Dynamic Buffer Index Disjointness

## Problem

Triton's membar analysis inserts shared memory barriers between operations that
may access overlapping regions. In multi-buffered pipelines, a producer
(async copy) and a consumer (local load) access different buffer slots within
the same iteration, but membar cannot distinguish them because the slot index
is dynamic. This causes **false positive** barrier insertions that hurt
performance.

The core question: given two `MemDescIndexOp` accesses into the same
allocation with dynamic indices, can the compiler prove they refer to
different buffer slots?

```mlir
%w_off = arith.addi %phase, %c2 : i32
%w_idx = arith.remsi %w_off, %c3 : i32
%write = ttg.memdesc_index %alloc[%w_idx]   // producer

%r_idx = arith.remsi %phase, %c3 : i32
%read  = ttg.memdesc_index %alloc[%r_idx]   // consumer
```

This document surveys approaches to solving this problem, evaluating each
on correctness, engineering effort, runtime cost, and maintenance risk.

## Approaches That Cannot Solve the Problem

### 1. Integer Range Analysis (Triton / MLIR)

The AMD backend extends MLIR's `IntegerRangeAnalysis`, a sparse forward
dataflow analysis that infers `[min, max]` bounds per SSA value. It is
implemented in `third_party/amd/lib/Analysis/RangeAnalysis.cpp` and currently
used by passes like `FoldTrueCmpIOp` and `ConvertToBufferOps`.

**Why it fails**: Range analysis is a **non-relational** interval domain. It
computes bounds for each value independently. For the example above:

| Value | Inferred Range |
|-------|----------------|
| `%w_idx` | `[0, 2]` |
| `%r_idx` | `[0, 2]` |

```
  What range analysis sees:        What actually happens:

  %r_idx: ├──────────────┤ [0,2]   iter 0: r=0, w=2 (disjoint)
  %w_idx: ├──────────────┤ [0,2]   iter 1: r=1, w=0 (disjoint)
          0      1      2          iter 2: r=2, w=1 (disjoint)
    "total overlap → may alias"      "never equal → always disjoint"
```

Both indices have range `[0, 2]` — overlapping completely. The fact that they
are derived from the **same base with different offsets** is lost once ranges
are computed. Range analysis answers "what values can X take?" but cannot
answer "is X always different from Y?"

Even with the AMD extension's loop abstract interpretation (which propagates
ranges per-iteration up to the trip count), the final lattice value is a
**union** over all iterations, producing `[0, 2]` for both.

**Could it complement another approach?** Marginally. It could validate that
a modulus is positive or fold a `cmpi` to a constant, but these are edge cases
that `BufferIndexExpr` already handles.

### 2. LLVM ScalarEvolution Alias Analysis (SCEV-AA)

SCEV-AA proves `NoAlias` by computing the symbolic difference between two
pointer expressions (`getMinusSCEV`) and checking whether the range of the
difference exceeds the access sizes.

**Why it fails**: SCEV does not have a first-class `srem` recurrence. It would
model each modular index as an opaque expression, returning
`SCEVCouldNotCompute` for the difference. Even if it could compute it, the
result set `{-2, -1, 1, 2}` is non-contiguous — the interval-based check
would lose precision and fall back to `MayAlias`.

### 3. MLIR GPU Barrier Elimination (`EliminateBarriers`)

MLIR's upstream `GPUEliminateBarriers` pass (Moses et al., PPoPP 2023) removes
`gpu.barrier` ops that don't enforce any conflicting memory effect pair. It
works at the **memory effects interface** level — presence or absence of
reads/writes to shared memory.

**Why it fails**: It operates at the buffer level, not the slot level. It can
remove a barrier when there are no conflicting effects at all, but cannot
distinguish accesses to different slots of the same buffer.

### 4. Relational Abstract Domains (Octagons, Polyhedra)

Domains like difference constraints (`x - y ≤ c`), octagons (`±x ± y ≤ c`),
or general polyhedra can express and verify relational properties between
variables. They could in principle prove `idx_write ≠ idx_read`.

**Why it's impractical**: MLIR has no built-in relational integer domain.
Implementing one requires thousands of lines (octagons) to tens of thousands
(polyhedra), with O(n³) runtime per join. Massively overengineered for a
two-variable, one-modulus pattern.

## Approaches That Can Solve the Problem

### 5. BufferIndexExpr — Symbolic Index Decomposition (Current)

`BufferIndexExpr` decomposes each `MemDescIndexOp` index into a canonical
`{baseValue, constantOffset, modulus}` by pattern-matching the defining
`arith` ops:

| IR Pattern | Decomposition |
|---|---|
| `arith.constant C` | `{nullptr, C, nullopt}` |
| `arith.addi(x, C)` | `{base(x), offset(x) + C, nullopt}` |
| `arith.remsi(x, N)` | `{base(x), offset(x), mod=N}` |
| `select(cmpi, addi(base, C), N)` | `{base, offset + C, mod=N}` |

Two indices are provably disjoint when they share the same SSA base, the same
modulus, and their offsets differ modulo N:

| Value | BufferIndexExpr |
|-------|-----------------|
| `%r_idx` = `remsi(%phase, 3)` | `{base=%phase, offset=0, mod=3}` |
| `%w_idx` = `remsi(addi(%phase, 2), 3)` | `{base=%phase, offset=2, mod=3}` |

Same base, same modulus, `0 ≠ 2 (mod 3)` → **provably disjoint**.

This is essentially a hand-rolled GCD test specialized for pipeliner-emitted
modular patterns. Unrecognized patterns fall through to an opaque
representation that conservatively assumes overlap.

| Property | Assessment |
|----------|------------|
| **Proves disjointness** | Yes, for `remsi`, `select/cmpi`, `addi` patterns. |
| **Engineering effort** | Low: ~110 lines, zero new dependencies. |
| **Runtime cost** | O(expression depth), ~nanoseconds. |
| **Maintenance** | Self-contained in membar; easy to debug (`{base, offset, mod}`). |
| **Limitation** | Only handles recognized patterns; must be extended manually for new forms. |

### 6. Presburger Constraint Solving

MLIR's **Presburger library** (`mlir::presburger::IntegerRelation`) is a
standalone integer constraint solver under `mlir/Analysis/Presburger/`. It is
**not** the affine dialect — it has no dependency on affine maps, memrefs, or
the affine dialect's dependence analysis infrastructure. It is a pure math
library that the affine dialect happens to use, but can be used independently.

The key capability: `addLocalModulo` encodes `x mod N` as an existential
variable `q` with constraints `x = N*q + r, 0 ≤ r < N`.
`IntegerRelation::isIntegerEmpty()` then uses the GCD test, Simplex, and
Fourier-Motzkin elimination to check satisfiability.

For our problem, the query "can `phase % 3 == (phase + 2) % 3`?" becomes:

```
r1 = phase - 3*q1,     0 ≤ r1 < 3
r2 = phase + 2 - 3*q2, 0 ≤ r2 < 3
r1 = r2
```

Substituting: `3*(q2 - q1) = 2`. The GCD test sees `gcd(3) ∤ 2` → **no
integer solution → provably disjoint**.

```
  Presburger encoding:

  Variables: phase, q1, q2, r1, r2

  Constraints:
  ┌──────────────────────────────────────────────┐
  │  r1 = phase - 3·q1          (mod definition) │
  │  0 ≤ r1 < 3                 (range bound)    │
  │  r2 = phase + 2 - 3·q2      (mod definition) │
  │  0 ≤ r2 < 3                 (range bound)    │
  │  r1 = r2                    (equality query)  │
  └──────────────────────────────────────────────┘
           ↓ substitute r1 = r2
  3·(q2 - q1) = 2
           ↓ GCD test
  gcd(3) = 3,  3 ∤ 2  →  no integer solution
           ↓
  System is empty → indices are always different
```

Instead of pattern-matching specific index forms, the approach would:

1. When `AllocationSlice::intersects()` sees two `MemDescIndexOp` accesses,
   walk the `arith` ops defining each index.
2. Recursively build a small `IntegerRelation`:
   - `arith.constant C` → equality constraint
   - `arith.addi(x, y)` → equality constraint
   - `arith.remsi(x, N)` → `addLocalModulo` (existential variable)
   - Same SSA base in both expressions → shared variable
   - Unknown op → unconstrained variable (conservative)
3. Add `idx1 = idx2` and call `isIntegerEmpty()`.

| Property | Assessment |
|----------|------------|
| **Proves disjointness** | Yes, for any affine + modulo index expression. |
| **Engineering effort** | Medium: ~200-300 lines for constraint building. Presburger library is already linked in Triton's build. |
| **Runtime cost** | GCD + Simplex on ~5 variables, ~10 constraints. ~microseconds, negligible. |
| **Maintenance** | Presburger library is stable upstream MLIR. Constraint-building code is the fragile part. |
| **Advantage over BufferIndexExpr** | Handles arbitrary compositions automatically (e.g., nested modulo, multi-step additions). No need to add new patterns manually. |

### 7. Classical GCD Test (Theoretical Basis)

The GCD test checks whether `gcd(a1, ..., an) | c` in a linear Diophantine
equation `a1*x1 + ... + an*xn = c`. If it fails, no integer solution exists.

Both BufferIndexExpr and the Presburger approach ultimately rely on this same
arithmetic property: **`N ∤ k` implies `(x mod N) ≠ ((x+k) mod N)` for all
integer x**. BufferIndexExpr encodes this implicitly via offset comparison;
Presburger encodes it explicitly via constraint solving.

The GCD test alone would suffice for the specific problem, but requires the
modular structure to be extracted from the IR first — which is what approaches
5 and 6 do in different ways.

### 8. Polyhedral Model / ISL

The Integer Set Library (ISL), used by LLVM Polly and GCC Graphite, performs
exact dependence analysis over quasi-affine expressions including modulo. It
has equivalent capability to MLIR's Presburger library but adds an external C
library dependency. Since the Presburger library is already in MLIR, using ISL
directly would be redundant.

## Comparison

| Approach | Proves Disjointness | Engineering Effort | Runtime Cost | Maintenance Risk |
|----------|:-------------------:|-------------------|-------------|-----------------|
| Integer Range Analysis | **No** | — | — | — |
| LLVM SCEV-AA | **No** | — | — | — |
| MLIR GPU `EliminateBarriers` | **No** | — | — | — |
| Relational Abstract Domain | Yes | Very High (~10k LOC) | O(n³) per join | High |
| **BufferIndexExpr** | **Yes** | **Low (~110 LOC)** | **~ns** | **Low** |
| **Presburger Constraint Check** | **Yes** | **Medium (~200-300 LOC)** | **~µs** | **Medium** |
| ISL / Polyhedral | Yes | High (external dep) | ~µs | Medium |

## Conclusion

Proving dynamic buffer index disjointness is a **relational** problem: it
requires showing that two values derived from the same base are always
different, not just that each value falls within some range. This rules out
non-relational approaches (range analysis, SCEV-AA) regardless of how
sophisticated they are.

Among approaches that work, **BufferIndexExpr** and **Presburger constraint
solving** are the two practical options:

- **BufferIndexExpr** is the right choice for the current scope. It covers
  the patterns emitted by the pipeliner (`remsi`, `select/cmpi`, `addi`) with
  minimal code and zero dependencies. It is essentially a hand-rolled GCD
  test specialized for buffer slot arithmetic.

- **Presburger constraint solving** is a viable upgrade path if we encounter
  index patterns that BufferIndexExpr cannot decompose. It handles arbitrary
  affine + modulo expressions using MLIR's existing Presburger library (not
  the affine dialect), at modest engineering cost and negligible runtime
  overhead. It could replace or augment BufferIndexExpr if the need arises.
