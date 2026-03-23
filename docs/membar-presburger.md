# MLIR Presburger Library for Membar Analysis

## 1. What is Presburger Arithmetic?

Presburger arithmetic is the first-order theory of the natural numbers with
addition (but not multiplication). It is **decidable**: any statement of the
form "does there exist an integer satisfying these linear constraints?" has
an algorithmic answer. This makes it the mathematical foundation for
reasoning about array subscripts, loop bounds, and modular index arithmetic
in compilers.

A Presburger formula can express:

- Linear equalities: `a*x + b*y = c`
- Linear inequalities: `a*x + b*y >= c`
- Integer divisibility: `N | (a*x + b)` (via existential variables)
- Modular arithmetic: `x mod N = r` (via floor division encoding)
- Existential quantification: "there exists an integer q such that..."

```
  Presburger arithmetic — what it can express:

  ┌─────────────────────────────────────────────────────┐
  │  Linear constraints                                 │
  │    a₁x₁ + a₂x₂ + ... + aₙxₙ + c ≥ 0              │
  │    a₁x₁ + a₂x₂ + ... + aₙxₙ + c = 0              │
  │                                                     │
  │  Existential variables (locals)                     │
  │    ∃q ∈ ℤ : x = N·q + r,  0 ≤ r < N               │
  │    (encodes x mod N = r)                            │
  │                                                     │
  │  Decidable emptiness                                │
  │    "Does the system have an integer solution?"      │
  │    → GCD test, Simplex, Fourier-Motzkin             │
  └─────────────────────────────────────────────────────┘
```

## 2. MLIR's Presburger Library

### Overview

MLIR includes a **standalone** Presburger arithmetic library under
`mlir/Analysis/Presburger/`. It is a pure math library — it has **no
dependency** on the affine dialect, affine maps, memrefs, or any IR
constructs. The affine dialect happens to use it, but any MLIR pass can
use it independently.

```
  MLIR layer diagram:

  ┌─────────────────────────────────────────────┐
  │         Affine Dialect                      │
  │  (affine.for, affine.if, AffineMap,         │
  │   FlatAffineValueConstraints, ...)          │
  │                 │ uses                      │
  ├─────────────────┼───────────────────────────┤
  │         Presburger Library                  │  ◄── standalone math
  │  (IntegerRelation, IntegerPolyhedron,       │      library
  │   PresburgerSet, Simplex, Matrix, ...)      │
  │                                             │
  │  No dependency on any dialect or IR!        │
  ├─────────────────────────────────────────────┤
  │         MLIR Core                           │
  │  (Value, Operation, Block, Region, ...)     │
  └─────────────────────────────────────────────┘

  Triton can use the Presburger library directly,
  without importing the Affine dialect.
```

### Key Classes

| Class | Role |
|-------|------|
| `PresburgerSpace` | Describes the variable structure: how many domain, range, symbol, and local (existential) variables |
| `IntegerRelation` | A single convex set of integer linear constraints (equalities and inequalities) |
| `IntegerPolyhedron` | An `IntegerRelation` with no domain/range split — a pure integer set |
| `PresburgerRelation` | Union of `IntegerRelation` disjuncts — supports complement, subtract, intersect |
| `PresburgerSet` | A `PresburgerRelation` typed as a set (no domain/range) |
| `Simplex` | Tableau-based solver for feasibility, optimization, redundancy detection |
| `Matrix` / `IntMatrix` | Resizable integer matrix used for constraint storage |

### Variable Organization

Variables in an `IntegerRelation` are organized by kind:

```
  Column layout of an IntegerRelation:

  ┌──────────┬──────────┬──────────┬──────────┬──────────┐
  │  Domain  │  Range   │ Symbols  │  Locals  │ Constant │
  │  vars    │  vars    │          │ (∃-vars) │  term    │
  └──────────┴──────────┴──────────┴──────────┴──────────┘

  VarKind::Domain  — input dimensions (for relations)
  VarKind::Range   — output dimensions (for relations)
  VarKind::SetDim  — dimensions (for sets; alias for Range)
  VarKind::Symbol  — parameters (fixed but unknown)
  VarKind::Local   — existential variables (introduced by
                     addLocalFloorDiv, addLocalModulo)
```

For simple disjointness queries (like buffer index analysis), we only need
`SetDim` variables and `Local` variables. No domain/range or symbols are
required.

### Core API

**Construction:**

```cpp
auto space = PresburgerSpace::getSetSpace(/*numDims=*/3,
                                          /*numSymbols=*/0,
                                          /*numLocals=*/0);
IntegerPolyhedron poly(space);
```

**Adding constraints:**

Each constraint is a row of coefficients `[dim0, dim1, ..., local0, ..., const]`.

```cpp
// Inequality: x0 >= 0  →  [1, 0, 0, ..., 0] (coeff * vars + const >= 0)
poly.addInequality({1, 0, 0, 0});

// Equality: x0 - x1 = 0  →  [1, -1, 0, ..., 0]
poly.addEquality({1, -1, 0, 0});
```

**Modular arithmetic:**

```cpp
// Encode: result = expr mod N
// Internally creates locals q (quotient) and r (remainder) with:
//   expr = N*q + r,  0 ≤ r < N
unsigned localIdx = poly.addLocalModulo(exprCoeffs, modulus);
```

**Emptiness check:**

```cpp
bool empty = poly.isIntegerEmpty();
// Uses GCD test → Simplex → Fourier-Motzkin elimination
// Returns true if no integer point satisfies all constraints
```

### Algorithms

The library chains multiple algorithms for the emptiness check:

```
  isIntegerEmpty() algorithm pipeline:

  ┌─────────────────────┐
  │  GCD Test           │  O(1) per equality
  │  Check gcd(coeffs)  │  "Does gcd divide constant?"
  │  divides constant   │
  ├─────────┬───────────┤
  │  Pass   │  Fail     │
  │  ↓      │  → EMPTY  │
  ├─────────┴───────────┤
  │  Gaussian           │  O(n²) per variable
  │  Elimination        │  Reduce equalities
  ├─────────┬───────────┤
  │  ↓      │           │
  ├─────────┴───────────┤
  │  Simplex Solver     │  Tableau-based feasibility
  │  + Generalized      │  with integer rounding
  │  Basis Reduction    │
  ├─────────┬───────────┤
  │  Feasible│ Infeasible│
  │  ↓      │  → EMPTY  │
  ├─────────┴───────────┤
  │  Fourier-Motzkin    │  Variable elimination
  │  Elimination        │  (when needed for
  │                     │   existential projection)
  └─────────────────────┘
```

For the small constraint systems we'd build (5-10 variables, 5-10
constraints), the GCD test alone resolves most cases in nanoseconds. The
Simplex solver handles the rest in microseconds.

## 3. Applications in Membar Analysis

### 3.1 Dynamic Buffer Index Disjointness

**The Problem.**
In multi-buffered pipelines, a producer (async copy) and consumer (local
load) access different slots of the same shared memory allocation. The slot
index is dynamic — computed from a loop-carried phase counter:

```
  scf.for iter_args(%phase = 0) {
      producer:  async_copy → slot[(%phase + 2) % 3]
      consumer:  local_load ← slot[%phase % 3]
      yield %phase + 1
  }

  ┌──────────┬──────────┬──────────┐
  │  Slot 0  │  Slot 1  │  Slot 2  │
  └──────────┴──────────┴──────────┘
       ↑                      ↑
    consumer               producer
    phase % 3          (phase+2) % 3

  Can these ever be the same slot?
```

Membar analysis sees both accesses targeting the same allocation with
unknown offsets and conservatively inserts a barrier.

Note: this shows the **unified-counter** pattern (Gluon pipelines), where
both indices derive from the same `%phase` SSA value. The common
(NVIDIA-style) pipeliner generates **separate counters** (`insertIdx` /
`extractIdx`) as distinct block arguments — that IR shape is addressed in
Section 6 as a further extension.

**BufferIndexExpr (current approach).**
Pattern matching decomposes index arithmetic into `{base, offset, modulus}`
and compares structurally:

```
  Pattern matching approach:

  remsi(addi(%phase, 2), 3)
       │
       ▼
  Match remsi → mod = 3
    Match addi → offset += 2
      Match %phase → base = %phase
  Result: {base=%phase, offset=2, mod=3}

  remsi(%phase, 3)
       │
       ▼
  Match remsi → mod = 3
    Match %phase → base = %phase
  Result: {base=%phase, offset=0, mod=3}

  Same base, same mod, 0 ≠ 2 (mod 3) → disjoint
```

This works well for the common patterns but requires manually extending
the pattern matcher for each new index idiom.

**Presburger alternative.**
Instead of pattern matching, encode the disjointness question as a
constraint system and let the solver decide:

```
  Presburger approach — encode and solve:

  IR:  %w = remsi(addi(%phase, 2), 3)     ← producer index
       %r = remsi(%phase, 3)               ← consumer index

  Step 1: Walk arith ops, build constraints
  ┌──────────────────────────────────────────────────────┐
  │  Variables: phase (shared), q₁, r₁, q₂, r₂         │
  │                                                      │
  │  Producer index (r₁ = (phase + 2) mod 3):           │
  │    r₁ = (phase + 2) - 3·q₁                          │
  │    0 ≤ r₁ < 3                                        │
  │                                                      │
  │  Consumer index (r₂ = phase mod 3):                  │
  │    r₂ = phase - 3·q₂                                │
  │    0 ≤ r₂ < 3                                        │
  │                                                      │
  │  Query: can r₁ = r₂?                                │
  │    r₁ = r₂                                           │
  └──────────────────────────────────────────────────────┘

  Step 2: Solve
    Substitute r₁ = r₂:
      (phase + 2) - 3·q₁ = phase - 3·q₂
      3·(q₂ - q₁) = 2

    GCD test: gcd(3) = 3,  3 ∤ 2
    → No integer solution exists
    → Indices are ALWAYS different
    → No barrier needed
```

**Implementation sketch.**
The Presburger approach would be implemented as a new check in
`AllocationSlice::intersects()`:

```cpp
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"

using namespace mlir::presburger;

struct IndexConstraintBuilder {
  IntegerPolyhedron poly;
  DenseMap<Value, unsigned> valueToVar;
  unsigned nextVar = 0;

  IndexConstraintBuilder(unsigned numVars)
      : poly(PresburgerSpace::getSetSpace(numVars, 0, 0)) {}

  unsigned getOrCreateVar(Value v) {
    auto [it, inserted] = valueToVar.try_emplace(v, nextVar);
    if (inserted) nextVar++;
    return it->second;
  }

  // Recursively walk arith ops and add constraints
  unsigned encodeIndex(Value index) {
    if (auto constOp = index.getDefiningOp<arith::ConstantIntOp>()) {
      unsigned var = poly.appendVar(VarKind::SetDim);
      // var = constant
      SmallVector<int64_t> eq(poly.getNumCols(), 0);
      eq[var] = 1;
      eq.back() = -constOp.value();
      poly.addEquality(eq);
      return var;
    }

    if (auto addOp = index.getDefiningOp<arith::AddIOp>()) {
      unsigned lhs = encodeIndex(addOp.getLhs());
      unsigned rhs = encodeIndex(addOp.getRhs());
      unsigned result = poly.appendVar(VarKind::SetDim);
      // result = lhs + rhs
      SmallVector<int64_t> eq(poly.getNumCols(), 0);
      eq[result] = 1; eq[lhs] = -1; eq[rhs] = -1;
      poly.addEquality(eq);
      return result;
    }

    if (auto remOp = index.getDefiningOp<arith::RemSIOp>()) {
      unsigned dividend = encodeIndex(remOp.getLhs());
      if (auto modConst = remOp.getRhs()
              .getDefiningOp<arith::ConstantIntOp>()) {
        int64_t N = modConst.value();
        // Build coefficient vector for the dividend expression
        SmallVector<int64_t> dividendCoeffs(poly.getNumCols(), 0);
        dividendCoeffs[dividend] = 1;
        // addLocalModulo returns the remainder variable index
        unsigned remVar = poly.addLocalModulo(dividendCoeffs, N);
        return remVar;
      }
    }

    // Unrecognized: treat as unconstrained (opaque) variable
    return getOrCreateVar(index);
  }
};

bool areIndicesProvablyDifferent(Value idx1, Value idx2) {
  IndexConstraintBuilder builder(/*initial vars=*/0);
  unsigned v1 = builder.encodeIndex(idx1);
  unsigned v2 = builder.encodeIndex(idx2);

  // Add equality constraint: v1 = v2
  SmallVector<int64_t> eq(builder.poly.getNumCols(), 0);
  eq[v1] = 1; eq[v2] = -1;
  builder.poly.addEquality(eq);

  // If the system is empty, v1 can never equal v2
  return builder.poly.isIntegerEmpty();
}
```

**BufferIndexExpr vs Presburger.**

```
  Pattern Matching              Presburger Constraint Solving
  (BufferIndexExpr)             (IntegerPolyhedron)

  ┌─────────────────────┐      ┌─────────────────────┐
  │ Recognizes:          │      │ Handles:             │
  │  • remsi(x, N)      │      │  • Any composition   │
  │  • addi(x, C)       │      │    of linear + modulo│
  │  • select/cmpi mod   │      │    arith ops         │
  │                      │      │  • Nested modulo     │
  │ Fails on:            │      │  • Multi-step adds   │
  │  • andi(x, N-1)     │      │  • Mixed patterns    │
  │  • nested modulo     │      │  • andi (power-of-2, │
  │  • complex chains    │      │    non-negative x)   │
  │                      │      │  • Unrecognized ops  │
  │ Extend by:           │      │    → unconstrained   │
  │  Adding new pattern  │      │    → conservative    │
  │  match cases         │      │                      │
  │                      │      │ Extend by:           │
  │                      │      │  Automatic for       │
  │                      │      │  supported encodings;│
  │                      │      │  unsupported ops →   │
  │                      │      │  conservative        │
  └─────────────────────┘      └─────────────────────┘
```

| Property | BufferIndexExpr | Presburger |
|----------|:-:|:-:|
| Handles `remsi` + `addi` | Yes | Yes |
| Handles `select/cmpi` mod | Yes | Yes |
| Handles `andi(x, N-1)` (power-of-2) | No | Yes (power-of-2 N, non-negative x) |
| Handles nested `remsi(remsi(...))` | No | Yes |
| Handles `muli` + `addi` chains | No | Yes (constant factor only) |
| Extending for new patterns | Manual | Automatic for linear + modulo ops |
| Implementation size | ~110 LOC | ~200-300 LOC |
| Runtime cost | ~ns | ~µs (GCD dominates) |
| New dependencies | None | `MLIRPresburger` (already in MLIR) |

### 3.2 Multi-Dimensional Subslice Overlap

The current `AllocationSlice::intersects()` checks subslice disjointness
using static offsets from `MemDescSubsliceOp`. When subslice offsets are
dynamic (e.g., computed from a loop variable), the check falls back to
conservative overlap.

Presburger constraints can prove disjointness of multi-dimensional
subslice ranges with dynamic offsets. For example, within a single CTA,
a loop body may access two different tiles of a shared buffer — one for
the current iteration and one for the next:

```
  Two subslices of a shared buffer within one CTA:

  ┌─────────────────────────────────────────────┐
  │  shared buffer: memdesc<4x128xf16>          │
  │  ┌────────┬────────┬────────┬────────┐      │
  │  │ Tile 0 │ Tile 1 │ Tile 2 │ Tile 3 │      │
  │  └────────┴────────┴────────┴────────┘      │
  │       ↑                ↑                    │
  │   subslice A       subslice B               │
  │   offset = i       offset = i + 1           │
  └─────────────────────────────────────────────┘

  Constraint system:
    A_offset = i,          0 ≤ i < 4
    B_offset = i + 1,      0 ≤ i + 1 < 4
    Can A_offset = B_offset?
      i = i + 1 → 0 = 1 → contradiction → EMPTY → disjoint

  For modular wrap:
    A_offset = i % 4,  B_offset = (i + 2) % 4
    → same GCD test as buffer index disjointness
```

### 3.3 Warp-Specialized Pipeline Buffers

> **Note**: The basic warp-local access problem (Problem 2 in the doc set)
> has been solved via `warpsPerCTA` comparison with a bijection argument
> (commit [`df6d5be`](https://github.com/triton-lang/triton/commit/df6d5be2206ec6f32cf47116d23f3b6235873bfe)).
> That approach compares static encoding metadata and works for all current
> cases. The Presburger formulation below would be relevant for more complex
> warp-partition patterns where the partition depends on dynamic values or
> non-trivial address arithmetic — e.g., warp-specialized pipelines with
> asymmetric producer/consumer partitioning.

In warp-specialized pipelines (e.g., Gluon tutorial `08-warp-specialization`),
different warps may access different regions of shared memory as part of a
producer-consumer handoff. Presburger constraints can formalize the
warp-partition analysis:

```
  Warp-specialized pipeline:

  ┌────────────────────────────────┐
  │ Warp 0 (producer):            │
  │   writes to smem[warp_id * S] │
  │   where S = per-warp slice    │
  │                               │
  │ Warp 1 (consumer):            │
  │   reads from smem[warp_id * S]│
  └────────────────────────────────┘

  Query: can producer_addr = consumer_addr
         when producer_warp ≠ consumer_warp?

  Presburger encoding:
    addr_p = warp_p * S + offset_p,  0 ≤ offset_p < S
    addr_c = warp_c * S + offset_c,  0 ≤ offset_c < S
    addr_p = addr_c

  Note: Presburger has no ≠ primitive. Encode warp_p ≠ warp_c
  as a disjunction (two separate polyhedra via PresburgerSet):
    Case 1: warp_p - warp_c ≥ 1
    Case 2: warp_c - warp_p ≥ 1
  Both must be empty for disjointness.

  Solve (Case 1, warp_p > warp_c):
    S * (warp_p - warp_c) = offset_c - offset_p
    warp_p - warp_c ≥ 1  →  |S * (warp_p - warp_c)| ≥ S
    0 ≤ offset_p, offset_c < S  →  |offset_c - offset_p| < S
    → S ≤ |offset_c - offset_p| < S → contradiction → EMPTY
  Case 2 is symmetric → also EMPTY → disjoint
```

### 3.4 Circular Buffer Wrap-Around Safety

When a pipelined loop uses circular buffering with non-trivial stage
distances, correctness requires that the producer and consumer never
access the same physical slot. With more than 2 buffers and variable
stage offsets, the disjointness proof involves modular arithmetic over
multiple pipeline stages.

```
  4-buffer pipeline with stage distance 3:

  ┌────────┬────────┬────────┬────────┐
  │ Slot 0 │ Slot 1 │ Slot 2 │ Slot 3 │
  └────────┴────────┴────────┴────────┘

  Producer:  slot[(phase + 3) % 4]
  Consumer:  slot[phase % 4]

  Presburger query: can (phase + 3) mod 4 = phase mod 4?
    r₁ = phase + 3 - 4·q₁,  0 ≤ r₁ < 4
    r₂ = phase - 4·q₂,      0 ≤ r₂ < 4
    r₁ = r₂
    → 4·(q₂ - q₁) = 3
    → gcd(4) = 4,  4 ∤ 3 → EMPTY → always disjoint

  What about stage distance 2 with 4 buffers?
  Producer:  slot[(phase + 2) % 4]
  Consumer:  slot[phase % 4]
    → 4·(q₂ - q₁) = 2
    → gcd(4) = 4,  4 ∤ 2 → EMPTY → always disjoint

  What about stage distance 4 with 4 buffers?  (degenerate)
  Producer:  slot[(phase + 4) % 4]
  Consumer:  slot[phase % 4]
    → 4·(q₂ - q₁) = 4
    → gcd(4) = 4,  4 | 4 → NOT EMPTY → may alias!
    → Barrier required (correct: offset 4 ≡ 0 mod 4)
```

This generalizes: given a shared phase counter, Presburger automatically
determines whether a given `(numBuffers, stageDistance)` pair guarantees
disjointness, without hard-coding the arithmetic. When the pipeliner uses
separate counters instead of a shared phase, the loop-carried extension
(Section 6) is needed to establish the relationship between them.

### 3.5 Gluon: Complex Index Patterns

Gluon gives users full control over buffer indexing. Users may write index
computations that don't match the patterns recognized by `BufferIndexExpr`.
Presburger handles these automatically.

**Example 1: Power-of-2 modulo via bitwise AND**

Gluon users optimizing for power-of-2 buffer counts may use `& (N-1)`
instead of `% N`:

```python
# Gluon kernel with power-of-2 buffering
NUM_BUFFERS = 4  # power of 2

for i in range(num_iters):
    read_slot  = alloc.index(phase & (NUM_BUFFERS - 1))       # phase & 3
    write_slot = alloc.index((phase + 2) & (NUM_BUFFERS - 1)) # (phase+2) & 3
    data = read_slot.load(layout=reg_layout)
    tok = amd.async_copy(src_ptr, write_slot)
    phase += 1
```

`BufferIndexExpr` does not recognize `arith.andi`. Presburger can encode
`x & (N-1)` as `x mod N` when N is a power of 2:

```
  Encoding x & (N-1) for N = 2^k:

  x & (N-1) = x mod N   (for non-negative x and power-of-2 N)

  Presburger: use addLocalModulo(x, N) directly
  → Same GCD test applies
  → Automatically proven disjoint
```

**Example 2: Multi-level indexing**

A Gluon kernel might use nested buffer structures:

```python
# Gluon kernel with two-level buffering
# Outer: double buffer for pipeline stages
# Inner: partition across K tiles
outer_idx = phase % 2
inner_idx = (k_tile + offset) % NUM_K_TILES
slot = alloc.index(outer_idx * NUM_K_TILES + inner_idx)
```

This produces `arith` IR like:

```mlir
%outer = arith.remsi %phase, %c2
%inner = arith.remsi %k_shifted, %c4
%prod  = arith.muli %outer, %c4
%idx   = arith.addi %prod, %inner
```

`BufferIndexExpr` cannot handle `muli`. Presburger can encode this because
one factor (`NUM_K_TILES`) is a compile-time constant, making the
multiplication linear:

```
  Constraint system for multi-level index:

  Variables: phase, k_tile, offset, outer, q_o, inner, q_i, prod, idx

  outer = phase - 2·q_o,                0 ≤ outer < 2
  inner = (k_tile + offset) - 4·q_i,    0 ≤ inner < 4
  prod = 4·outer               ← linear (constant factor)
  idx = prod + inner

  (If both factors were dynamic, the multiplication would be
  non-linear and the constraint builder would treat the result
  as unconstrained — falling back to conservative.)
```

**Example 3: Conditional buffer selection**

```python
# Gluon kernel with conditional buffering
if use_fast_path:
    slot = alloc.index(phase % NUM_BUFFERS)
else:
    slot = alloc.index((phase + 1) % NUM_BUFFERS)
```

This produces a `select` with different modular offsets. Presburger
handles this as a union of two constraint systems (via `PresburgerSet`):

```
  For each branch:
    Fast path:  idx = phase mod N,  offset = 0
    Slow path:  idx = (phase+1) mod N,  offset = 1

  Both share the same base and modulus.
  PresburgerSet can represent the union and check disjointness
  against a counterpart index across both cases.
```

## 4. Other Uses: Pipeliner Verification

The Presburger library can also serve as a **verification tool** for the
pipeliner, independent of membar. After the pipeliner creates multi-buffered
loops, a verification pass could:

1. Extract producer and consumer `MemDescIndexOp` indices.
2. Build a Presburger constraint system.
3. Assert that the system is empty (indices are always disjoint).
4. Emit a diagnostic if the assertion fails.

This provides a safety net independent of the pipeliner's correctness:

```
  Pipeliner verification flow:

  ┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
  │  Pipeliner   │────▶│  Presburger      │────▶│  Membar      │
  │  creates     │     │  Verification    │     │  Analysis    │
  │  multi-buf   │     │  (assert disjoint│     │  (skip       │
  │  loop        │     │   or emit diag)  │     │   barrier)   │
  └──────────────┘     └──────────────────┘     └──────────────┘
```

Unlike the membar applications in section 3, this is not about deciding
whether to insert a barrier — it is about catching pipeliner bugs that
would produce overlapping buffer accesses. It could be enabled under a
debug/assert flag without affecting production compilation.

## 5. Integration with Triton

### Current State

Triton does **not** currently use the Presburger library. The CMake target
`TritonAnalysis` links `MLIRAnalysis` but does not explicitly link
`MLIRPresburger`. The Presburger library is part of the MLIR distribution
that Triton builds against (pinned at the LLVM commit in
`cmake/llvm-hash.txt`), so it is available — just not used.

### What Would Change

To integrate Presburger-based disjointness checking:

1. **CMake**: Add `MLIRPresburger` (or the appropriate MLIR target) to
   `lib/Analysis/CMakeLists.txt`.

2. **Header**: Include `mlir/Analysis/Presburger/IntegerRelation.h` in
   `Membar.cpp`.

3. **Implementation**: Add an `areIndicesProvablyDifferent()` function that
   builds constraints from `arith` op chains and calls `isIntegerEmpty()`.

4. **Integration point**: In `AllocationSlice::intersects()` (or a
   `MembarSliceFilterFn`), when two accesses share the same allocation but
   have dynamic `MemDescIndexOp` indices, call the Presburger check before
   falling back to conservative overlap.

```
  Integration in AllocationSlice::intersects():

  intersects(other):
    if intervals don't overlap → false (existing)
    if subslice offsets known → static check (existing)
    if both have MemDescIndexOp indices:
      ┌───────────────────────────────────────┐
      │  NEW: Presburger disjointness check   │
      │                                       │
      │  Build constraints from arith ops     │
      │  Add equality: idx1 = idx2            │
      │  Call isIntegerEmpty()                │
      │  If empty → return false (disjoint)   │
      └───────────────────────────────────────┘
    return true (conservative)
```

### Adoption: Replace, Not Layer

Presburger handles every pattern that `BufferIndexExpr` handles (`remsi`,
`addi`, `select/cmpi`) and more (`andi`, nested modulo, `muli` by
constant). Having both at runtime is technically redundant — Presburger is
a strict superset. The choice is between the two, not a layered
combination:

| | BufferIndexExpr | Presburger |
|---|---|---|
| **When to choose** | Sufficient for current patterns, minimal dependency | Need broader pattern coverage or future-proofing |
| **Trade-off** | Simpler (~110 LOC), no build dep, ~ns | More general (~200-300 LOC), `MLIRPresburger` dep, ~µs |

If the patterns emitted by the pipeliner and Gluon remain limited to
`remsi` / `addi` / `select+cmpi`, `BufferIndexExpr` is the simpler
choice. If more complex index idioms emerge (power-of-2 bitwise AND,
multi-level indexing, nested modulo), switching to Presburger avoids
ongoing pattern-matcher maintenance.

## 6. Further Extension: Loop-Carried Induction Variables

The initial Presburger design (Section 3.1) builds constraints by walking
the arith op DAG backwards from each `MemDescIndexOp` index. This works
when both producer and consumer indices derive from the **same SSA value**
(e.g., a single `%phase` counter in Gluon pipelines).

The common (NVIDIA-style) pipeliner generates a different IR shape: two
**separate loop-carried block arguments** (`%insertIdx` and `%extractIdx`)
with no shared SSA ancestor. Since block arguments have no defining op, the
constraint builder treats them as unrelated unconstrained variables — and
the disjointness query trivially says "may alias."

```
  Common pipeliner IR:

  scf.for iter_args(%insertIdx = C_insert, %extractIdx = C_extract) {
      %write = ttg.memdesc_index %alloc[%insertIdx]
      %read  = ttg.memdesc_index %alloc[%extractIdx]

      %nextInsert  = incrementModulo(%insertIdx, N)
      %nextExtract = incrementModulo(%extractIdx, N)
      yield %nextInsert, %nextExtract
  }

  Initial design sees:
    %insertIdx  → block arg → unconstrained variable x
    %extractIdx → block arg → unconstrained variable y
    Query: can x = y?  → trivially yes → conservative barrier
```

### Why Presburger Can Handle This (With Extension)

The mathematical relationship between the two counters is:

- Both start at known constants (`C_insert`, `C_extract`)
- Both advance by the same recurrence: `(x + 1) % N` per iteration
- At iteration `i`: `insertIdx = (C_insert + i) mod N`,
  `extractIdx = (C_extract + i) mod N`

The disjointness question reduces to: can `(C_insert + i) mod N =
(C_extract + i) mod N`? The `i` cancels, leaving `N | (C_insert -
C_extract)` — the same GCD test. When the stage distance is not divisible
by `N`, the counters are always different.

```
  Extended constraint builder — loop-carried analysis:

  Detect: both indices are block args of the same scf.for
          with incrementModulo recurrence and constant inits

  Introduce shared iteration variable i:
    insertIdx  = (C_insert + i)  mod N
    extractIdx = (C_extract + i) mod N

  Presburger encoding:
    r₁ = C_insert  + i - N·q₁,   0 ≤ r₁ < N
    r₂ = C_extract + i - N·q₂,   0 ≤ r₂ < N
    r₁ = r₂

  Substituting:
    N·(q₂ - q₁) = C_insert - C_extract

  GCD test: N ∤ (C_insert - C_extract)?
    → EMPTY → always disjoint → no barrier needed

  Example: 3-buffer pipeline, stage distance 2
    C_insert = 2, C_extract = 0, N = 3
    3·(q₂ - q₁) = 2,  gcd(3) = 3,  3 ∤ 2
    → always disjoint ✓
```

### What the Constraint Builder Needs

The extension requires recognizing **modular induction variables** —
loop-carried block arguments with an `incrementModulo` recurrence:

1. Detect that the index is a block argument of an `scf.for`.
2. Trace the corresponding yield operand to find the recurrence.
3. Match the `incrementModulo` pattern: `select(cmpi(x+1, N), 0, x+1)` or
   `remsi(x+1, N)`.
4. Extract the constant initial value from `iter_args`.
5. Introduce a shared iteration variable `i` and encode
   `blockArg = (init + i) mod N`.

This is itself a pattern match, but a qualitatively different one from
`BufferIndexExpr`'s approach:

- `BufferIndexExpr` matches on arbitrary **index arithmetic** chains
  (open-ended variety of arith op compositions).
- The loop-carried extension matches on **loop recurrence structure** (a
  single, well-known pipeliner idiom: `incrementModulo`).

The common pipeliner always emits exactly this shape, so the pattern is
stable.

### Relationship to BufferIndexExpr

This extension gives Presburger a capability that `BufferIndexExpr`
fundamentally cannot achieve. `BufferIndexExpr` requires a shared SSA
base — two separate block arguments will never have one. Presburger's
constraint system can relate the two variables through the shared
iteration variable `i`, a synthetic variable not present in the IR.

```
  Why BufferIndexExpr cannot handle separate counters:

  BufferIndexExpr decomposes each index independently:
    %insertIdx  → {base=%insertIdx,  offset=0, mod=N}
    %extractIdx → {base=%extractIdx, offset=0, mod=N}
                         ↑ different bases → cannot compare

  Presburger introduces a shared variable i:
    %insertIdx  = (C_insert  + i) mod N  ─┐
    %extractIdx = (C_extract + i) mod N  ─┤ shared i
                                           ↓
    Relationship between the two is captured
    → GCD test resolves disjointness
```

### Not Part of Initial Design

This extension is not needed for the initial Presburger integration, which
targets the same Gluon-generated unified-counter IR that `BufferIndexExpr`
already handles. The initial value is in covering more complex arith op
chains (Section 3.5) and providing a fallback for patterns that
`BufferIndexExpr` misses.

The loop-carried extension becomes relevant when we want to handle the
common pipeliner's separate-counter IR without requiring pipeliner-side
changes. It is a natural second step after the initial constraint builder
is in place.

## 7. Limitations

1. **No variable-variable multiplication.** Presburger arithmetic does not
   support multiplying two variables (`x * y`). When an index involves
   multiplication of two dynamic values, the constraint builder must treat
   it as an unconstrained variable. Multiplication by a **constant** is
   fine (it's just repeated addition).

2. **Existential variable growth.** Each `addLocalModulo` introduces 2
   local variables and 3 constraints. Deeply nested modular expressions
   grow the system, though in practice pipeline indices are shallow (1-2
   levels).

3. **Runtime cost.** While negligible for small systems (~µs), the Simplex
   solver's worst case is exponential. For the index patterns in practice
   (≤10 variables, ≤15 constraints), this is not a concern.

4. **Build dependency.** Requires linking `MLIRPresburger`. This is a
   stable, upstream MLIR library with no risk of removal, but it does add
   a build-time dependency that `BufferIndexExpr` avoids.

5. **Bitwise operations.** `andi`, `ori`, `xori` are not directly
   expressible in Presburger arithmetic. `x & (N-1)` can be encoded as
   `x mod N` only when N is a power of 2 and x is non-negative. General
   bitwise operations require over-approximation or special-case handling.

## 8. Summary

| Aspect | BufferIndexExpr | With Presburger |
|--------|:-:|:-:|
| **Buffer index disjointness** | Pattern matching | Constraint solving (linear + modulo) |
| **Patterns handled** | `remsi`, `addi`, `select/cmpi` | Any linear + modular composition (not variable×variable or general bitwise) |
| **New Gluon patterns** | Must extend pattern matcher | Automatic for supported linear/modular encodings |
| **Subslice overlap** | Static offsets only | Dynamic offsets (with constraints) |
| **Separate-counter pipeliners** | Cannot handle (no shared SSA base) | Possible via loop-carried extension (Section 6) |
| **Pipeliner verification** | Not available | Automated disjointness assertion |
| **Build dependency** | None | `MLIRPresburger` (stable upstream) |
| **Engineering effort** | Done (~110 LOC) | ~200-300 LOC for constraint builder |
| **Runtime** | ~ns per check | ~µs per check (GCD dominates) |

The MLIR Presburger library is a powerful, stable, and zero-external-dependency
tool for reasoning about integer constraints in compiler analyses. For
Triton's membar analysis, it offers a principled generalization of the
pattern-matching approach — handling a broader class of linear + modular index
expressions without manual pattern extension. Its boundaries are clear:
variable-variable multiplication and general bitwise operations fall outside
the Presburger-expressible fragment and require conservative fallback. Within
that fragment, it scales automatically as Gluon users write increasingly
complex index computations.
