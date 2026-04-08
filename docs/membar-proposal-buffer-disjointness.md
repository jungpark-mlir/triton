# Proposal: Buffer Index Disjointness for Membar Analysis

## Problem

In multi-buffered pipelined kernels, producer and consumer access
different slots of the same shared memory allocation via a dynamic
index. Membar cannot distinguish the slots and inserts a false-positive
CTA barrier on every iteration.

```
memdesc<3x128x128xf16> allocation
  +----------+----------+----------+
  |  Slot 0  |  Slot 1  |  Slot 2  |
  +----------+----------+----------+
      ^ consumer               ^ producer
      phase % 3          (phase+2) % 3

  Always different slots, but membar sees same allocation
  --> barrier inserted (false positive)
```

This affects every multi-buffered kernel — FA MQA decode, GEMM
pipelined loops, and Gluon user-written pipelines. The barrier
stalls all warps on every pipeline iteration with no benefit.

## Options

### Option A: BufferIndexExpr (Symbolic Pattern Matching)

Pattern-match the index arithmetic into a canonical form
`{base, offset, modulus}`. Two indices are provably disjoint when
they share the same SSA base, same modulus, and different offsets.

```
slot[remsi(phase, 3)]       --> {base=%phase, offset=0, mod=3}
slot[remsi(phase + 2, 3)]   --> {base=%phase, offset=2, mod=3}
                                     same       0!=2      same
                                        --> provably disjoint
```

**Where it lives:** Core `Membar.cpp` — all backends benefit.

**Recognized patterns:** `arith.remsi`, `arith.addi`, `select/cmpi`
conditional wrapping. Unrecognized patterns fall through to
conservative (barrier inserted).

**Effort:** ~200 LOC. `BufferIndexExpr` struct + `analyzeBufferIndex()`
pattern matcher + `isProvablyDifferentFrom()` comparison +
`AllocationSlice` additions for loop-carried tracking.

**Tested:** Yes — local testing confirms correct barrier suppression
on FA MQA decode and standard Gluon pipeline patterns.

**Limitations:**
- Requires both indices to derive from the **same SSA counter** with
  different constant offsets. This is what Gluon produces (unified
  `%phase`), but the common NVIDIA-style pipeliner uses **separate
  loop-carried counters** (`insertIdx` / `extractIdx`) — different
  SSA bases that `BufferIndexExpr` cannot relate.
- Each new index idiom needs a new pattern in `analyzeBufferIndex`.
  Current patterns cover what Gluon and the pipeliner's
  `createIncrementModulo` produce today.

### Option B: Presburger Constraint Solving

Encode the index arithmetic as linear constraints over integers
and ask MLIR's Presburger solver whether `idx_a = idx_b` is
satisfiable.

```
// Encode:
//   idx_a = (phase + 2) mod 3  -->  exists q1: idx_a = phase + 2 - 3*q1
//   idx_b = phase mod 3        -->  exists q2: idx_b = phase - 3*q2
//   0 <= idx_a <= 2,  0 <= idx_b <= 2
//
// Query: is {idx_a = idx_b} satisfiable?
//
// Presburger solver: NO --> provably disjoint
```

**Where it lives:** Core `Membar.cpp` — all backends benefit.
Uses `mlir/Analysis/Presburger/IntegerRelation` (standalone library,
no IR dependency).

**How the encoder works:** Walk the arith SSA def-use chain from the
`MemDescIndexOp` index value. For each op, emit constraints:

| arith op | Constraint |
|----------|-----------|
| `arith.addi %x, C` | `result = x + C` (equality) |
| `arith.remsi %x, N` | `exists q: result = x - N*q, 0 <= result < N` |
| `arith.muli %x, C` | `result = C*x` (equality) |
| block argument | unconstrained `SetDim` variable |
| unrecognized op | unconstrained `SetDim` variable (conservative) |

Then add the query constraint `idx_a = idx_b` and call
`isIntegerEmpty()`. If empty, the indices are provably different.

**Effort:** ~300 LOC. Encoder (per-op translation rules) +
`IntegerPolyhedron` construction + `isIntegerEmpty()` call +
integration into `AllocationSlice::intersects()`. Link target
`MLIRPresburger` (already in LLVM/MLIR, no external dependency).

**Limitations:**
- Same IR shape constraint as BufferIndexExpr: requires both
  indices to derive from the **same SSA counter**. The solver is
  more powerful than pattern matching, but it still operates on
  SSA def-use chains — if the two indices have unrelated SSA
  bases, the solver treats them as unconstrained and cannot prove
  disjointness.
- The Presburger library is ~12k LOC internally (Simplex, FM
  elimination, branch-and-bound). It is a well-tested MLIR
  component, but internal debugging is non-trivial. However, our
  interaction surface is small (~5 API calls) and the constraint
  matrices are tiny (4-8 rows) — verifiable by `dump()`.

**Tested:** Yes — local testing confirms correct results on FA MQA
decode and standard Gluon pipeline patterns. No performance issues
observed (solver time negligible for small constraint systems).

### Option C: Buffer Slot Coloring (Attribute-Based)

Tag each `MemDescIndexOp` with a `buffer_color` integer attribute.
The membar filter treats different colors as disjoint — no
arithmetic analysis needed.

```mlir
%read  = ttg.memdesc_index %alloc[%phase] {buffer_color = 0}
%write = ttg.memdesc_index %alloc[%next]  {buffer_color = 1}
// 0 != 1 --> disjoint
```

**Where it lives:** Backend filter (AMD `membarFilter`). Requires
the code generator (pipeliner or Gluon runtime) to assign colors.

**How it works:** The producer stamps `buffer_color` on each
`MemDescIndexOp`. The filter compares integers — trivial.

**Effort:** ~150 LOC. Attribute definition + filter clause +
pipeliner/Gluon integration to assign colors.

**Limitations:**
- Correctness depends on correct color assignment by the producer.
  If the pipeliner assigns wrong colors (e.g., single-buffer case),
  the filter silently suppresses a real hazard. No construction-time
  safety — unlike BufferIndexExpr/Presburger where same offsets are
  never reported as disjoint.
- AMD-specific (lives in backend filter, not core membar).
- Attributes must survive lowering passes. If a pass recreates the
  op without copying attributes, the color is lost.
- Requires a new Gluon API (`colored_memdesc_index` or similar).

**Advantage over A and B:** Works with **any IR shape**, including
the common pipeliner's separate `insertIdx`/`extractIdx` counters.
The filter doesn't inspect the index arithmetic at all — it only
compares the attribute.

## IR Shape Compatibility

The critical distinction between the options is which pipeliner
IR shapes they support:

**Gluon pipelines** use a unified `%phase` counter:

```mlir
%r_idx = arith.remsi %phase, %c3
%read  = ttg.memdesc_index %alloc[%r_idx]
%w_off = arith.addi %phase, %c2
%w_idx = arith.remsi %w_off, %c3
%write = ttg.memdesc_index %alloc[%w_idx]
```

Both indices derive from `%phase` → A, B, and C all work.

**Common (NVIDIA-style) pipeliner** uses separate counters:

```mlir
// Producer stage:
%insertIdx = ... // loop-carried, increments mod N
%write = ttg.memdesc_index %alloc[%insertIdx]

// Consumer stage:
%extractIdx = ... // separate loop-carried, increments mod N
%read = ttg.memdesc_index %alloc[%extractIdx]
```

Different SSA bases → A and B cannot relate them. C works
(color on the op, doesn't inspect the index).

| IR Shape | A: BufferIndexExpr | B: Presburger | C: Coloring |
|----------|:--:|:--:|:--:|
| Unified counter (Gluon) | Yes | Yes | Yes |
| Separate counters (common pipeliner) | No | No | Yes |
| Arbitrary index computation (Gluon user code) | Pattern-dependent | Yes (if arith) | Yes |

## Comparison

| | A: BufferIndexExpr | B: Presburger | C: Coloring |
|---|---|---|---|
| **Where** | Core membar | Core membar | Backend filter |
| **Mechanism** | Pattern matching | Constraint solving | Attribute comparison |
| **All backends** | Yes | Yes | No (AMD) |
| **Effort** | ~200 LOC | ~300 LOC | ~150 LOC |
| **Safety** | By construction | By construction | Depends on producer |
| **Gluon unified counter** | Yes | Yes | Yes |
| **Common pipeliner** | No | No | Yes |
| **Pattern coverage** | `remsi`, `addi`, `select/cmpi` | Any `arith` op chain | Any (attribute) |
| **New patterns** | New matcher needed | Encoder handles automatically | N/A |
| **Dependency** | None | `MLIRPresburger` link | Attribute + API |
| **Debuggability** | Trivial (our code) | Good (`dump()` constraints) | Trivial (integer comparison) |
| **Runtime cost** | O(1) per comparison | O(1) for small systems | O(1) |
| **Tested locally** | Yes | Yes | Not yet |

## A vs B: BufferIndexExpr vs Presburger

Since both A and B cover the same IR shapes (unified counter), the
choice between them is about **simplicity vs generality**:

**Choose A if:**
- The recognized patterns (`remsi`, `addi`, `select/cmpi`) are
  sufficient for all foreseeable use cases.
- You want zero new dependencies and trivially debuggable code.
- You accept that new index patterns require new matchers.

**Choose B if:**
- You want to handle any `arith` op chain without writing pattern
  matchers for each one (e.g., `andi` masking, nested modulo,
  multi-level indexing).
- You're comfortable with the Presburger library as a dependency
  (well-tested MLIR component, but internal complexity is high).
- You want a solution that doesn't need maintenance when the
  pipeliner or Gluon changes how it computes indices.

**Key observation:** B subsumes A. Every pattern A recognizes, B
handles. B also handles patterns A doesn't (power-of-2 `andi`,
nested modulo, multi-level indexing). The extra ~100 LOC buys
strictly broader coverage and no per-pattern maintenance.

## Decision Points

1. **For Gluon (unified counter): A or B?**
   Both work. B is more general and already tested. A is simpler
   and has zero dependencies. If the team is comfortable with
   Presburger as a dependency, B is the stronger choice.

2. **For common pipeliner (separate counters): C or pipeliner change?**
   Neither A nor B can help here. Two options:
   - **Add C** as a complement — the pipeliner stamps colors, the
     filter compares them. Works with the current IR shape.
   - **Change the pipeliner** to use a unified counter (like Gluon)
     — then A or B handles it. This is a larger change but avoids
     attribute-based machinery.

3. **Can A/B and C coexist?**
   Yes. They operate at different layers — A/B in core
   `AllocationSlice::intersects()`, C in the backend filter. If
   A/B proves disjointness, the filter is never consulted. If A/B
   can't (unrecognized pattern or different SSA bases), C provides
   a fallback.
