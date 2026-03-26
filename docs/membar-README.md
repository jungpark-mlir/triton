# Membar Analysis — Design Documents

This directory contains a series of design documents on Triton's shared memory
barrier (membar) analysis. The documents cover two orthogonal optimization
problems: eliminating false positive barriers due to dynamic buffer indexing,
and eliminating barriers when shared memory access is inherently warp-local.

## Problem Overview

Triton's membar analysis inserts CTA-wide barriers (`__syncthreads()` /
`s_barrier`) between operations that may access overlapping shared memory
regions. The analysis is conservative: when it cannot prove disjointness, it
inserts a barrier. Two concrete sources of false positives are addressed here:

1. **Dynamic buffer index disjointness** — in multi-buffered pipelines, the
   producer and consumer access different slots of the same allocation via a
   dynamic phase counter. Membar cannot distinguish the slots statically.

2. **Warp-local access patterns** — when a shared memory layout guarantees
   each warp accesses only its own disjoint partition, a CTA-wide barrier is
   unnecessary.

---

## Document Map

### Start Here: The Two Problems and Solution Comparison

**[membar-comparison.md](membar-comparison.md)**
— The entry point. Describes both problems, summarizes the two candidate
solutions for dynamic index disjointness (Symbolic Index Analysis and Buffer
Slot Coloring), and explains where things stand. Read this first.

---

### Problem 1: Dynamic Buffer Index Disjointness

These documents address false positive barriers in multi-buffered pipelines
where the buffer slot index is dynamic.

```
  memdesc<3x128x128xf16> allocation
  ┌──────────┬──────────┬──────────┐
  │  Slot 0  │  Slot 1  │  Slot 2  │
  └──────────┴──────────┴──────────┘
      ↑ consumer               ↑ producer
      phase % 3          (phase+2) % 3
      → always different, but membar can't tell
```

**[membar-dynamic-index-disjointness.md](membar-dynamic-index-disjointness.md)**
— Full design and implementation of the primary solution: `BufferIndexExpr`,
a symbolic index decomposition that pattern-matches `arith.remsi` /
`arith.addi` / `select/cmpi` modular idioms to prove slot disjointness.
Covers modulus tracking, loop-carried dependency handling, IR generation
requirements, and lit test results.

**[membar-buffer-slot-coloring.md](membar-buffer-slot-coloring.md)**
— Full design of the alternative solution: stamping `MemDescIndexOp` with a
`buffer_color` integer attribute so that the AMD membar filter treats different
colors as disjoint. Covers attribute placement rationale, pipeliner integration,
and Gluon API design.

**[membar-integer-range-analysis-evaluation.md](membar-integer-range-analysis-evaluation.md)**
— Evaluates whether Triton's existing `TritonIntegerRangeAnalysis` (AMD
backend) could solve this problem instead. Conclusion: it cannot, because it
is a non-relational interval domain that cannot express `A ≠ B`. Explains why
`BufferIndexExpr` (a relational, symbolic approach) is necessary.

**[membar-disjointness-approaches.md](membar-disjointness-approaches.md)**
— Broad survey of all approaches to this problem: integer range analysis,
LLVM SCEV-AA, MLIR GPU barrier elimination, relational abstract domains,
`BufferIndexExpr`, Presburger constraint solving, and ISL/Polyhedral.
Compares correctness, engineering effort, runtime cost, and maintenance risk.
Use this as a reference when evaluating future alternatives.

**[membar-presburger.md](membar-presburger.md)**
— Deep dive into MLIR's standalone Presburger arithmetic library
(`mlir/Analysis/Presburger/`) as a future-proof generalization of
`BufferIndexExpr`. Covers the library's API, algorithms (GCD test, Simplex,
Fourier-Motzkin), and how it can handle index patterns that `BufferIndexExpr`
cannot (power-of-2 bitwise AND, multi-level indexing, nested modulo).
Also discusses applications beyond the core problem: multi-dimensional
subslice overlap, warp-specialized buffer partitioning, and pipeliner
verification.

---

### Problem 2: Warp-Local Shared Memory Access

**[membar-warp-local-access.md](membar-warp-local-access.md)**
— Design and implementation of warp-local barrier suppression, covering two
sub-problems:

- **Problem 2-1: Write/read op pair barriers.** When a writer (TDM copy,
  async_copy, local_store) and reader (local_load) both distribute warps
  identically, each warp's byte-address partition is disjoint (one-to-one
  mapping), so no CTA barrier is needed. **Implemented** via `warpsPerCTA`
  comparison (commit [`df6d5be`](https://github.com/triton-lang/triton/commit/df6d5be2206ec6f32cf47116d23f3b6235873bfe))
  for `AsyncTDMCopyGlobalToLocalOp` → `local_load`; extends naturally to
  `AsyncCopyGlobalToLocalOp` and `local_store`/`local_load`.

- **Problem 2-2: `MemWaitOpTrait` unconditional barrier.** A separate
  codepath in membar unconditionally inserts a CTA barrier after
  `async_wait`, bypassing `isIntersected` entirely. `async_wait` is not
  a read or write — the real dependency is a RAW hazard between
  `async_copy` (writer) and `local_load` (reader). Proposed two-step
  fix: (1) refactor to let `isIntersected` handle the RAW dependency
  (behavior-preserving for all backends), then (2) add AMD filter to
  suppress when `warpsPerCTA` matches.

The originally proposed GF(2) linear independence test is documented as a
design alternative but was not implemented — the `warpsPerCTA` comparison is
simpler, handles padded layouts, and covers all practical cases.

---

## Reading Order

> **Note:** The implemented `BufferIndexExpr` approach currently targets
> Gluon-generated pipeliner IR, which uses a unified phase counter. The
> common (NVIDIA-style) pipeliner produces a different IR shape (separate
> `insertIdx` / `extractIdx` counters) that requires Buffer Coloring or
> pipeliner-side changes. See the IR Shape Compatibility section in the
> comparison doc for details.

**For a quick overview of both problems:**
→ [membar-comparison.md](membar-comparison.md)

**To understand the implemented solution (Problem 1):**
→ [membar-dynamic-index-disjointness.md](membar-dynamic-index-disjointness.md)

**To understand why other approaches (range analysis, SCEV, etc.) don't work:**
→ [membar-integer-range-analysis-evaluation.md](membar-integer-range-analysis-evaluation.md)
→ [membar-disjointness-approaches.md](membar-disjointness-approaches.md)

**To evaluate or extend to a more general solution:**
→ [membar-presburger.md](membar-presburger.md)

**To understand the alternative (attribute-based) approach:**
→ [membar-buffer-slot-coloring.md](membar-buffer-slot-coloring.md)

**To understand the warp-local barrier elimination (implemented):**
→ [membar-warp-local-access.md](membar-warp-local-access.md)

---

## Relationship Between Documents

```
  membar-comparison.md  ──────────────────────────────────┐
  (entry point, both problems, solution comparison)        │
         │                                                 │
         ├── Problem 1: Dynamic Index Disjointness         │
         │       │                                         │
         │       ├── membar-dynamic-index-disjointness.md  │  ← implementation
         │       │   (BufferIndexExpr — primary solution)  │
         │       │                                         │
         │       ├── membar-buffer-slot-coloring.md        │  ← alternative
         │       │   (color attribute — AMD-specific)      │
         │       │                                         │
         │       ├── membar-integer-range-analysis-        │  ← ruled out
         │       │   evaluation.md                         │
         │       │                                         │
         │       ├── membar-disjointness-approaches.md     │  ← survey
         │       │   (all approaches compared)             │
         │       │                                         │
         │       └── membar-presburger.md                  │  ← future path
         │           (Presburger as upgrade / extension)   │
         │                                                 │
         └── Problem 2: Warp-Local Access ─────────────────┘
                 │
                 └── membar-warp-local-access.md  ← IMPLEMENTED
                     (warpsPerCTA comparison + one-to-one address mapping)
```
