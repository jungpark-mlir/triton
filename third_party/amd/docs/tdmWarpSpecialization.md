# TDM Warp Specialization

Author: Jungwook Park

Status: Early design exploration — not yet implemented.

## Table of Contents

- [Motivation](#motivation)
- [Background](#background)
  - [Warp-pipeline group split](#warp-pipeline-group-split)
  - [TDM copy and async\_wait coupling](#tdm-copy-and-async_wait-coupling)
  - [async\_wait is count-based, not handle-based](#async_wait-is-count-based-not-handle-based)
- [Challenges](#challenges)
  - [Challenge 1: Co-predication](#challenge-1-co-predication)
  - [Challenge 2: Count-based copy-to-wait association](#challenge-2-count-based-copy-to-wait-association)
  - [Challenge 3: Descriptor construction separation](#challenge-3-descriptor-construction-separation)
- [Solution: Narrowed Warp Mapping](#solution-narrowed-warp-mapping)
  - [What changes](#what-changes)
  - [Constraint: no mixing within an epoch](#constraint-no-mixing-within-an-epoch)
  - [Compiler actions](#compiler-actions)
- [Triggering: `tdm_specialize` on Descriptor](#triggering-tdm_specialize-on-descriptor)
  - [Gluon API](#gluon-api)
  - [IR representation](#ir-representation)
  - [Compiler transform](#compiler-transform)
- [Epoch-Based Analysis](#epoch-based-analysis)
  - [Epoch boundaries and algorithm](#epoch-boundaries-and-algorithm)
  - [Properties](#properties)
- [Phase-Shift Synergy](#phase-shift-synergy)
- [Alternative Triggering Mechanisms](#alternative-triggering-mechanisms)
  - [Option A: Stage-level attribute with paired copies](#option-a-stage-level-attribute-with-paired-copies)
  - [Option B: (stride, count) on ops](#option-b-stride-count-on-ops)
  - [Option C: warpsPerCTA on ops](#option-c-warpspercta-on-ops)
  - [Option D: Compiler flag only](#option-d-compiler-flag-only)
  - [Option E: Auto-detection](#option-e-auto-detection)
  - [Option F: Warp-level programming](#option-f-warp-level-programming)
- [Summary](#summary)

---

## Motivation

In warp-pipelined kernels on gfx1250, TDM (Tensor Data Mover) copies transfer data from global memory to LDS asynchronously. In the current design, **all 8 warps** in a workgroup issue identical TDM copies — each warp copies its sub-tile of the block. However, the global memory fetch latency is the same regardless of how many warps issue the request.

The key observation: if only the **upper warp group** (warps {0,1,2,3} — the group that starts one pipeline stage ahead) issues TDM copies, the global memory request is invoked **earlier** relative to when the lower group needs the data. The phase shift provides natural latency tolerance — by the time the lower group reaches the point where it needs the data, the upper group has already waited for the copy to complete and the data is in LDS.

This "TDM warp specialization" aims to improve latency hiding without changing the programming model.

---

## Background

### Warp-pipeline group split

The `ConvertWarpPipeline` pass splits a workgroup of 8 warps into two groups based on `threadIdx.x`:

```
threadsPerPipelineGroup = warpSize × 4

Group assignment:
  warpIDX = threadIdx.x / threadsPerPipelineGroup
  warpLow  = (warpIDX == 0)   →  warps {0, 1, 2, 3}  →  threads 0–127
  warpHigh = (warpIDX != 0)   →  warps {4, 5, 6, 7}  →  threads 128–255
```

With round-robin SIMD assignment, each group has exactly one warp per SIMD:

| SIMD | warpLow group | warpHigh group |
|------|---------------|----------------|
| 0    | warp 0        | warp 4         |
| 1    | warp 1        | warp 5         |
| 2    | warp 2        | warp 6         |
| 3    | warp 3        | warp 7         |

The `warpLow` group enters the loop first (one stage ahead). The `warpHigh` group is deferred by a `CondBarrier`.

### TDM copy and async_wait coupling

TDM copies and `async_wait` are **inseparable** at the hardware level:

- `tdm_copy` issues an asynchronous global→LDS transfer for the calling warp's sub-tile.
- `async_wait(N)` stalls the calling warp until at most N of **that warp's** outstanding TDM copies remain in flight.

A warp can only wait on its own outstanding copies. Warp 0 cannot `async_wait` for a copy issued by warp 4.

### async_wait is count-based, not handle-based

`async_wait(N)` does not name a specific copy. It operates on a per-warp FIFO counter:

```
tdm_copy(A)        # FIFO: [A]           outstanding = 1
tdm_copy(B)        # FIFO: [A, B]        outstanding = 2
tdm_copy(scaleA)   # FIFO: [A, B, sA]    outstanding = 3
tdm_copy(scaleB)   # FIFO: [A, B, sA, sB] outstanding = 4

async_wait(2)      # wait until ≤2 remain → A, B done (FIFO order)
... use A, B ...

async_wait(0)      # wait until 0 remain → all done
... use scaleA, scaleB ...
```

The association between `async_wait(2)` and "copies A and B are done" is implicit — it relies on FIFO ordering and the count. There is no handle or token mechanism to bind a specific wait to a specific copy.

---

## Challenges

Three hazards had to be resolved to arrive at a viable design. These same hazards are what prevented simpler alternatives from working.

### Challenge 1: Co-predication

Any mechanism that restricts TDM copies to a subset of warps must also restrict `async_wait` to the **same** subset. A warp can only wait on its own outstanding copies.

This rules out approaches that annotate only the copy instruction or only the stage containing the copy (since `async_wait` sits **between** stages and would not inherit the annotation). Both the copy and the wait must be predicated together.

### Challenge 2: Count-based copy-to-wait association

When a kernel has multiple TDM copies and multiple async_waits with different counts, the compiler cannot easily determine which wait corresponds to which copy. If different warp groups issue different copies, each group's FIFO counter reflects only its own copies, and the original wait counts become wrong.

**Resolution:** The [epoch-based analysis](#epoch-based-analysis) uses `async_wait(0)` (full FIFO drain) as epoch boundaries. Within each epoch, the compiler applies an **all-or-nothing** rule: either all copies are specialized and all waits are predicated, or the epoch is left untouched. This avoids any need to match individual copies to individual waits.

### Challenge 3: Descriptor construction separation

The TDM descriptor (`make_tensor_desc`) bakes sub-tile sizes, warp distribution, and offsets at creation time. The `tdm_copy` op merely executes with a pre-built descriptor.

If only 4 of 8 warps execute a copy, but the descriptor was built for 8 warps, each active warp copies only its 1/8 sub-tile — leaving half the block unwritten. The descriptor itself must be built for 4 warps to produce correct 1/4 sub-tiles.

Since `make_tensor_desc` and `tdm_copy` are separate ops, approaches that annotate only the copy op are insufficient. The specialization signal must be on the descriptor.

**Resolution:** The chosen design places `tdm_specialize=True` on `make_tensor_desc`, ensuring the descriptor is built for 4 warps from the start.

---

## Solution: Narrowed Warp Mapping

The technique behind TDM warp specialization is a **narrowed warp mapping**: only the first 4 warps are active for TDM operations, and the block is remapped so these 4 warps cover the full tile.

### What changes

With `num_warps=8`, the default mapping assigns each warp a sub-tile:

```
Default (8 warps active):
  warp 0 → sub-tile 0      warp 4 → sub-tile 4
  warp 1 → sub-tile 1      warp 5 → sub-tile 5
  warp 2 → sub-tile 2      warp 6 → sub-tile 6
  warp 3 → sub-tile 3      warp 7 → sub-tile 7
```

With specialization, the mapping narrows to 4 warps with larger sub-tiles:

```
Specialized (4 warps active):
  warp 0 → sub-tile 0 (2× larger)
  warp 1 → sub-tile 1 (2× larger)
  warp 2 → sub-tile 2 (2× larger)
  warp 3 → sub-tile 3 (2× larger)
  warp 4–7 → inactive (skip copy and wait)
```

The full block is still copied to LDS. The LDS content is identical. Only the work distribution changes.

Example with `blockShape = [256, 64]`:

| tdm_specialize | Effective numWarps | Warp distribution | Sub-tile per warp |
|----------------|-------------------|------------------|-------------------|
| False (default) | 8                | [8, 1]           | [32, 64]          |
| True            | 4                | [4, 1]           | [64, 64]          |

### Constraint: no mixing within an epoch

Specialized and non-specialized TDM copies **cannot be mixed within the same epoch** (where an epoch is the region between two `async_wait(0)` boundaries). This is a hard constraint, not a heuristic.

The reason is the per-warp FIFO counter. If a warp issues both specialized and non-specialized copies within the same epoch, its FIFO conflates copies with different sub-tile sizes and warp mappings. An intermediate `async_wait(N)` cannot correctly distinguish between them, breaking the "unchanged wait counts" property.

If the compiler detects a mix within an epoch, it should emit an error.

### Compiler actions

Given the narrowed mapping, the compiler:

1. Builds the TDM descriptor for `numWarps=4`.
2. Predicates `tdm_copy` — only warps 0–3 execute.
3. Predicates `async_wait` — only warps 0–3 wait.
4. Inserts a barrier after the wait — so warps 4–7 see the updated LDS.
5. Verifies no mixing of specialized and non-specialized copies within any epoch.

Wait counts are unchanged: each active warp issues the same number of copies as before, just with larger sub-tiles.

The narrowed mapping itself is straightforward. The design question is: **how does the programmer signal that this mapping should be applied?** The [alternatives section](#alternative-triggering-mechanisms) evaluates different triggering mechanisms. The chosen approach is `tdm_specialize=True` on the descriptor.

---

## Triggering: `tdm_specialize` on Descriptor

### Gluon API

The programmer passes `tdm_specialize=True` when creating the TDM descriptor:

```python
@triton.jit
def kernel(...):
    # Kernel launched with num_warps=8
    desc_a = tdm.make_tensor_desc(..., tdm_specialize=True)
    desc_b = tdm.make_tensor_desc(..., tdm_specialize=True)

    for k in range(0, K, BLOCK_K):
        tdm.async_wait(N)                          # compiler predicates
        with gl.amd.warp_pipeline_stage("memory"):
            tdm.copy(desc_a, buf_a)                # compiler predicates
            tdm.copy(desc_b, buf_b)                # compiler predicates
        with gl.amd.warp_pipeline_stage("compute"):
            acc = wmma(a, b, acc)                   # all warps
```

Without warp-pipelining (standalone):

```python
@triton.jit
def kernel(...):
    desc = tdm.make_tensor_desc(..., tdm_specialize=True)

    for k in range(0, K, BLOCK_K):
        tdm.copy(desc, buf)          # only warps 0–3 execute
        tdm.async_wait(0)            # only warps 0–3 wait
        # barrier ensures LDS visibility for warps 4–7
        data = buf.load(...)         # all warps read from LDS
        acc = wmma(data, ..., acc)   # all warps compute
```

When `tdm_specialize` is omitted (or `False`), the descriptor uses the module's `num_warps` — current behavior.

### IR representation

```mlir
module attributes {
  ttg.num-warps = 8
}

// Descriptor with specialization enabled
%desc_a = tt.make_tensor_desc ... { tdm.specialize = true }

// No specialization (default)
%desc_b = tt.make_tensor_desc ...
```

The `tdm.specialize` boolean attribute on the descriptor op flows naturally through the IR. At lowering time, the compiler reads it and builds the descriptor for 4 warps.

### Compiler transform

**In ConvertWarpPipeline (warp-pipelined kernels):**

The pass already computes `warpLow` and `warpHigh` predicates. It extends its logic:

```
1. For each tdm_copy in the pipelined loop:
     if tdm_copy.descriptor has tdm.specialize = true:
       mark tdm_copy as specialized
2. Apply epoch-based analysis:
     predicate specialized tdm_copy with warpLow
     predicate associated async_wait with warpLow
     verify/insert barrier for LDS visibility
```

**In general lowering (non-pipelined kernels):**

```
1. Detect tdm_copy ops whose descriptor has tdm.specialize = true.
2. Predicate with: warp_id < 4.
3. Predicate associated async_wait ops (epoch-based analysis).
4. Insert s_barrier after predicated async_wait for LDS visibility.
```

---

## Epoch-Based Analysis

### Epoch boundaries and algorithm

`async_wait(0)` is a **full FIFO drain**. The compiler splits the operation sequence at every `async_wait(0)` into **epochs**:

```
         ┌─── epoch 1 ──────────────────────┐
         │ tdm_copy(desc_spec_A) ← specialized│
         │ tdm_copy(desc_spec_B) ← specialized│
         │ async_wait(2)         ← predicate  │
         │ ... use A ...                      │
         │ async_wait(0)  ──── DRAIN ─────────┘  ← epoch boundary
         │
         ├─── epoch 2 ──────────────────────┐
         │ tdm_copy(desc_full_C) ← full     │
         │ async_wait(0)  ──── DRAIN ────────┘  ← leave alone
         │
         ├─── epoch 3 ──────────────────────┐
         │ tdm_copy(desc_spec_D) ← specialized│
         │ async_wait(0)  ──── DRAIN ─────────┘  ← predicate
```

```
For each epoch (bounded by async_wait(0)):
    Collect all tdm_copy ops.
    If ALL use specialized descriptors:
        Predicate all tdm_copy and async_wait in this epoch.
    Else if NONE use specialized descriptors:
        Leave untouched.
    Else (mix):
        Error — mixing is not allowed within an epoch.
```

### Properties

- **Per-epoch granularity.** Different epochs can independently be specialized or not.
- **Local reasoning.** After `async_wait(0)`, the FIFO is empty. Each epoch is self-contained.
- **Wait count correctness.** Within an epoch, all copies were issued by the same warp set. Intermediate `async_wait(N)` counts are correct without recomputation.
- **Natural boundaries.** Real kernels already use `async_wait(0)` at prologue/epilogue boundaries.

---

## Phase-Shift Synergy

The warp-pipeline phase shift is what makes this optimization most effective. Without it, restricting TDM to one group still works but provides less benefit.

```
Phase 0:  Upper(warpLow)  → tdm_copy(tile[1])         Lower(warpHigh) → (deferred)
Phase 1:  Upper            → async_wait; mfma(tile[0])  Lower → mfma(tile[0])
Phase 2:  Upper            → tdm_copy(tile[2])         Lower → mfma(tile[1])
```

Data flow for tile[1]: upper issues `tdm_copy` at Phase 0, does `async_wait` at Phase 1 (data in LDS), lower needs tile[1] at Phase 2 — one full phase **after** the data is already in LDS. The deferral provides free latency hiding.

This also creates an asymmetry: the lower group's `async_wait` disappears entirely in the specialized design — the existing `CondBarrier` at stage boundaries already ensures LDS visibility.

---

## Alternative Triggering Mechanisms

The [challenges](#challenges) apply to all alternatives. The epoch-based analysis resolves the copy-to-wait association problem (Challenge 2) for all of them. The remaining differentiator is how each approach handles descriptor construction (Challenge 3) and API design.

### Option A: Stage-level attribute with paired copies

```python
with gl.amd.warp_pipeline_stage("memory", tdm_specialize=True):
    tdm.copy(desc_a, buf_a)
```

**Viable.** The descriptor is built for 8 warps (unchanged). The compiler has each active warp issue copies for both itself and its partner (warp 0 copies for warp 0 + warp 4, etc.). This avoids descriptor rebuild but **doubles FIFO depth** (2 copies per tensor per warp), requiring adjusted wait counts. Also tied to warp-pipelining — cannot be used standalone.

### Option B: (stride, count) on ops

```python
tdm.copy(desc_a, buf_a, warp_group=(4, 2))    # warps {0, 2, 4, 6}
```

Flexible warp subset notation: `active_warps = { i × stride | i ∈ [0, count) }`. However, the attribute on `tdm_copy` does not affect the separate `make_tensor_desc` op — the descriptor is still built for 8 warps, leaving half the block unwritten. Insufficient as sole mechanism (Challenge 3).

### Option C: warpsPerCTA on ops

```python
tdm.copy(desc_a, buf_a, warpsPerCTA=[4, 1])
```

Conflates layout-level `warpsPerCTA` (tensor encoding) with module-level `num_warps` (TDM descriptor construction). Same descriptor construction problem as Option B.

### Option D: Per-loop attribute

```python
gl.amd.tdm_warp_specialize()   # directive before the pipelined loop
for k in range(0, K, BLOCK_K):
    ...
```

```mlir
scf.for ... { tdm.specialize = true } { ... }
```

Attaches the specialization signal to the `scf.for` loop rather than individual descriptors. The compiler applies narrowed mapping to all TDM copies within that loop. Gives per-loop granularity (useful in multi-loop kernels) without per-descriptor control. **Viable**, but does not resolve Challenge 3 on its own — the compiler must still rebuild descriptors internally, which requires it to identify and modify `make_tensor_desc` ops feeding into the loop. The chosen per-descriptor approach makes this explicit at the source.

### Option E: Compiler flag only

```
-amdgpu-tdm-warp-specialize={off|on|auto}
```

Too coarse for per-kernel or per-descriptor control. However, useful as an **escape hatch** alongside the chosen approach.

### Option F: Auto-detection

The compiler detects TDM specialization candidates from IR patterns (pipeline stages + TDM copies + sufficient warps). Cannot reliably determine **profitability** without profiling data. Best as a supplement, not the sole mechanism.

### Option G: Warp-level programming

```python
with gl.amd.warp_group(count=4, stride=1):
    tdm.copy(desc_a, buf_a)
    tdm.async_wait(N)
```

Breaks Triton's block-level SPMD abstraction. Forces the programmer to reason about warp-to-SIMD mapping, FIFO counters, and per-warp descriptors. Non-portable across architectures. May be the only option if finer-grained control (e.g., different warp subsets for different copies) is ever needed.

---

## Summary

| Aspect | Chosen approach | Rationale |
|--------|----------------|-----------|
| **Signal** | `tdm_specialize=True` on `make_tensor_desc` | Explicit opt-in; self-documenting intent |
| **Gluon API** | `tdm.make_tensor_desc(..., tdm_specialize=True)` | Boolean, simple, no warp count math for programmer |
| **IR** | `tdm.specialize = true` attribute on descriptor op | Per-descriptor granularity; flows through existing IR |
| **Descriptor** | Built with `numWarps=4` (first 4 warps) | Self-consistent sub-tiles; no separate rebuild step |
| **Compiler analysis** | Epoch-based: split at `async_wait(0)`, check per-epoch | Local reasoning; enables partial application |
| **Predication** | `warpLow` in pipelined kernels; `warp_id < 4` otherwise | Matches pipeline group split; works standalone too |
| **Wait counts** | Unchanged per active warp | Active warps issue same number of copies |
| **LDS visibility** | Existing `CondBarrier` + barrier after predicated `async_wait` | Leverages pipeline infrastructure |
| **Phase-shift synergy** | Lower group gets free latency hiding from deferral | Extra benefit in warp-pipelined kernels |
| **Scope** | Not tied to warp-pipelining | Usable in any kernel with TDM copies |
| **Escape hatch** | Compiler flag `-amdgpu-tdm-warp-specialize={off\|on\|auto}` | Override for profiling and debugging |
