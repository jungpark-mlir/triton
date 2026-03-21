# Membar: Warp-Local Shared Memory Access

## Problem

Triton's membar analysis treats shared memory as a flat address space shared by
all threads in a CTA. When two operations touch the same allocation, membar
inserts a CTA-wide barrier (`__syncthreads()` / `s_barrier`) even if the
layout guarantees that each warp only accesses its own partition. In such
cases, the barrier is unnecessary — there is no cross-warp data dependency.

```
  Shared Memory (one buffer slot)
  ┌─────────────────────────────────────────────────┐
  │  Warp 0 region      │  Warp 1 region            │
  │  [0x000, 0x100)     │  [0x100, 0x200)           │
  │                     │                           │
  │  W0 writes here     │  W1 writes here           │
  │  W0 reads here      │  W1 reads here            │
  └─────────────────────┴───────────────────────────┘
        ↑ no cross-warp overlap → barrier unnecessary
```

## Background: How Addresses Are Computed

### Composed Layout

For `local_store` / `local_load`, the lowering computes a **composed layout**:

```cpp
auto regLayout = toLinearLayout(regTy);       // register: logical elements → threads
auto sharedLayout = toLinearLayout(memDescTy); // shared: logical elements → offsets
auto cvt = regLayout.invertAndCompose(sharedLayout);
// cvt: {register, lane, warp, block} → {offset}
```

This `cvt` maps thread coordinates to shared memory byte offsets:

```
  Input dimensions              Output dimension
  ┌──────────┐
  │ register │──┐
  ├──────────┤  │    LinearLayout     ┌────────┐
  │   lane   │──┼───────────────────▶ │ offset │
  ├──────────┤  │    (XOR / GF(2))    └────────┘
  │   warp   │──┘
  └──────────┘
```

Each input dimension contributes basis vectors. The offset is computed as:

```
offset = Σ(reg_bit_i × reg_base_i) ⊕ Σ(lane_bit_j × lane_base_j)
         ⊕ Σ(warp_bit_k × warp_base_k)
```

where `⊕` is XOR (addition in GF(2)).

### Warp-Disjoint vs Cross-Warp Access

How the `warp` bases relate to the `register + lane` bases determines whether
warps access overlapping or disjoint address ranges:

```
  Cross-warp (typical):              Warp-disjoint:
  warp bases overlap with            warp bases are independent of
  register/lane bases                register/lane bases

  offset bits: [b7 b6 b5 b4 b3 b2 b1 b0]
               ├── reg/lane ─┤├─ warp ─┤   ← warp selects high bits
               ├── overlapping ────────┤   ← warp mixed with reg/lane

  Cross-warp:                        Warp-disjoint:
  W0: {0x00..0x3F, 0x80..0xBF}      W0: [0x00, 0x40)
  W1: {0x40..0x7F, 0xC0..0xFF}      W1: [0x40, 0x80)
       ↑ interleaved                      ↑ contiguous, non-overlapping
```

### Existing Warp-Level Sync Detection

Triton already has a partial version of this concept for `ConvertLayoutOp`
scratch buffers. `isCvtDimSync` checks if the conversion's composed layout
is trivial over the warp dimension. When true, membar emits `warp.sync`
instead of a CTA-wide barrier and avoids clearing CTA-wide pending
dependencies. This mechanism is limited to `ConvertLayoutOp` scratch
buffers.

## Detection via GF(2) Linear Independence

### Formal Condition

Given two composed layouts `cvt_A` and `cvt_B` (for the write and read ops),
no cross-warp conflict exists if, for all warp ids `w ≠ w'`, the address
sets are disjoint:

```
Addr_X(w) = { cvt_X(r, l, w, b) | ∀ register indices r, ∀ lane ids l }
∀ w ≠ w':  Addr_A(w) ∩ Addr_B(w') = ∅
```

### GF(2) Independence Check

Since `LinearLayout` uses GF(2) arithmetic, this reduces to checking
**linear independence** of basis vectors.

For a single composed layout `cvt`, let:
- `R ∪ L` = basis vectors from `register` and `lane` dimensions
- `W` = basis vectors from `warp` dimension

**Theorem**: The per-warp address sets are disjoint if and only if
`rank(R ∪ L ∪ W) = rank(R ∪ L) + |W|` over GF(2).

When warp bases are independent, different warp IDs flip offset bits that
no register/lane combination can toggle:

```
  Example: 4 warps, 2 warp bits

  Basis matrix (rows = basis vectors, columns = offset bits):

  register bases:  [ 0 0 0 0 0 0 0 1 ]   ← bit 0
                   [ 0 0 0 0 0 0 1 0 ]   ← bit 1
                   [ 0 0 0 0 0 1 0 0 ]   ← bit 2
  lane bases:      [ 0 0 0 0 1 0 0 0 ]   ← bit 3
                   [ 0 0 0 1 0 0 0 0 ]   ← bit 4
  ─────────────────────────────────────
  warp bases:      [ 0 0 1 0 0 0 0 0 ]   ← bit 5  ← independent!
                   [ 0 1 0 0 0 0 0 0 ]   ← bit 6  ← independent!

  → Warp 0: offsets [0x00, 0x20)
    Warp 1: offsets [0x20, 0x40)
    Warp 2: offsets [0x40, 0x60)
    Warp 3: offsets [0x60, 0x80)    → all disjoint ✓
```

For a **pair** of operations, both composed layouts must satisfy this check,
and their warp bases must produce the same partitioning (i.e., same warp
basis vectors in offset space).

The check is a standard **Gaussian elimination** on the combined basis matrix
— O(n²) where n is the number of offset bits (typically 10-15). Negligible
cost.

## MMA Dot Operand Layouts

MMA operand layouts are **not always cross-warp**. The warp distribution for
dot operands is constructed by `broadcastedDotOperandLayout`:

```
  MMA output C tiled by warpsPerCTA = {2, 2}:

  ┌─────────┬─────────┐
  │  W0     │  W1     │        Operand A:         Operand B:
  │         │         │        ┌────┬────┐        ┌────┬────┐
  ├─────────┼─────────┤        │W0  │W0  │        │W0  │W1  │
  │  W2     │  W3     │        │W1  │W1  │        │W2  │W3  │
  │         │         │        ├────┼────┤        ├────┼────┤
  └─────────┴─────────┘        │W2  │W2  │        │W0  │W1  │
                               │W3  │W3  │        │W2  │W3  │
  C: warps tile M × N          └────┴────┘        └────┴────┘
                               A: partition M,     B: partition N,
                                  broadcast K         broadcast K
```

The code creates identity (partition) along the non-K dimension and zeros
(broadcast) along K:

```cpp
for (auto d : order) {
  if (d == kDim)
    layout *= LinearLayout::zeros1D(shape[d], "warp", dimNames[d]); // broadcast
  else
    layout *= LinearLayout::identity1D(shape[d], "warp", dimNames[d]); // partition
}
```

Whether this translates to warp-disjoint shared memory access depends on the
composed layout. Consider operand A with `warpsPerCTA = {4, 1}`, shape
`[64, 64]`, `f16`:

```
  Non-swizzled shared layout (row stride = 128 bytes):

  Shared memory offsets:
  ┌──────────────────────────┐ 0x0000
  │ Rows  0-15  (Warp 0)    │
  ├──────────────────────────┤ 0x0800
  │ Rows 16-31  (Warp 1)    │
  ├──────────────────────────┤ 0x1000
  │ Rows 32-47  (Warp 2)    │
  ├──────────────────────────┤ 0x1800
  │ Rows 48-63  (Warp 3)    │
  └──────────────────────────┘ 0x2000

  Warp bases map to bits 11-12 of offset → independent of
  register/lane bases (bits 0-10) → warp-disjoint ✓
```

```
  Swizzled shared layout (maxPhase=4, perPhase=2):

  Shared memory offsets:
  ┌──────────────────────────┐
  │ Row 0:  col XOR (row>>2) │   ← row bits mixed into column
  │ Row 1:  col XOR (row>>2) │      address via swizzle
  │ ...                      │
  └──────────────────────────┘

  Warp bases (from row index) overlap with register/lane bases
  (from column index) after XOR → NOT independent → cross-warp ✗
```

The GF(2) check handles both cases correctly.

## Async Copy, TDM Copy, and Async Wait

### TDM Copy (AMD gfx1250)

TDM (Tensor Data Mover) copies are **inherently warp-partitioned on the
write side**. The TDM linear layout distributes the block across warps:

```
  getTDMLinearLayout:

  ┌─────────────────────────────────────────────────────────────┐
  │ identity("message", messageShape)                           │
  │    × identity("warp", warpsPerCTA)    ← warp partitions    │
  │    × cgaLayout                        ← cluster             │
  └─────────────────────────────────────────────────────────────┘

  For a 64×64 block with 4 warps distributed as {4, 1}:

  TDM DMA targets:
  ┌──────────────────┐
  │ W0: rows  0-15   │ ← warp 0 DMA writes here
  ├──────────────────┤
  │ W1: rows 16-31   │ ← warp 1 DMA writes here
  ├──────────────────┤
  │ W2: rows 32-47   │ ← warp 2 DMA writes here
  ├──────────────────┤
  │ W3: rows 48-63   │ ← warp 3 DMA writes here
  └──────────────────┘
  Write side: always warp-disjoint by construction
```

The composed mapping `tdmLayout.invertAndCompose(sharedLayout)` determines
the actual shared memory offsets. Because TDM assigns each warp a disjoint
sub-block, the destination addresses are warp-disjoint by construction.

**However**, the consumer (`local_load`) uses a different layout. Its
warp-to-address mapping depends on the register encoding:

```
  TDM write → local_load (cross-warp reader):

  Write:                    Read (MMA operand):
  ┌──────────┐              ┌──────────┐
  │ W0 only  │              │ W0 + W1  │ ← reads from both
  ├──────────┤              │          │    warp 0 and warp 1
  │ W1 only  │              ├──────────┤    regions
  ├──────────┤              │ W2 + W3  │
  │ W2 only  │              │          │
  ├──────────┤              └──────────┘
  │ W3 only  │              CTA barrier needed!
  └──────────┘

  TDM write → local_load (matching partition):

  Write:                    Read:
  ┌──────────┐              ┌──────────┐
  │ W0 only  │              │ W0 only  │
  ├──────────┤              ├──────────┤
  │ W1 only  │              │ W1 only  │
  ├──────────┤              ├──────────┤
  │ W2 only  │              │ W2 only  │
  ├──────────┤              ├──────────┤
  │ W3 only  │              │ W3 only  │
  └──────────┘              └──────────┘
  No CTA barrier needed — each warp reads its own data
```

### NVIDIA `async_copy_global_to_local`

NVIDIA's `cp.async` is a **per-thread** async copy. Each thread independently
copies its assigned elements. The destination address depends on the shared
encoding. With `SwizzledSharedEncodingAttr`, swizzling mixes lane and warp
bits, making writes cross-warp. Whether the write pattern is warp-disjoint
can be checked with the same GF(2) independence test on the composed layout.

### `async_wait` Semantics

`async_wait` ensures outstanding DMA operations have completed and their
results are visible in shared memory:

- **AMD**: CTA-wide memory visibility (all warps see the data). Not an
  execution barrier.
- **NVIDIA**: Per-thread `cp.async.wait_group`. Cross-warp visibility
  requires a subsequent `__syncthreads()`.

`async_wait` is **not an obstacle** to the warp-local optimization:

```
  Timeline (warp-disjoint case):

  Warp 0:  [TDM write → region A] ... [async_wait] ... [local_load ← region A]
  Warp 1:  [TDM write → region B] ... [async_wait] ... [local_load ← region B]

  Each warp reads only what it wrote → no cross-warp dependency
  → async_wait is sufficient, no CTA barrier needed
```

The existing AMD membar filter handles DMA-completion sequencing (token chain
to `async_wait`). Warp-disjointness is an orthogonal check that could further
reduce barriers.

### Interaction Summary

| Operation | Write Warp-Disjoint? | Read Warp-Disjoint? | Barrier? |
|-----------|:--------------------:|:--------------------:|:--------:|
| TDM copy → local_load (matching partition) | Yes | Yes | **No** |
| TDM copy → local_load (MMA layout) | Yes | Usually no | **Yes** |
| NVIDIA async_copy → local_load | Depends | Depends | Usually **yes** |
| local_store → local_load (warp-local) | Depends | Depends | **No** if both |

## Integration with Membar

### Option A: Filter Function

Add a filter (similar to the AMD async-write filter) that checks
warp-disjointness for a pair of operations:

```
  membar analysis
       │
       ▼
  ┌─────────────────────────────────────────┐
  │ AllocationSlice::intersects()           │
  │  1. Check allocation interval overlap   │
  │  2. Check BufferIndexExpr disjointness  │  ← buffer slot
  │  3. Check warp-disjointness (NEW)       │  ← per-warp partition
  │  4. Check subslice overlap              │
  └─────────────────────────────────────────┘
```

For step 3, compute the composed layouts from both operations and run the
GF(2) independence check. If both layouts have independent warp bases with
matching partitions, return `false` (no intersection).

**Pros**: Composable with existing checks. Single point of implementation.

**Cons**: Requires computing composed layouts during membar analysis. The
register-side layout must be available from the operation's tensor type.

### Option B: Pre-Analysis Annotation

A pre-pass annotates `local_store`/`local_load` ops with a `warp_disjoint`
attribute when the layout guarantees warp-local access. Membar checks this
attribute.

**Pros**: Clean separation. Layout analysis happens once.

**Cons**: Another pass to maintain. Attribute fragility across transforms.

## Applicability

### When This Optimization Applies

1. **Blocked register layout with non-swizzled shared layout**: The warp
   dimension naturally partitions both the register and shared address spaces.

2. **MMA dot operand with non-swizzled shared layout**: When `warpsPerCTA`
   partitions the M (or N) dimension and the shared layout maps rows (or
   columns) to contiguous, non-overlapping offset ranges.

3. **Warp-specialized Gluon kernels**: Explicit per-warp shared memory
   partitioning.

4. **Per-warp scratch buffers**: Warp-local reductions or shuffles.

### When This Does NOT Apply

1. **Swizzled shared encodings**: XOR-mixing of row/column bits breaks warp
   independence.

2. **Cross-warp MMA operand loading**: When the MMA layout requires each warp
   to read data from multiple warp regions.

3. **Cooperative DMA patterns**: Where the shared encoding distributes writes
   across all warps.

### Interaction with Multi-Buffering

Warp-disjointness and buffer slot disjointness (`BufferIndexExpr`) are
**orthogonal** and compose naturally:

```
  ┌─────────────────────────────────────┐
  │ Multi-buffer allocation             │
  │ ┌───────────┬───────────┬─────────┐ │
  │ │  Slot 0   │  Slot 1   │ Slot 2  │ │  ← BufferIndexExpr
  │ │┌───┬─────┐│           │         │ │     proves slot
  │ ││W0 │ W1  ││           │         │ │     disjointness
  │ │├───┼─────┤│           │         │ │
  │ ││W2 │ W3  ││           │         │ │  ← GF(2) check proves
  │ │└───┴─────┘│           │         │ │     warp disjointness
  │ └───────────┴───────────┴─────────┘ │     within a slot
  └─────────────────────────────────────┘
```

## Summary

| Aspect | Current State | Proposed |
|--------|--------------|----------|
| **Warp awareness** | `ConvertLayoutOp` scratch only (`isCvtDimSync`) | General shared memory ops via GF(2) independence |
| **Detection** | Trivial-over-warp check | LinearLayout basis Gaussian elimination |
| **Async/TDM** | Not considered | TDM write always disjoint; read checked via same mechanism |
| **MMA operands** | Assumed cross-warp | Checkable per-configuration via composed layout |
| **Scope** | Situational; most encodings are cross-warp by design | Same, but catches cases where layout happens to be warp-local |
