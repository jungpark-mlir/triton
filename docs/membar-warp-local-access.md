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

## Implementation Status

**Implemented** in commit
[`df6d5be`](https://github.com/triton-lang/triton/commit/df6d5be2206ec6f32cf47116d23f3b6235873bfe)
via the `warpsPerCTA` comparison approach (see "Implemented Detection"
below). The GF(2) linear independence approach described later in this
document was the original design proposal but was superseded during
implementation by a simpler and strictly more general method.

### Implemented Detection: warpsPerCTA Comparison

The implemented check compares the `warpsPerCTA` distribution on both
the write and read sides. If both operations distribute warps identically
across tensor dimensions, each warp owns the same partition of tensor
elements on both sides.

The key insight is the **bijection argument**: all Triton shared memory
encodings (padded, swizzled, linear, rotating) are bijections from
tensor elements to byte addresses. Swizzling permutes which byte offset
a `(row, col)` pair maps to, but never collapses two distinct elements
onto the same address. Therefore, identical tensor-space partitioning
implies disjoint byte-address partitioning — regardless of encoding.

```
  Writer (TDM copy)                 Reader (local_load)
  warpsPerCTA = [4, 1]              warpsPerCTA = [4, 1]

  Tensor space (elements):          Tensor space (elements):
  ┌──────────────────┐              ┌──────────────────┐
  │ W0: rows  0-15   │              │ W0: rows  0-15   │
  ├──────────────────┤              ├──────────────────┤
  │ W1: rows 16-31   │              │ W1: rows 16-31   │
  ├──────────────────┤              ├──────────────────┤
  │ W2: rows 32-47   │              │ W2: rows 32-47   │
  ├──────────────────┤              ├──────────────────┤
  │ W3: rows 48-63   │              │ W3: rows 48-63   │
  └──────────────────┘              └──────────────────┘
         │                                 │
         │  shared encoding                │  shared encoding
         │  (bijection)                    │  (bijection)
         ▼                                 ▼
  Byte addresses:                   Byte addresses:
  ┌──────────────────┐              ┌──────────────────┐
  │ W0: addr set A   │              │ W0: addr set A   │
  ├──────────────────┤              ├──────────────────┤
  │ W1: addr set B   │              │ W1: addr set B   │
  ├──────────────────┤              ├──────────────────┤
  │ W2: addr set C   │              │ W2: addr set C   │
  ├──────────────────┤              ├──────────────────┤
  │ W3: addr set D   │              │ W3: addr set D   │
  └──────────────────┘              └──────────────────┘
  A∩B = B∩C = ... = ∅              A∩B = B∩C = ... = ∅

  Key: bijection means distinct elements → distinct addresses.
  Same warp partition in tensor space → same disjoint partitions
  in address space, regardless of encoding (padded, swizzled, etc.)
```

Trailing 1s in `warpsPerCTA` are stripped before comparison to handle
rank changes from `memdesc_reshape`/`trans` (e.g., `[4,1]` matches
`[4,1,1]`).

### Why This Supersedes GF(2)

The GF(2) approach operates on `LinearLayout` basis vectors and proves
disjointness via linear independence over GF(2). While mathematically
elegant, it has practical limitations:

1. **Fails for padded layouts.** Non-power-of-2 strides cause carry
   interactions that GF(2) (XOR-only arithmetic) cannot model. The
   primary motivating use case (FA MQA decode) uses padded shared
   layouts.

2. **Requires `LinearLayout` computation.** The composed layout must be
   computed during membar analysis, adding complexity.

3. **No additional coverage.** The warpsPerCTA comparison covers all
   practical cases where warp-local access matters. The only theoretical
   case GF(2) catches but warpsPerCTA does not is 2D warp distributions
   (e.g., `warpsPerCTA = [2, 2]`) where warps partition both rows and
   columns simultaneously. In practice, such layouts are cross-warp
   (MMA operands with K-broadcast) and the optimization doesn't apply.

The bijection argument is stronger: it works for **all** shared memory
encodings without needing to inspect the address computation at all.

### Integration

The check is implemented as a new `filterWarpLocalAccesses` clause in
the AMD `membarFilter` in `MembarUtility.cpp`:

```cpp
bool membarFilter(...) {
  return (filterAsyncLocalLoadsDependencies(...) ||
          filterLDSMemoryBarriersDependencies(...) ||
          filterWarpLocalAccesses(op1, op2));  // NEW
}
```

Currently scoped to operation pairs involving `AsyncTDMCopyGlobalToLocalOp`
to avoid changing barrier behavior for existing non-TDM code paths. The
`hasMatchingWarpDistribution` helper extracts `warpsPerCTA` from either
a TDM op (via `tdmGetWarpDistribution`) or a register-side op (via
the encoding's `getWarpsPerCTA`), normalizes trailing 1s, and compares.

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

where `⊕` is XOR (addition in GF(2)). See below for what GF(2) means and
why `LinearLayout` uses it.

### GF(2): The Arithmetic Behind LinearLayout

**GF(2)** (Galois Field of order 2) is arithmetic on just two values,
`{0, 1}`, with two operations:

```
  GF(2) addition (= XOR):          GF(2) multiplication (= AND):

  0 + 0 = 0                        0 × 0 = 0
  0 + 1 = 1                        0 × 1 = 0
  1 + 0 = 1                        1 × 0 = 0
  1 + 1 = 0  ← key difference!     1 × 1 = 1
```

The crucial property: `1 + 1 = 0`, not 2. There are no carries. This is
exactly bitwise XOR. And multiplication is exactly bitwise AND.

**A vector over GF(2)** is simply a bit pattern. For example, `[0,1,0,0]`
is a 4-bit vector. Adding two vectors means XOR-ing them bitwise:

```
  [1, 0, 1, 0]  ⊕  [0, 1, 1, 0]  =  [1, 1, 0, 0]
       ↑                  ↑                 ↑
   basis A            basis B           A XOR B
```

**Why LinearLayout uses GF(2).** GPU shared memory layouts often use
XOR-based swizzling to avoid bank conflicts. The address each thread
accesses is not a simple sum of coordinates — it's a XOR combination.
`LinearLayout` models this directly: each input bit (from register index,
lane ID, or warp ID) selects a basis vector (via AND), and all selected
basis vectors are combined via XOR:

```
  Concrete example: 2 lanes, 2 warps, 2 register elements

  Basis vectors (each maps one input bit to offset bits):
    register bit 0:  [0, 0, 0, 1]     ← offset bit 0
    lane bit 0:      [0, 0, 1, 0]     ← offset bit 1
    warp bit 0:      [0, 1, 0, 0]     ← offset bit 2

  Thread (reg=1, lane=1, warp=0):
    offset = (1 AND [0,0,0,1]) XOR (1 AND [0,0,1,0]) XOR (0 AND [0,1,0,0])
           = [0,0,0,1] XOR [0,0,1,0] XOR [0,0,0,0]
           = [0,0,1,1]  → offset 3

  Thread (reg=1, lane=1, warp=1):
    offset = [0,0,0,1] XOR [0,0,1,0] XOR [0,1,0,0]
           = [0,1,1,1]  → offset 7

  Warp 0 accesses: {0, 1, 2, 3}    (offsets 0-3)
  Warp 1 accesses: {4, 5, 6, 7}    (offsets 4-7)
  → disjoint, because the warp basis [0,1,0,0] flips a bit that
    no register/lane combination can produce
```

**Linear independence over GF(2)** means: no vector in a set can be
produced by XOR-ing any combination of the others. In practical terms,
each basis vector contributes a unique bit pattern that cannot be
replicated by combining other basis vectors. This is exactly what
determines whether changing the warp ID produces an address that some
other register/lane combination could also reach.

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

## Design Alternative: GF(2) Linear Independence

> **Note**: This was the original design proposal. The implementation uses
> the simpler `warpsPerCTA` comparison with the bijection argument instead
> (see "Implementation Status" above). This section is retained as
> background for understanding the `LinearLayout` address model and for
> cases where finer-grained analysis might be needed in the future.

### Formal Condition

Given two composed layouts `cvt_A` and `cvt_B` (for the write and read ops),
no cross-warp conflict exists if, for all warp ids `w ≠ w'`, the address
sets are disjoint:

```
Addr_X(w) = { cvt_X(r, l, w, b) | ∀ register indices r, ∀ lane ids l }
∀ w ≠ w':  Addr_A(w) ∩ Addr_B(w') = ∅
```

### The Key Insight

The question "can two different warps access the same address?" becomes
a question about GF(2) linear algebra:

- Each warp ID is a different bit pattern (warp 0 = `00`, warp 1 = `01`,
  warp 2 = `10`, warp 3 = `11` for 4 warps).
- The warp ID bits select warp basis vectors via AND, then XOR them into
  the offset.
- If the warp basis vectors are **linearly independent** of the
  register/lane basis vectors, then changing the warp ID flips offset
  bits that no register/lane combination can compensate for. The
  resulting address ranges are guaranteed disjoint.
- If a warp basis vector **can** be produced by XOR-ing some register/lane
  basis vectors, then a different warp could reach the same address by
  toggling different register/lane bits. The address ranges overlap.

### GF(2) Independence Check

For a single composed layout `cvt`, let:
- `R ∪ L` = basis vectors from `register` and `lane` dimensions
- `W` = basis vectors from `warp` dimension

**Theorem**: The per-warp address sets are disjoint if and only if
`rank(R ∪ L ∪ W) = rank(R ∪ L) + |W|` over GF(2).

In plain terms: put all basis vectors (register, lane, and warp) into a
matrix and count the number of linearly independent rows (the rank). If
adding the warp rows increases the rank by exactly the number of warp
rows, they are independent — no warp basis can be built from
register/lane bases.

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

Counter-example — when warp bases are NOT independent (cross-warp):

```
  Example: warp basis overlaps with register basis

  register bases:  [ 0 0 0 0 0 0 0 1 ]   ← bit 0
                   [ 0 0 0 0 0 0 1 0 ]   ← bit 1
                   [ 0 0 1 0 0 0 0 0 ]   ← bit 5  ← same as warp!
  lane bases:      [ 0 0 0 0 1 0 0 0 ]   ← bit 3
                   [ 0 0 0 1 0 0 0 0 ]   ← bit 4
  ─────────────────────────────────────
  warp bases:      [ 0 0 1 0 0 0 0 0 ]   ← bit 5  ← NOT independent

  rank(R ∪ L) = 5,  rank(R ∪ L ∪ W) = 5  (warp row adds nothing)
  5 ≠ 5 + 1 → NOT independent

  What happens: warp 0 with reg bit 2 set → offset has bit 5 set
                warp 1 with reg bit 2 clear → offset also has bit 5 set
                → same address reached by different (warp, reg) combos
                → address ranges overlap → barrier needed
```

For a **pair** of operations, both composed layouts must satisfy this check,
and their warp bases must produce the same partitioning (i.e., same warp
basis vectors in offset space).

The check is a standard **Gaussian elimination** on the combined basis matrix
— O(n²) where n is the number of offset bits (typically 10-15). Negligible
cost.

### Assumptions Behind the Criterion

The GF(2) rank condition is a mathematical fact about linear algebra over
GF(2). Its applicability to barrier elimination depends on the following
preconditions about Triton's `LinearLayout` and hardware:

1. **Complete layout.** The composed layout `cvt` maps all thread coordinates
   `{register, lane, warp}` to shared memory offsets — i.e., every element
   accessed by the operation is represented. This is guaranteed by Triton's
   `toLinearLayout`, which always produces a complete mapping for the tensor
   shape.

2. **Static layout.** The composed layout is fully determined at compile time.
   There is no runtime-dependent reshaping of the thread-to-offset mapping.
   This holds for all current Triton encodings (`BlockedEncodingAttr`,
   `SwizzledSharedEncodingAttr`, `AMDMfmaEncodingAttr`, etc.).

3. **Warp dimension corresponds to hardware warps.** The `warp` dimension in
   `LinearLayout` maps to the hardware warp IDs used by `warp.sync`. This is
   an invariant of Triton's lowering: the `warp` dimension is derived from
   `warpsPerCTA` and corresponds directly to `threadIdx / warpSize`.

All three conditions hold in Triton today. If a future encoding introduces
runtime-dependent address remapping or redefines the warp dimension, this
check would need to be re-evaluated.

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

The `warpsPerCTA` comparison handles both cases correctly: matching
`warpsPerCTA` on both sides means warp-disjoint, mismatching means
cross-warp (barrier needed). The GF(2) check (design alternative)
would also distinguish them, though it was not implemented.

## Batched MMA (Warps Across Batch Dimension)

A distinct pattern from the standard MMA operand layout is **batched MMA**,
where warps are distributed across a batch dimension rather than M/N of a
single MMA tile. Each warp executes an independent MMA on its own data.

### Motivating Example: FA MQA Decode with Split-K

In flash attention MQA decode, batched WMMA is used for split-k where 4
warps are distributed across the batch dimension. TDM loads K and V with
a 2D shape `[4 * BLOCK_N, HEAD_SZ]`, which is then reshaped to
`[4, BLOCK_N, HEAD_SZ]` for WMMA. Each warp independently loads and
consumes its own K/V partition:

```
  TDM load: [4 * BLOCK_N, HEAD_SZ]      Batched WMMA view:
  ┌──────────────────────────┐           [4, BLOCK_N, HEAD_SZ]
  │ W0: rows [0, BN)         │   ──→    W0: MMA on its own BN×HS slice
  ├──────────────────────────┤
  │ W1: rows [BN, 2·BN)      │   ──→    W1: MMA on its own BN×HS slice
  ├──────────────────────────┤
  │ W2: rows [2·BN, 3·BN)    │   ──→    W2: MMA on its own BN×HS slice
  ├──────────────────────────┤
  │ W3: rows [3·BN, 4·BN)    │   ──→    W3: MMA on its own BN×HS slice
  └──────────────────────────┘
  No cross-warp data dependency → CTA barrier unnecessary
```

The pipeline uses triple buffering with `a = i % 3`, `b = (i + 1) % 3`,
`c = (i + 2) % 3`. Currently, membar inserts barriers between TDM writes
and the subsequent `local_load`:

```python
  self.async_wait(6)
  self.issue_global_load_v(i + 2, buf=c)    # TDM write → buffer c
  # <-- barrier inserted (false positive)
  v = self.shared_load_v(buf=a)             # local_load ← buffer a
```

Two independent problems contribute to this false positive:

1. **Buffer index disjointness** — TDM writes to buffer `c = (i+2)%3`
   while `local_load` reads from buffer `a = i%3`. These are different
   buffer slots, but membar cannot distinguish them because the indices
   are dynamic. `BufferIndexExpr` handles this (Problem 1 in the doc set).

2. **Same-buffer write→read from a previous iteration** — The TDM write
   that filled buffer `a` happened 2 iterations ago. Even if buffer index
   disjointness resolves the cross-buffer case, there is still a pending
   RAW dependency on buffer `a` itself. `async_wait` ensures the data is
   visible, but membar inserts a CTA-wide barrier because it cannot tell
   that the access is warp-local.

The warp-local check is the **stronger and more general** fix. If both
the TDM write and the WMMA `local_load` are warp-disjoint with matching
partitions, no cross-warp data dependency exists regardless of buffer
slot — eliminating barriers from both problems simultaneously.

### Why This Differs from Standard MMA Operands

In the standard MMA case (Section "MMA Dot Operand Layouts"), warps are
distributed across M and N of a single MMA tile, with K broadcast across
warps. Operand A partitions M (warp-disjoint along M) but broadcasts K
(all warps read the same K data from shared memory).

In batched MMA, warps are across the batch dimension. There is **no
K-dimension broadcast** — each warp has its own K data entirely. This
makes it a simpler case: the warp dimension selects which batch element
(which `BLOCK_N × HEAD_SZ` block in shared memory), and each warp's
threads address only within that block.

### Detection

The `warpsPerCTA` comparison detects this case directly:

- **TDM write side**: `tdmGetWarpDistribution` returns `warpsPerCTA`
  matching the block shape partitioning (e.g., `[4, 1]` for 4 warps
  distributed along rows).
- **Local_load (WMMA operand) side**: the distributed encoding's
  `warpsPerCTA` from `getWarpsPerCTA` (e.g., `[4, 1]` for batched WMMA
  with all warps along the batch/M dimension).

If both sides report the same normalized `warpsPerCTA`, the bijection
argument guarantees disjoint byte-address partitioning regardless of
shared encoding (padded, swizzled, or otherwise).

## Async Copy, TDM Copy, and Async Wait

### TDM Copy (AMD gfx1250)

TDM (Tensor Data Mover) copies produce a **warp-partitioned write distribution**.
The TDM linear layout distributes the block across warps:

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
  Write side: warp-disjoint (TDM assigns disjoint sub-blocks per warp)
```

The composed mapping `tdmLayout.invertAndCompose(sharedLayout)` determines
the actual shared memory offsets. The TDM layout assigns each warp a disjoint
sub-block, making the *write-side* distribution warp-disjoint. However, the
full barrier elimination result also depends on the consumer's composed layout
(see below).

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
can be checked with the same `warpsPerCTA` comparison (or the GF(2) design
alternative). The current implementation is scoped to AMD TDM ops and does
not cover NVIDIA `cp.async`.

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
to `async_wait`). Warp-disjointness is an orthogonal check that further
reduces barriers when the access pattern is warp-local.

### Interaction Summary

| Operation | Write Warp-Disjoint? | Read Warp-Disjoint? | Barrier? |
|-----------|:--------------------:|:--------------------:|:--------:|
| TDM copy → local_load (matching partition) | Yes | Yes | **No** |
| TDM copy → local_load (batched MMA, warps across batch) | Yes | Yes | **No** |
| TDM copy → local_load (standard MMA, K broadcast) | Yes | Usually no | **Yes** |
| NVIDIA async_copy → local_load | Depends | Depends | Usually **yes** |
| local_store → local_load (warp-local) | Depends | Depends | **No** if both |

## Integration with Membar

**Implemented**: `MembarFilterFn` in the AMD backend (Option A below).

### Option A: Filter Function (Implemented)

The warp-local check is implemented as a `filterWarpLocalAccesses`
clause in the AMD `membarFilter` (`MembarUtility.cpp`). The filter
receives both operations in a potential RAW/WAR/WAW pair and returns
`true` to suppress the barrier when `warpsPerCTA` distributions match.

```
  membar analysis
       │
       ▼
  ┌──────────────────────────────────────────────────────┐
  │ BlockInfo::isIntersected()                           │
  │   for each (slice, ops) pair with overlapping slices │
  │     → MembarFilterFn(op1, op2, ...)                  │
  │       1. filterAsyncLocalLoadsDependencies (existing) │
  │       2. filterLDSMemoryBarriersDependencies          │
  │       3. filterWarpLocalAccesses (NEW)                │
  │          → hasMatchingWarpDistribution(op1, op2)      │
  │            compares normalized warpsPerCTA             │
  └──────────────────────────────────────────────────────┘
```

**Pros**: Fits the existing architecture. Has access to both operations
(needed to extract encoding metadata). No `LinearLayout` computation.

**Current scope**: Only fires when at least one op is
`AsyncTDMCopyGlobalToLocalOp`. Can be extended to `local_store` /
`local_load` pairs in the future.

### Option B: Pre-Analysis Annotation (Not implemented)

A pre-pass annotates `local_store`/`local_load` ops with a `warp_disjoint`
attribute when the layout guarantees warp-local access. Membar checks this
attribute.

**Pros**: Clean separation. Layout analysis happens once.

**Cons**: Another pass to maintain. Attribute fragility across transforms.

## Applicability

### When This Optimization Applies

1. **Blocked register layout with matching warp distribution**: The
   `warpsPerCTA` on the write and read sides match, and each warp owns
   a disjoint partition of tensor elements. Works with any shared encoding
   (padded, swizzled, linear) due to the bijection argument.

2. **MMA dot operand with matching warp distribution**: When `warpsPerCTA`
   partitions the M (or N) dimension identically on both sides. The shared
   encoding is irrelevant — the bijection guarantees address disjointness.

3. **Batched MMA with warps across batch dimension**: Each warp executes
   an independent MMA on its own partition (e.g., FA MQA decode with
   split-k). No K-dimension broadcast — strictly simpler than case 2.

4. **Warp-specialized Gluon kernels**: Explicit per-warp shared memory
   partitioning.

5. **Per-warp scratch buffers**: Warp-local reductions or shuffles.

### When This Does NOT Apply

1. **Mismatched warp distributions**: When the writer and reader distribute
   warps differently (e.g., TDM writes with `[4, 1]` but MMA reads with
   `[2, 2]` due to K-broadcast). The `warpsPerCTA` comparison correctly
   rejects these.

2. **Cross-warp MMA operand loading**: When the MMA layout requires each warp
   to read data from multiple warp regions (K-broadcast in standard MMA).

3. **Cooperative DMA patterns**: Where the shared encoding distributes writes
   across all warps.

4. **Non-TDM code paths** (current scope limitation): The implementation
   currently only fires when at least one operation is
   `AsyncTDMCopyGlobalToLocalOp`. The `warpsPerCTA` comparison logic itself
   is general and could be extended to `local_store`/`local_load` pairs.

### Padded and Non-Power-of-2 Shared Layouts

Padded shared layouts add extra bytes per row to avoid bank conflicts —
e.g., allocating `[64, 72]` instead of `[64, 64]` with 8 elements of
padding. This changes the row stride from a power-of-2 to a
non-power-of-2 value (e.g., 144 bytes instead of 128).

**The warp-disjointness property is preserved by padding.** Padding adds
unused space at the end of each row but does not change which rows belong
to which warp. If warp 0 owns rows `[0, BLOCK_N)` and warp 1 owns rows
`[BLOCK_N, 2*BLOCK_N)`, the per-warp byte ranges remain contiguous and
non-overlapping regardless of stride:

```
  Non-padded (stride = 128, power of 2):
  W0: [0, BLOCK_N × 128)          W1: [BLOCK_N × 128, 2·BLOCK_N × 128)
                                   → disjoint ✓

  Padded (stride = 144, NOT power of 2):
  W0: [0, BLOCK_N × 144)          W1: [BLOCK_N × 144, 2·BLOCK_N × 144)
                                   → still disjoint ✓
```

**The implemented `warpsPerCTA` comparison handles padded layouts
natively.** Because the check operates on tensor-space warp partitioning
rather than byte addresses, padding is irrelevant — the bijection
argument guarantees that identical warp distributions produce disjoint
byte ranges regardless of the encoding's stride or swizzling. This was a
key advantage over the GF(2) approach, which cannot model non-power-of-2
strides.

The FA MQA decode case (the primary motivation) uses
`PaddedSharedEncodingAttr` (e.g., `padded_shared<[32:+4]>` → stride
272 bytes). The implementation correctly suppresses barriers in this
case because `warpsPerCTA` comparison does not inspect address
computation at all.

**Why GF(2) fails here** (for reference): `LinearLayout` computes
offsets via XOR (GF(2) addition), which equals integer addition only
when there are no carries. A row stride with multiple bits set (e.g.,
144 = `0b10010000`) produces carry interactions when multiple row-index
bits are combined:

```
  GF(2) vs integer arithmetic for stride = 3 (0b11):

  row = 1:  GF(2): 3     integer: 3     ✓ (matches)
  row = 2:  GF(2): 6     integer: 6     ✓ (matches)
  row = 3:  GF(2): 3⊕6=5 integer: 9     ✗ (carry breaks it)
```

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
  │ ││W2 │ W3  ││           │         │ │  ← warpsPerCTA check proves
  │ │└───┴─────┘│           │         │ │     warp disjointness
  │ └───────────┴───────────┴─────────┘ │     within a slot
  └─────────────────────────────────────┘
```

## Summary

| Aspect | Before | Implemented |
|--------|--------|-------------|
| **Warp awareness** | `ConvertLayoutOp` scratch only (`isCvtDimSync`) | General shared memory ops via `warpsPerCTA` comparison + bijection argument |
| **Detection** | Trivial-over-warp check | `warpsPerCTA` comparison with trailing-1 normalization |
| **Async/TDM** | Not considered | TDM `warpsPerCTA` via `tdmGetWarpDistribution`; consumer `warpsPerCTA` from distributed encoding |
| **MMA operands** | Assumed cross-warp | Checkable; batched MMA with matching distributions detected |
| **Batched MMA** | Not distinguished from standard MMA | Detected: warps across batch dimension → matching `warpsPerCTA` → barrier suppressed |
| **Padded layouts** | Not considered | Handled natively (bijection argument is encoding-agnostic) |
| **Swizzled layouts** | Not considered | Handled natively (bijection argument: swizzle permutes but never collapses addresses) |
| **Scope** | Situational | Currently TDM op pairs only; extensible to `local_store`/`local_load` |
| **GF(2) approach** | N/A | Not implemented; superseded by simpler `warpsPerCTA` comparison |
