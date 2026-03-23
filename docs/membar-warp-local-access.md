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

## Detection via GF(2) Linear Independence

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

The GF(2) check handles both cases correctly.

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
register/lane bases address only within that block.

### Detection

The GF(2) independence check applies directly:

- **TDM write side**: warp-disjoint by construction (TDM layout assigns
  each warp a contiguous row block).
- **Local_load (WMMA operand) side**: the composed layout maps warps to
  different row blocks. If the shared encoding is non-swizzled, warp bases
  map to high-order row-offset bits, independent of register/lane bases.

The conditions for this to work are the same as for standard MMA operands:
non-swizzled (or warp-compatible swizzled) shared encoding required. See
"Padded and Non-Power-of-2 Shared Layouts" below for an additional
consideration relevant to this case.

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
| TDM copy → local_load (batched MMA, warps across batch) | Yes | Yes | **No** |
| TDM copy → local_load (standard MMA, K broadcast) | Yes | Usually no | **Yes** |
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

3. **Batched MMA with warps across batch dimension**: Each warp executes
   an independent MMA on its own partition (e.g., FA MQA decode with
   split-k). No K-dimension broadcast — strictly simpler than case 2.

4. **Warp-specialized Gluon kernels**: Explicit per-warp shared memory
   partitioning.

5. **Per-warp scratch buffers**: Warp-local reductions or shuffles.

### When This Does NOT Apply

1. **Swizzled shared encodings** (for GF(2) check): XOR-mixing of row/column
   bits breaks warp independence in the `LinearLayout` representation.

2. **Padded shared encodings** (for GF(2) check): Non-power-of-2 row strides
   are not faithfully representable in GF(2) arithmetic. The warp-disjointness
   property still holds geometrically — use the integer row-range check or
   unpadded-equivalent check instead (see above).

3. **Cross-warp MMA operand loading**: When the MMA layout requires each warp
   to read data from multiple warp regions.

4. **Cooperative DMA patterns**: Where the shared encoding distributes writes
   across all warps.

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

**However, the GF(2) detection mechanism cannot verify it.** `LinearLayout`
computes offsets via XOR (GF(2) addition), which equals integer addition
only when there are no carries. A row stride with multiple bits set (e.g.,
144 = `0b10010000`) produces carry interactions when multiple row-index
bits are combined:

```
  GF(2) vs integer arithmetic for stride = 3 (0b11):

  row = 1:  GF(2): 3     integer: 3     ✓ (matches)
  row = 2:  GF(2): 6     integer: 6     ✓ (matches)
  row = 3:  GF(2): 3⊕6=5 integer: 9     ✗ (carry breaks it)
```

When `toLinearLayout` produces basis vectors for such a layout, the XOR
composition does not faithfully reproduce the actual integer address
computation. The GF(2) independence check may give incorrect results.

**Alternative detection approaches for padded layouts:**

1. **Row-range check (integer arithmetic).** If warps are distributed
   along a single contiguous dimension (e.g., rows) and each warp gets a
   contiguous block of rows, the byte ranges `[w * rows_per_warp * stride,
   (w+1) * rows_per_warp * stride)` are trivially disjoint regardless of
   stride value. This check uses integer arithmetic and does not require
   `LinearLayout` at all.

2. **Check the unpadded equivalent.** If the only difference is the row
   stride (padding), verify GF(2) independence on the unpadded layout. If
   warp bases are independent in the unpadded version, they are independent
   in the padded version too — padding widens each warp's byte range
   without causing overlap.

3. **Use encoding metadata directly.** `PartitionedSharedEncodingAttr`
   (used for TDM on gfx1250) carries the warp-partitioning information
   explicitly. The check could read the partitioning from the encoding
   rather than deriving it from basis vectors.

For the FA MQA decode case, approach 1 is the most practical: TDM
distributes warps along the row dimension, and each warp's WMMA reads
from its own contiguous row range. The stride value is irrelevant to
disjointness — only the row assignment matters.

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
| **Detection** | Trivial-over-warp check | GF(2) Gaussian elimination; integer row-range check for padded layouts |
| **Async/TDM** | Not considered | TDM write-side disjoint; full result requires checking consumer layout too |
| **MMA operands** | Assumed cross-warp | Checkable per-configuration via composed layout |
| **Batched MMA** | Not distinguished from standard MMA | Simpler case: no K-broadcast, each warp fully independent |
| **Padded layouts** | Not considered | Property holds; GF(2) inapplicable, use integer row-range check |
| **Scope** | Situational; most encodings are cross-warp by design | Same, but catches cases where layout happens to be warp-local |
