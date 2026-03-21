# Membar: Warp-Local Shared Memory Access

## Problem

Triton's membar analysis treats shared memory as a flat address space shared by
all threads in a CTA. When a `local_store` and `local_load` touch the same
allocation, membar inserts a CTA-wide barrier (`__syncthreads()` / `s_barrier`)
even if the layout guarantees that each warp only accesses its own partition.
In such cases, the barrier is unnecessary — there is no cross-warp data
dependency.

```
Warp 0: writes [0x000, 0x100)     reads [0x000, 0x100)
Warp 1: writes [0x100, 0x200)     reads [0x100, 0x200)
Warp 2: writes [0x200, 0x300)     reads [0x200, 0x300)
                                                          ← no cross-warp overlap
                                                          ← barrier is unnecessary
```

## Background: How Addresses Are Computed

### Composed Layout

For `local_store` / `local_load`, the lowering computes a **composed layout**:

```cpp
auto regLayout = toLinearLayout(regTy);       // register layout: maps logical elements to threads
auto sharedLayout = toLinearLayout(memDescTy); // shared layout: maps logical elements to offsets
auto cvt = regLayout.invertAndCompose(sharedLayout);
// cvt: {register, lane, warp, block} → {offset}
```

This `cvt` maps thread coordinates `(register_idx, lane_id, warp_id,
block_id)` to shared memory byte offsets. The actual address for each thread
is computed by `applyLinearLayout(cvt, {kReg, kLane=laneId, kWarp=warpId,
kBlock=blockId})`.

### Warp Dimension in the Address Function

The `kWarp` input dimension contributes some bits to the `kOffset` output.
How it contributes determines whether warps access overlapping or disjoint
address ranges:

- **Overlapping (typical)**: The `kWarp` bits are mixed with `kLane`/`kReg`
  bits in the offset computation (e.g., via swizzling). Different warps may
  access the same offsets, and a CTA barrier is required.

- **Disjoint (warp-local)**: The `kWarp` bits map to high-order offset bits
  that partition the address space. Each warp accesses a non-overlapping
  region. No CTA barrier is needed — at most a `warp.sync` / `__syncwarp()`.

### Existing Warp-Level Sync Detection

Triton already has a partial version of this concept for `ConvertLayoutOp`
scratch buffers:

```cpp
// lib/Analysis/Membar.cpp, line 336
if (auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
  auto kWarp = StringAttr::get(op->getContext(), "warp");
  isWarpSync = mlir::isCvtDimSync(srcLayout, dstLayout, kWarp);
}
```

`isCvtDimSync` checks if the conversion's composed layout is **trivial over
the warp dimension** (identity mapping + no broadcasting). When true, membar
avoids clearing CTA-wide pending dependencies and instead only syncs within
the warp.

This mechanism is limited to `ConvertLayoutOp` scratch buffers and does not
apply to general `local_store` / `local_load` pairs.

## Proposed Detection

### Core Idea

For a pair of shared memory operations (e.g., `local_store` + `local_load`)
that membar considers as potentially conflicting, determine whether the
address sets are **warp-disjoint**: the set of offsets accessed by warp W in
one operation is disjoint from the set accessed by warp W' (W ≠ W') in the
other.

### Formal Condition

Given two composed layouts `cvt_A` and `cvt_B` (for the write and read ops),
each mapping `{register, lane, warp, block} → {offset}`:

**No cross-warp conflict exists if and only if:**
For all warp ids `w ≠ w'`, the sets `Addr_A(w)` and `Addr_B(w')` are
disjoint, where:

```
Addr_X(w) = { cvt_X(r, l, w, b) | for all register indices r, lane ids l }
```

### Detection via LinearLayout

In the `LinearLayout` framework, warp-disjointness can be detected by
analyzing the basis vectors. A sufficient condition:

1. Extract the `kWarp → kOffset` sublayout from both composed layouts.
2. Check that the `kWarp` bases map to offset bits that are **not** covered
   by the `kReg` or `kLane` bases — i.e., the warp selects an address
   partition that is independent of the within-warp thread/register indexing.

More precisely, let `W_A` and `W_B` be the sets of offset bits influenced by
`kWarp` in `cvt_A` and `cvt_B` respectively, and let `T_A` and `T_B` be the
sets of offset bits influenced by `kReg + kLane`. If `W_A ∩ T_B = ∅` and
`W_B ∩ T_A = ∅` and `W_A = W_B`, then the warp dimension partitions the
address space identically in both ops, and no cross-warp conflict exists.

This can be checked by inspecting the `LinearLayout` basis matrices without
materializing actual addresses.

### Simpler Sufficient Condition

A practically useful sufficient condition that covers common cases:

For a single composed layout `cvt`, the access is **warp-local** if:
- `cvt` restricted to `{kWarp}` produces offset values that are all multiples
  of the per-warp address range size, and
- The per-warp address range `|{cvt(r, l, w, b) | ∀r, ∀l}|` for any fixed `w`
  is less than or equal to the stride between warp partitions.

In practice, this means the warp ID selects a "chunk" of shared memory, and
within that chunk, register and lane indices tile the addresses.

## Integration with Membar

### Option A: AllocationSlice with Warp Partitioning

Extend `AllocationSlice` to carry per-warp interval information:

```cpp
struct AllocationSlice {
  // ... existing fields ...
  std::optional<WarpPartitionInfo> warpPartition;
};

struct WarpPartitionInfo {
  int64_t warpStride;    // byte stride between warps
  int64_t perWarpSize;   // bytes each warp accesses
  int numWarps;
};
```

In `intersects()`, if both slices have `WarpPartitionInfo` with matching
parameters, they don't intersect (each warp only conflicts with itself, and
within a warp, execution is lockstep — no barrier needed).

**Pros**: Minimal change, fits existing `AllocationSlice` model.

**Cons**: Requires constructing the composed layout at analysis time (membar
currently works at the allocation level, not the lowering level). May need
layout information that is not easily available in `AllocationSlice`.

### Option B: Layout-Aware Filter

Add a **filter function** (similar to the AMD async-write filter) that checks
warp-disjointness for a pair of operations:

```cpp
bool isWarpDisjoint(Operation *write, Operation *read, Allocation *alloc) {
  auto writeMemDesc = getMemDescType(write);
  auto readMemDesc = getMemDescType(read);
  if (!writeMemDesc || !readMemDesc)
    return false;
  auto writeCvt = computeComposedLayout(write);
  auto readCvt = computeComposedLayout(read);
  return checkWarpDisjointness(writeCvt, readCvt);
}
```

**Pros**: Composable with existing filter mechanism. Can be backend-specific.

**Cons**: Requires computing composed layouts during membar analysis, which
currently happens later during lowering to LLVM. The layout composition logic
would need to be available earlier.

### Option C: Pre-Analysis Annotation

Run a pre-pass before membar that annotates `local_store`/`local_load` ops
with a `warp_disjoint` attribute when the layout guarantees warp-local access.
Membar then checks this attribute.

**Pros**: Clean separation. Layout analysis happens once, membar consumes a
simple boolean.

**Cons**: Another pass to maintain. Attribute could be lost by transforms
between annotation and membar.

## When Does This Apply?

### Cases Where Warps Access Disjoint Shared Memory

1. **Blocked register layout with matching shared layout**: If the register
   layout assigns each warp a contiguous block of elements, and the shared
   layout maps elements to contiguous offsets in the same order, the warp
   dimension naturally partitions the shared memory.

2. **Warp-specialized kernels**: Gluon kernels where different warps perform
   different roles (e.g., producer vs consumer) and explicitly partition
   shared memory by warp ID.

3. **Per-warp scratch buffers**: Temporary shared memory used for warp-local
   reductions or shuffles where the data never crosses warp boundaries.

4. **`ConvertLayoutOp` with warp-trivial conversion**: Already handled by
   `isCvtDimSync` for scratch buffers.

### Cases Where This Does NOT Apply

1. **`SwizzledSharedEncodingAttr`**: The documentation explicitly states that
   all threads in `{0, ..., 32*num_warps-1}` may access any element. Swizzling
   mixes warp and lane bits in the offset, making the access cross-warp by
   design.

2. **Matrix multiply operands** (`DotOperandLayout`): Typically loaded by
   multiple warps cooperatively, with each warp reading a different part of
   the shared buffer but potentially overlapping with adjacent warps due to
   the MMA instruction's data consumption pattern.

3. **NVIDIA `async_copy_global_to_local`**: Each thread issues its own
   `cp.async` instruction. The destination address depends on the shared
   encoding, which is typically cross-warp (`SwizzledSharedEncodingAttr`).

## Async Copy, TDM Copy, and Async Wait

### TDM Copy (AMD gfx1250)

TDM (Tensor Data Mover) copies are **inherently warp-partitioned on the
write side**. The TDM linear layout distributes the block across warps:

```cpp
// lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp
getTDMLinearLayout = identityStandardND("message", messageShape, order)
                   * identityStandardND("warp", warpsPerCTA, order)
                   * cgaLayout;
```

The `warp` dimension is a higher-order factor than `message` (per-warp data
elements). Each warp's DMA instruction targets a distinct sub-block of the
tensor, producing non-overlapping shared memory writes. The per-warp
sub-block shape is computed by `tdmGetAdjustedBlockShape`, which divides
the block dimensions by the warp distribution.

The composed mapping `tdmLayout.invertAndCompose(sharedLayout)` determines
which shared memory offsets each warp's DMA writes to. Because the TDM
layout assigns each warp a disjoint sub-block, the destination addresses
are warp-disjoint by construction.

**However**, the consumer (`local_load`) uses a different layout — its
composed mapping `regLayout.invertAndCompose(sharedLayout)` depends on the
register encoding of the result tensor (e.g., `BlockedLayout`,
`AMDWMMALayout`, `AMDMFMALayout`). The reader's warp-to-address mapping
may NOT match the writer's:

```
TDM write:   warp 0 → offsets [0x000, 0x100)    warp 1 → [0x100, 0x200)
local_load:  warp 0 → offsets [0x000, 0x080) ∪ [0x100, 0x180)   ← cross-warp!
```

In this (common) scenario, warp 0 reads data written by both warp 0 and
warp 1, so a CTA barrier is still needed. The warp-local optimization only
applies when the reader also has a matching warp partition.

### NVIDIA `async_copy_global_to_local`

NVIDIA's `cp.async` is a **per-thread** async copy instruction. Each thread
independently copies its assigned elements from global to shared memory.
The shared memory address per thread is determined by the shared encoding
and the source tensor's layout.

Whether the resulting write pattern is warp-disjoint depends on the shared
encoding. With `SwizzledSharedEncodingAttr`, the swizzling mixes lane and
warp bits, making writes cross-warp. With other encodings it could be
warp-partitioned, though this is uncommon in practice.

### `async_wait` Semantics

`async_wait` ensures that outstanding async DMA operations have completed
and their results are visible in shared memory. Key points:

- **AMD**: `async_wait` makes DMA results visible **CTA-wide** — all warps
  can see the data, not just the issuing warp. This is a memory visibility
  guarantee, not an execution barrier (no CTA execution sync).
- **NVIDIA**: `cp.async.wait_group` waits for the issuing thread's async
  copies. A subsequent `__syncthreads()` is needed for cross-warp visibility.

For the warp-local optimization, `async_wait` is not a concern:

- If both the DMA write and the subsequent `local_load` are warp-disjoint
  with matching partitioning, each warp only reads data that it wrote (or
  that was made visible to it after the wait). No cross-warp data flow
  means no CTA barrier is needed.
- The existing AMD membar filter already handles the DMA-completion aspect
  (suppressing false RAW barriers when the `local_load` token chains to an
  `async_wait`). Warp-disjointness is an orthogonal check that could
  further reduce barriers even without the token-chain filter.

### Interaction Summary

| Operation | Write Warp-Partitioned? | Read Warp-Partitioned? | Barrier Needed? |
|-----------|:-----------------------:|:----------------------:|:---------------:|
| TDM copy → local_load (matching partition) | Yes | Yes | **No** |
| TDM copy → local_load (MMA layout) | Yes | Typically no | **Yes** |
| NVIDIA async_copy → local_load | Depends on encoding | Depends on encoding | Usually **yes** |
| local_store → local_load (warp-local) | Depends on layout | Depends on layout | **No** if both warp-local |

The key takeaway: **TDM copy is always warp-partitioned on the write side,
but this alone is insufficient** — the read side must also be warp-partitioned
with a matching scheme. The most common consumer layout (MMA operand) is
cross-warp, so the practical benefit is limited to cases where the reader
layout happens to align with the TDM write partition (e.g., blocked layouts
with warp-major ordering, or Gluon kernels with explicit warp partitioning).

## Challenges

### Layout Availability at Membar Time

Membar runs during the `TritonGPUToLLVM` conversion, where `MemDescType`
encodings are available. However, the **composed layout** (`regLayout.
invertAndCompose(sharedLayout)`) requires knowing the register-side layout
of the operation's tensor operand, which may not be trivially available from
the `MemDescType` alone.

For `local_store`, the source tensor has a register layout. For `local_load`,
the result tensor has a register layout. Both are needed to compute the
composed layout and determine warp-disjointness.

### Interaction with Multi-Buffering

If the allocation is multi-buffered (e.g., `memdesc<3x128x128xf16>`) and
accessed via `MemDescIndexOp`, the warp-disjointness check applies to the
sub-buffer level. The `MemDescIndexOp` selects the buffer slot (handled by
`BufferIndexExpr`), and within each slot, the warp-disjointness check
determines whether a barrier is needed.

These are orthogonal optimizations that compose naturally.

### Shared Encoding Design Intent

Most shared memory encodings in Triton are designed for **cross-warp
cooperative access** — this is the fundamental purpose of shared memory in
GPU programming. Warp-local shared memory access is a special case that
arises in specific optimization patterns (warp-level scratch, warp
specialization). The barrier elimination opportunity is real but
situational.

## Summary

| Aspect | Current State | Proposed |
|--------|--------------|----------|
| **Warp awareness in membar** | None for `local_store`/`local_load`; only `ConvertLayoutOp` scratch via `isCvtDimSync` | Extend to general shared memory ops via composed layout analysis |
| **Detection mechanism** | N/A | Check warp-disjointness of composed layouts using `LinearLayout` basis analysis |
| **Integration** | N/A | Filter function (Option B) or pre-analysis annotation (Option C) |
| **Applicability** | N/A | Blocked layouts with warp partitioning, warp-specialized kernels, per-warp scratch |
| **Limitation** | N/A | Most shared encodings are cross-warp by design; layout composition must be available at membar time |
