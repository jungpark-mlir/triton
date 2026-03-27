# WMMA Dot Lowering Loop Schedule Guide

## Overview

The WMMA dot lowering pass in `WMMA.cpp` emits WMMA (Wave Matrix Multiply-Accumulate) intrinsics for AMD GPUs during Triton's conversion from TritonGPU IR to LLVM IR. The pass lowers `DotOp` and `DotScaledOp` operations into sequences of WMMA intrinsic calls over a 4-dimensional iteration space: **B** (batch), **M** (rows), **N** (columns), **K** (reduction).

Two environment variables control how the pass schedules the emission of these intrinsic calls at compile time.

## Environment Variables

### `TRITON_AMD_WMMA_LOOP_ORDER`

Controls the nesting order of the B/M/N/K loop axes when emitting WMMA intrinsics. The outermost axis in the string is iterated first (slowest-varying); the innermost axis is iterated last (fastest-varying).

**Default:** `kbmn` (K outermost, then B, M, N innermost)

**Format:** A 4-character permutation of the letters `b`, `m`, `n`, `k` (case-insensitive). Non-axis characters are stripped before parsing.

**Examples:**

```bash
# Default behavior — K is the outermost loop
export TRITON_AMD_WMMA_LOOP_ORDER=kbmn

# Tile (M/N) outermost, K innermost
export TRITON_AMD_WMMA_LOOP_ORDER=bmnk

# N outermost
export TRITON_AMD_WMMA_LOOP_ORDER=nbmk

# Unset to use default
unset TRITON_AMD_WMMA_LOOP_ORDER
```

**Legacy aliases:**

| Alias value     | Equivalent |
|-----------------|------------|
| `tile_k`, `mnk` | `bmnk`     |
| `k_outer`       | `kbmn`     |

**Fallback:** If the value is malformed (wrong length, duplicate axes, unknown characters), the default `kbmn` order is used silently.

---

### `TRITON_AMD_WMMA_SUBTILE`

Groups consecutive tiles along M, N, and K into subtile blocks before applying the loop order. Within each subtile block, the iterations stay in their natural order; the loop-order sorting applies across blocks.

**Default:** `1x1x1` (no subtiling; each tile is its own block)

**Format:** Three positive integers separated by `x` or `,`, in the order `MxNxK`. Values are in tile counts (not element counts).

**Examples:**

```bash
# Group 2 M-tiles, 2 N-tiles, 1 K-tile into each subtile block
export TRITON_AMD_WMMA_SUBTILE=2x2x1

# Group 4 K-tiles together
export TRITON_AMD_WMMA_SUBTILE=1x1x4

# Comma-separated is also valid
export TRITON_AMD_WMMA_SUBTILE=2,2,1

# Unset to disable subtiling
unset TRITON_AMD_WMMA_SUBTILE
```

**Fallback:** If the value is malformed (wrong number of parts, non-positive integers, unparseable), the default `1x1x1` is used silently.

---

## How It Works

### Tile Collection Phase

Before emitting any WMMA intrinsics, the pass iterates over the output register layout and collects a list of accumulator tiles. Each tile records its register offset and its (B, M, N) coordinates. This is done for both `DotOp` (struct `AccTile`) and `DotScaledOp` (struct `ScaledAccTile`).

### Work Item Construction

Each tile is crossed with every K repetition to produce a flat list of **work items** — `(tileIdx, k, bIdx, mIdx, nIdx)` tuples. The B/M/N indices are ordinal positions derived from sorting the unique coordinate values that appear in the tile list.

### Sorting

The work items are stable-sorted with a two-pass comparator:

1. **Block pass:** For each axis in the loop-order string (outermost first), compare `index / step` where `step` comes from the subtile shape. This groups iterations into subtile blocks and orders the blocks according to `TRITON_AMD_WMMA_LOOP_ORDER`.

2. **Intra-block pass:** For each axis in the same order, compare `index % step`. This preserves the natural ordering of iterations within each subtile block.

Ties are broken by original `tileIdx`.

### Emission

After sorting, each work item is emitted in order: accumulator values are loaded from the `fc` register file, operands A and B (and scales for `DotScaledOp`) are fetched for the given (B, M/N, K) coordinate, the WMMA intrinsic is called, and results are written back to the `fc` register file.

Both `convertDot` and `convertScaledDot` use the same `emitScheduledWMMA` template function.

---

## Usage

Set the environment variables before invoking the Triton compiler (they are read at compile time via `std::getenv`):

```bash
# Example: K-outermost with 2x2x1 subtiling
export TRITON_AMD_WMMA_LOOP_ORDER=kbmn
export TRITON_AMD_WMMA_SUBTILE=2x2x1
python my_triton_kernel.py

# Example: tile-outermost (accumulator-stationary)
export TRITON_AMD_WMMA_LOOP_ORDER=bmnk
unset TRITON_AMD_WMMA_SUBTILE
python my_triton_kernel.py
```

These variables apply globally to all WMMA dot lowerings in the compilation.
