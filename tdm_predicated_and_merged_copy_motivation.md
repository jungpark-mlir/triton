# Motivation for Predicated and Merged TDM Copy

This note explains why we want two related capabilities for AMD TDM copies:

1. predicated / partial TDM copy, where only a selected subset of warps performs useful copy work, and
2. merged TDM copy, where multiple compatible predicated copies are lowered into one TDM instruction.

The goal is not to expose a general tuning knob that users should try for every TDM copy. For a plain TDM copy, the default remains to omit the hint and let all warps participate. The mask is useful when the kernel already has a reason to assign different roles to different warp groups.

## Hardware Model

The gfx1250 TDM copy instruction is issued at warp granularity and uses descriptor fields that are effectively per warp. Today, a single high-level `tdm.async_load` copies one logical block into shared memory, and the compiler distributes that block across all warps in the CTA.

That default is the right behavior for ordinary copies. However, the hardware can also support cases where only some warps perform useful work while other warps see a descriptor predicate of zero. Those inactive warps still execute the instruction in uniform control flow, but the hardware treats their copy as a no-op.

This is the basis for predicated TDM copy: keep the same logical copy, keep the same shared-memory layout, but change which warps are responsible for producing the data.

## Why Predicated / Partial Copy

Predicated TDM copy is needed for kernels where warp groups already have different roles.

The clearest case is warp-pipelining. For example, with eight warps split into an upper-half group and a lower-half group, one group may act as the leading producer while the other group is doing compute or belongs to a later pipeline stage. In that situation, we want the leading group to issue the TDM copy for the current stage and the other group to be predicated off.

Without this support, the copy is still distributed across all warps. That does not match the pipeline structure, because warps that should be available for another role still participate in the producer copy.

The important point is that `warp_used_hint` does not change the user-visible copy. The user still provides the normal layout for the full shared-memory block. During TDM-copy lowering, the compiler redistributes the producer side of the copy across the selected active warps. If `K` warps are active, the per-warp TDM tile is re-encoded as `block / K`, so those `K` warps still cover the same full block in one TDM instruction. The following `local_load` can use the original layout as usual.

This can also give the copy a longer latency window when multiple warps share a SIMD. With `num_warps = 8`, each SIMD has two resident warps:

```text
SIMD0: warp0, warp4
SIMD1: warp1, warp5
SIMD2: warp2, warp6
SIMD3: warp3, warp7
```

Without predication, both warps on a SIMD produce part of the LDS tile. The full tile cannot be consumed until the later warp's TDM copy has also completed:

```text
time ------------------------------------------------>

warp0:   g0(part 0)        ...        wait/load full LDS tile
warp4:              g4(part 4)        ...        wait/load full LDS tile
                  ^                           ^
                  |                           |
        warp4's part is issued later          full tile needs both parts
```

With `warp_used_hint = 0b00001111`, the first warp on each SIMD produces the full share for that SIMD pair, while the second warp is predicated off for this TDM copy:

```text
time ------------------------------------------------>

warp0:   G0(larger tile: covers old part0 + part4)  ...  wait/load full LDS tile
warp4:              pred=0 / no TDM data produced   ...  load same LDS tile
        ^                                             ^
        |                                             |
        data for both warp0 and warp4 is issued       consumers need it later
```

This mirrors the warp-pipelined case: producer responsibility moves to the earlier warp group, so the memory request has more time in flight before the later group reaches the `local_load`.

This avoids two less desirable alternatives:

- Emitting multiple TDM instructions with the original per-warp tile. That keeps shapes simple, but one logical `async_load` can become multiple TDM instructions and the wait-count behavior now depends on the hint.
- Asking the user to describe a larger virtual tensor or shared-memory shape and then mask off part of it. That can exceed the intended LDS budget or fail block-shape divisibility, and it changes the meaning of the copy. The hint would no longer be a pure performance hint.

Predicated copy gives us the intended behavior: the same logical block is copied, but the producer work is assigned to the warp group that should own it.

## Why Merged TDM Copy

Merged TDM copy is the natural follow-up once different warp groups can be predicated independently.

The hardware descriptor is per warp, so different active warp groups can carry different descriptor fields in the same physical TDM instruction. If two copies are compatible and their active warp masks are disjoint, the compiler can select the descriptor fields per warp and issue one TDM instruction instead of two.

A motivating example is a GEMM pipeline where one warp group loads A while another warp group loads B. Logically, those are separate copies because they have different sources and destinations. But if their warp masks are disjoint and their shared-memory destinations do not overlap, they can be represented by one combined TDM instruction:

- warps in mask A use descriptor A,
- warps in mask B use descriptor B,
- no warp belongs to both masks.

This reduces the number of issued TDM instructions while preserving the original logical copies. It also matches the intended warp-role structure of the kernel: different groups perform different producer work in the same pipeline step.

For this to be correct, merging must be limited to cases with clear compatibility:

- the participating masks are disjoint,
- the copies do not have an intervening TDM wait or TDM operation that orders them,
- the shared-memory destinations do not overlap,
- mbarrier cases are left out initially, because arrival-count rewriting is a separate problem.

The merged-copy pass should also interact carefully with wait counts. In compiler-managed pipelined code, the wait-count pass can recompute counts after merging. For hand-written kernels, either the user must write waits consistent with the merged emission, or the merge representation must preserve enough information for the wait-count pass to translate from logical operations to physical instructions.

## What This Does Not Claim

This is not intended as a general user tuning knob for ordinary TDM copies. If a single load is not warp-pipelined and cannot be merged with another copy, there is no clear motivation to try different masks. The normal all-warp copy remains the expected default.

This also does not require a separate "one warp group" shared-memory layout. The layout describes the full shared block. Predication only changes the producer-side TDM lowering, and the data is still written into the same layout that the consumer reads.

In the long run, end users should not need to set these masks manually. A compiler pass should derive them when the kernel structure makes the motivation clear, such as warp-pipelined producer groups or mergeable disjoint copies.
