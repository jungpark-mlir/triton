"""GFX1250 Gluon GEMMs with race-free TDM refill and cluster synchronization.

This module is a parallel implementation of ``kernels.py`` that repairs a
read/write race present in all four families there. Host-side glue, layouts,
launch parameters, and the benchmark protocol are identical, so the two modules
are directly comparable.

The race is not theoretical. At M=N=2048, K=4096, ``kernels.py``'s MXFP8 fails
its own correctness check in roughly 60% of runs, with a mismatch count that
varies from run to run. The other three pass at that size but are structurally
identical, so they are latent rather than safe. Every family here passes 10 of
10 runs.

Cost. BF16 is free: 694.5 us before, 694.7 us after, inside run-to-run spread.
The other three lose measured performance, consistently across three
order-balanced rounds at M=N=4096, K=65536:

    MXFP8       261.7 -> 292.0 us   (+11.6%)
    FP8xMXFP4   221.8 -> 234.2 us   (+5.6%)
    MXFP4       146.9 -> 163.6 us   (+11.4%)

That cost is structural, not an artifact of this implementation. See "TDM lead"
below.

The race. ``ClusterBarrierArriveOp`` lowers to ``s_barrier_signal -3`` guarded
by ``warpId == 0``, while ``ClusterBarrierWaitOp`` lowers to an unguarded
``s_barrier_wait -3``. Only warp 0 signals. ``kernels.py`` places the arrive at
the end of a slot's final read stage and then issues that slot's own refill in
the following compute stage. Because waves 4-7 trail waves 0-3 by one pipeline
stage, warp 0 signals that the slot is drained while waves 4-7 are still
issuing reads against it, so the refill can overwrite operands that are still
being read.

The repair has two parts, and both are needed.

* Deferred refill. A tile refills the slot consumed by the *previous* tile
  rather than its own. That slot's reads completed a full tile earlier, so the
  destination is genuinely free. Tile t's refill supplies tile t+1, so only the
  first slot is prefetched and the last tile issues no refill.
* Adjacent arrive/wait. The pair sits together at the top of the tile's first
  read stage, where warp 0 has provably passed the previous tile in full. This
  also keeps arrives and waits strictly alternating, one pair per refilling
  tile, which ``cluster.arrive``/``cluster.wait`` require and which the
  original placement violated.

A same-slot MXFP8 exception was tested and rejected. Although repeated long-K
determinism passed, the phase-shifted wave groups put the leading refill in the
same pipeline phase as the trailing group's high LDS reads. A static barrier
sequence and finite determinism runs do not prove same-iteration drainage.

Two further notes on what did *not* help, both measured on BF16. Hoisting the
arrive earlier to buy slack is unsafe for the reason above. Moving the tile's
own reads inside the arrive/wait window so the rendezvous overlaps them is
safe and verified, but 0.08% slower over seven paired runs: each CTA has a full
tile of work between consecutive barriers, so the wait is not blocking and
there is no latency to hide.

TDM lead, and why three families pay for correctness. Warp 0 may only signal
once waves 4-7 have finished the slot being refilled, and the earliest point
that holds is the top of the following tile. With S slots, the slot freed by
tile t-1 is next read at tile t+S-1, so the refill window is exactly S-1 tiles.
No race-free schedule can beat that bound. The racy schedule issued at the end
of tile t for tile t+S and so spanned about S-0.75 tiles, a quarter-tile more.
For BF16 that margin is slack. For the other three the transfer very nearly
fills the window already, so losing a quarter tile puts them over, the TDM
engine stops being saturated, and throughput drops off a cliff.

The obvious remedy, one more slot, does not fit. Measured requirements against
the 327,680-byte limit: MXFP8 needs 430,456 at three slots, MXFP4 needs 443,128
at three, and FP8xMXFP4 overflows at four. All three are already at the LDS
ceiling, which is why they carry the slot counts they do.

MXFP8 therefore retains the deferred opposite-slot refill at the top of the
read pipeline, where the destination was drained by the previous tile.

Wait placement is the lever, and it is worth more than any of the above. The
S-1 tile window above is only an upper bound; a loop realizes it only if the
wait lets those S-1 tiles stay outstanding. Put the wait before the refill and
it cannot: the only descriptors outstanding at that point belong to the tile
about to be read, so the count has to be 0, the drain is total, and the engine
cannot even start the next refill until it finishes. Put it after, and both
tiles are outstanding, the FIFO retires the older set, and the count becomes
ops_per_tile * (S - 1). On MXFP8 that is 3 rather than 0 and it is worth 20 us
at 4096x4096x65536, with output bit-identical to the drained form.

A further option, untried, is to halve the TDM block, four K128 slots in the
LDS two K256 slots occupy today. That widens the window without more memory,
but it halves the contiguous transfer to 128 bytes per row and so doubles GL2
requests, which is the L2 efficiency this project spent effort protecting. It
is a real restructure, not a tuning knob, and it is not attempted here.

BF16 consumes its second K64 as one WMMA group. MXFP8, FP8xMXFP4, and MXFP4
issue deferred refills against slots drained by earlier tiles.

This module owns four optimized GEMM families:

* ``bf16``: BF16 operands with FP32 accumulation and BF16 output.
* ``mxfp8``: E4M3 operands with block-32 E8M0 scales.
* ``fp8_mxfp4``: unscaled E4M3 A and block-scaled packed E2M1 B.
* ``mxfp4``: packed E2M1 operands with block-32 E8M0 scales.

Every default kernel uses eight warps and a 4x4 CTA cluster. A cluster computes
one 1024x1024 output tile, so each CTA owns a 256x256 partition. Operand tiles
and scales are distributed across the cluster through CGA-aware shared-memory
layouts; TDM moves global tiles to and from that distributed storage.

The device functions intentionally remain explicit and format-specific.
Statement order around ``warp_pipeline_stage``, cluster barriers, TDM issue,
and ``tdm.async_wait`` is part of the performance contract, not stylistic
boilerplate. Before changing those regions, compare correctness, executable
AMDGCN, register/scratch use, and locked benchmark medians.

The ``snapshot_*`` helper prefix is temporarily retained because this package
was promoted losslessly from the validated experiment. It identifies source
lineage only; this package is now the canonical maintained owner.

Measured here at M=N=4096, K=65536: approximately 3.2 PFLOPS for BF16,
7.5 PFLOPS for MXFP8, 9.4 PFLOPS for FP8 x MXFP4, and 14.0 PFLOPS for MXFP4.
Performance depends on input data and GPU DVFS state; see README.md for the
reproducible benchmark protocol and current results.
"""

# Deferred-refill slot algebra, shared by all four families.
#
# ``kernels.py`` prefetches all S slots, then tile t refills slot t % S with
# the data for tile t + S. Here the prologue fills only slots 0 .. S-2, and
# tile t refills the slot that tile t - 1 consumed:
#
#     consumed slot   = t % S
#     refilled slot   = (t - 1) % S      == (t + S - 1) % S
#     refilled tile   = t + S - 1
#
# Tile 0 refills slot S - 1, the one the prologue left empty, so the loop head
# needs no special case. The loop runs while the supplied tile exists,
# t + S - 1 <= iter_max - 1, giving ``range(0, iter_max - S + 1)`` refilling
# tiles against ``range(0, iter_max - S)`` before. That extra refilling tile
# replaces one drain tile, so the drain shrinks from S to S - 1.
#
# TDM lead is unchanged for S = 2: the refill moves from the end of tile t to
# the start of tile t, and its consumer moves from tile t + 2 to tile t + 1, so
# both the old and new schedules keep four pipeline stages of lead. For S = 3
# the lead necessarily drops by one tile, because the slot freed by tile t - 1
# is next read at tile t + S - 1 rather than t + S.

import argparse
import gc
from pathlib import Path
import sys
import time

import torch
import triton
from triton._C.libtriton.gluon_ir import make_cga_layout
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm
from triton.tools.mxfp import MXFP4Tensor


REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from third_party.amd.python.examples.gluon.mxfp_gemm_cdna5 import (  # noqa: E402
    MXScaleTensor,
    init_data,
    torch_gemm_mxfp,
)


BLOCK_M = 1024
BLOCK_N = 1024
# K depth differs by format because packed E2M1 stores two values per byte and
# scaled WMMA consumes a different logical K span than ordinary BF16 WMMA.
BF16_BLOCK_K = 128
MXFP8_BLOCK_K = 256
MXFP4_BLOCK_K = 512
# These are retained launch invariants for the peak 4x4-cluster paths.
NUM_WARPS = 8
NUM_BUFFERS = 2
CTA_M = 4
CTA_N = 4
GROUP_SIZE_M = 4
NUM_XCDS = 8
SCALE_BLOCK = 32
PRESHUFFLE_FACTOR = 128
AGPR_ATTRS = (("amdgpu-agpr-alloc", "0,0"),)
KERNEL_NAMES = ("bf16", "mxfp8", "fp8_mxfp4", "mxfp4")
EXPERIMENTAL_KERNEL_NAMES = (
    "bf16_8stage",
    "mxfp8_bk128",
    "mxfp8_bk256_opt",
    "mxfp8_bk512_128",
    "mxfp8_512x128x128_cga1",
    "mxfp8_64x64x512_cga1",
    "mxfp8_64x64x512_cga2x2",
    "mxfp8_64x64x512_cga4x4",
    "mxfp8_128x64x512_cga4x4",
    "mxfp8_64x128x512_cga4x4",
    "mxfp8_64x64x1024_cga2x2",
    "mxfp8_128x128x256_cga4x4",
)


# ---------------------------------------------------------------------------
# Shared grid mapping and output epilogue.
#
# Programs are first spread across eight XCDs, then grouped along one output
# dimension to improve operand reuse. The default mapping reuses B while the
# ``reuse_a`` form swaps grouping axes for the mixed FP8 x MXFP4 kernel.
#
# The output helper converts FP32 accumulators to the requested output type in
# distributed LDS, then performs one cluster-wide TDM store. Eight elements of
# row padding avoid the bank conflicts of an unpadded identity layout.

@gluon.jit
def snapshot_get_xcd_swizzled_pids(
        M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr,
        GROUP_SIZE_M: gl.constexpr):
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    if NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = (tall_xcds * pids_per_xcd
                   + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@gluon.jit
def snapshot_get_xcd_swizzled_pids_reuse_a(
        M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr,
        GROUP_SIZE_N: gl.constexpr):
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    if NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = (tall_xcds * pids_per_xcd
                   + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid)
    num_pid_in_group = GROUP_SIZE_N * num_pid_m
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
    pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size_n)
    pid_m = (pid % num_pid_in_group) // group_size_n
    return pid_m, pid_n


@gluon.jit
def snapshot_cluster_wait():
    gl.amd.gfx1250.cluster.arrive()
    gl.amd.gfx1250.cluster.wait()


@gluon.jit
def snapshot_tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        STORE_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, CTA_N: gl.constexpr):
    c_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N // CTA_N, 8]], [BLOCK_M, BLOCK_N], [1, 0],
        STORE_LAYOUT.cga_layout)
    c_shared = gl.allocate_shared_memory(
        c_ptr.type.element_ty, [BLOCK_M, BLOCK_N], c_shared_layout)
    c_shared.store(acc.to(c_ptr.type.element_ty))
    c_desc = tdm.make_tensor_descriptor(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        block_shape=(BLOCK_M, BLOCK_N), layout=c_shared_layout)
    tdm.async_store(
        c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], c_shared)
    tdm.async_wait(0)


# MXFP8 uses a half-width output stage. Two serial stores reduce accumulator
# partition width and let the output allocation reuse the dead operand arena.
@gluon.jit
def mxfp8_tdm_store_serial_n2(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        OUTPUT_CGA_LAYOUT: gl.constexpr, CTA_M: gl.constexpr,
        CTA_N: gl.constexpr):
    cta_m: gl.constexpr = 256
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[half_n, 8]], [CTA_M, CTA_N, cta_m, half_n],
        [3, 2, 1, 0], OUTPUT_CGA_LAYOUT)
    shared = gl.allocate_shared_memory(
        c_ptr.type.element_ty, [CTA_M, CTA_N, cta_m, half_n], shared_layout)
    acc4 = acc.reshape(
        (CTA_M, cta_m, CTA_N, cta_n)).permute((0, 2, 1, 3))
    output0 = gl.amd.slice(
        acc4, [CTA_M, CTA_N, cta_m, half_n], [0, 0, 0, 0])
    output1 = gl.amd.slice(
        acc4, [CTA_M, CTA_N, cta_m, half_n], [0, 0, 0, half_n])
    desc = tdm.make_tensor_descriptor(
        base=c_ptr,
        shape=(M // cta_m, N // cta_n, cta_m, cta_n),
        strides=(cta_m * stride_cm, cta_n * stride_cn, stride_cm, stride_cn),
        block_shape=(CTA_M, CTA_N, cta_m, half_n), layout=shared_layout)
    base_m = pid_m * CTA_M
    base_n = pid_n * CTA_N
    shared.store(output0.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, 0], shared)
    tdm.async_wait(0)
    shared.store(output1.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, half_n], shared)
    tdm.async_wait(0)


@gluon.jit
def mxfp8_tdm_store_split_n2(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1,
        OUTPUT_CGA_LAYOUT: gl.constexpr, CTA_M: gl.constexpr,
        CTA_N: gl.constexpr):
    cta_m: gl.constexpr = 256
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[half_n, 8]], [CTA_M, CTA_N, cta_m, half_n],
        [3, 2, 1, 0], OUTPUT_CGA_LAYOUT)
    shared = gl.allocate_shared_memory(
        c_ptr.type.element_ty, [CTA_M, CTA_N, cta_m, half_n], shared_layout)
    output0 = acc0.reshape(
        (CTA_M, cta_m, CTA_N, half_n)).permute((0, 2, 1, 3))
    output1 = acc1.reshape(
        (CTA_M, cta_m, CTA_N, half_n)).permute((0, 2, 1, 3))
    desc = tdm.make_tensor_descriptor(
        base=c_ptr,
        shape=(M // cta_m, N // cta_n, cta_m, cta_n),
        strides=(cta_m * stride_cm, cta_n * stride_cn, stride_cm, stride_cn),
        block_shape=(CTA_M, CTA_N, cta_m, half_n), layout=shared_layout)
    base_m = pid_m * CTA_M
    base_n = pid_n * CTA_N
    shared.store(output0.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, 0], shared)
    tdm.async_wait(0)
    shared.store(output1.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, half_n], shared)
    tdm.async_wait(0)


# ---------------------------------------------------------------------------
# BF16 x BF16.
#
# Aggregate tile: 1024x1024x128 across a 4x4 cluster.
# Pipeline: two BK128 LDS slots; each tile is consumed as two K64 halves.
# The refill targets the slot the previous tile consumed and is issued in the
# first read stage, behind an adjacent cluster arrive/wait pair. Because the
# refill no longer sits between two WMMA groups, the second K64 is consumed as
# a single WMMA group rather than being split into two K32 groups.

@gluon.jit
def bf16_issue_loads(a_desc, b_desc, a_dst, b_dst, BLOCK_K: gl.constexpr):
    tdm.async_load(
        a_desc, [0, 0], a_dst, warp_used_hint=0b00001111)
    tdm.async_load(
        b_desc, [0, 0], b_dst, warp_used_hint=0b00001111)
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K])
    return a_desc, b_desc


@gluon.jit
def bf16_consume_slot(
        a_buf, b_buf, slot, acc, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        BLOCK_K: gl.constexpr):
    half_k: gl.constexpr = BLOCK_K // 2
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(0, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(
            0, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(
            half_k, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(
            half_k, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    return acc


@gluon.jit
def bf16_cluster_consume_and_refill(
        a_buf, b_buf, slot, refill_slot, a_desc, b_desc, acc,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        BLOCK_K: gl.constexpr):
    half_k: gl.constexpr = BLOCK_K // 2
    tdm.async_wait(0)
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        # ``refill_slot`` was drained by the previous tile, so this K128 write
        # needs no in-tile guard. The arrive sits directly ahead of the wait
        # because only warp 0 signals it; hoisting it into the previous tile
        # would have warp 0 assert drainage while waves 4-7, one stage behind,
        # still read that slot.
        gl.amd.gfx1250.cluster.arrive()
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc = bf16_issue_loads(
            a_desc, b_desc, a_buf.index(refill_slot),
            b_buf.index(refill_slot), BLOCK_K)
        a = a_buf.index(slot).slice(0, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(
            0, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(slot).slice(
            half_k, half_k, 1).load(layout=DOT_A)
        b = b_buf.index(slot).slice(
            half_k, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    return a_desc, b_desc, acc


@gluon.jit
def bf16_8stage_split_b(
        b, DOT_B: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_N: gl.constexpr):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    b_grid = b.reshape((64, CTA_N, cta_n))
    b0 = gl.convert_layout(gl.amd.slice(
        b_grid, [64, CTA_N, half_n], [0, 0, 0]
    ).reshape((64, packed_n)), DOT_B, assert_trivial=True)
    b1 = gl.convert_layout(gl.amd.slice(
        b_grid, [64, CTA_N, half_n], [0, 0, half_n]
    ).reshape((64, packed_n)), DOT_B, assert_trivial=True)
    return b0, b1


@gluon.jit
def bf16_8stage_consume_and_refill(
        a_buf, b_buf, slot, refill_slot, a_desc, b_desc, acc0, acc1,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        DOT_B_LOAD: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, CTA_N: gl.constexpr):
    tdm.async_wait(0)
    with gl.amd.warp_pipeline_stage(
            "bf16_8s_stage0_load_k0", priority=0):
        a0 = a_buf.index(slot).slice(0, 64, 1).load(layout=DOT_A)
        b0_full = b_buf.index(slot).slice(
            0, 64, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        b0, b1 = bf16_8stage_split_b(b0_full, DOT_B, BLOCK_N, CTA_N)
    with gl.amd.warp_pipeline_stage(
            "bf16_8s_stage1_compute_k0_n0", priority=1):
        acc0 = gl.amd.gfx1250.wmma(a0, b0, acc0)
    with gl.amd.warp_pipeline_stage("bf16_8s_stage2_bubble"):
        pass
    with gl.amd.warp_pipeline_stage(
            "bf16_8s_stage3_compute_k0_n1", priority=1):
        acc1 = gl.amd.gfx1250.wmma(a0, b1, acc1)
    with gl.amd.warp_pipeline_stage(
            "bf16_8s_stage4_load_k1", priority=0):
        a1 = a_buf.index(slot).slice(64, 64, 1).load(layout=DOT_A)
        b1_full = b_buf.index(slot).slice(
            64, 64, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        b2, b3 = bf16_8stage_split_b(b1_full, DOT_B, BLOCK_N, CTA_N)
    with gl.amd.warp_pipeline_stage(
            "bf16_8s_stage5_compute_k1_n0", priority=1):
        acc0 = gl.amd.gfx1250.wmma(a1, b2, acc0)
    with gl.amd.warp_pipeline_stage(
            "bf16_8s_stage6_wait_refill", priority=0):
        gl.amd.gfx1250.cluster.arrive()
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc = bf16_issue_loads(
            a_desc, b_desc, a_buf.index(refill_slot),
            b_buf.index(refill_slot), 128)
    with gl.amd.warp_pipeline_stage(
            "bf16_8s_stage7_compute_k1_n1", priority=1):
        acc1 = gl.amd.gfx1250.wmma(a1, b3, acc1)
    return a_desc, b_desc, acc0, acc1


@gluon.jit
def bf16_8stage_consume_tail(
        a_buf, b_buf, slot, acc0, acc1,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        DOT_B_LOAD: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, CTA_N: gl.constexpr):
    with gl.amd.warp_pipeline_stage("bf16_8s_tail_load_k0", priority=0):
        a0 = a_buf.index(slot).slice(0, 64, 1).load(layout=DOT_A)
        b0_full = b_buf.index(slot).slice(
            0, 64, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        b0, b1 = bf16_8stage_split_b(b0_full, DOT_B, BLOCK_N, CTA_N)
    with gl.amd.warp_pipeline_stage("bf16_8s_tail_compute_k0", priority=1):
        acc0 = gl.amd.gfx1250.wmma(a0, b0, acc0)
        acc1 = gl.amd.gfx1250.wmma(a0, b1, acc1)
    with gl.amd.warp_pipeline_stage("bf16_8s_tail_load_k1", priority=0):
        a1 = a_buf.index(slot).slice(64, 64, 1).load(layout=DOT_A)
        b1_full = b_buf.index(slot).slice(
            64, 64, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        b2, b3 = bf16_8stage_split_b(b1_full, DOT_B, BLOCK_N, CTA_N)
    with gl.amd.warp_pipeline_stage("bf16_8s_tail_compute_k1", priority=1):
        acc0 = gl.amd.gfx1250.wmma(a1, b2, acc0)
        acc1 = gl.amd.gfx1250.wmma(a1, b3, acc1)
    return acc0, acc1


@gluon.jit
def bf16_bk128_8stage_cluster_4x4_gfx1250(
        a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk,
        stride_bn, stride_cm, stride_cn, GRID_MN: gl.constexpr,
        SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr,
        WMMA_LAYOUT: gl.constexpr, LOAD_WMMA_LAYOUT: gl.constexpr,
        OUTPUT_CGA_LAYOUT: gl.constexpr):
    block_m: gl.constexpr = 1024
    block_n: gl.constexpr = 1024
    block_k: gl.constexpr = 128
    cta_m: gl.constexpr = 4
    cta_n: gl.constexpr = 4
    gl.static_assert(
        a_ptr.type.element_ty.is_bf16() and b_ptr.type.element_ty.is_bf16())
    gl.static_assert(gl.num_ctas() == cta_m * cta_n)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
        M, N, block_m, block_n, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 8)
    dot_b_load: gl.constexpr = gl.DotOperandLayout(1, LOAD_WMMA_LAYOUT, 8)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * block_m * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(block_m, block_k),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * block_n * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(block_n, block_k),
        layout=SHARED_LAYOUT_B)
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, block_m, block_k], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, block_n, block_k], SHARED_LAYOUT_B)
    a_desc, b_desc = bf16_issue_loads(
        a_desc, b_desc, a_buf.index(0), b_buf.index(0), block_k)
    acc0 = gl.zeros(
        (block_m, block_n // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    acc1 = gl.zeros(
        (block_m, block_n // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, block_k)
    gl.assume(iter_max >= 2)
    for tile_idx in range(0, iter_max - 1):
        slot = tile_idx % 2
        refill_slot = (tile_idx + 1) % 2
        a_desc, b_desc, acc0, acc1 = bf16_8stage_consume_and_refill(
            a_buf, b_buf, slot, refill_slot, a_desc, b_desc, acc0, acc1,
            dot_a, dot_b, dot_b_load, block_m, block_n, cta_n)
    tdm.async_wait(0)
    last_slot = (iter_max - 1) % 2
    acc0, acc1 = bf16_8stage_consume_tail(
        a_buf, b_buf, last_slot, acc0, acc1, dot_a, dot_b, dot_b_load,
        block_m, block_n, cta_n)
    a_buf._keep_alive()
    b_buf._keep_alive()
    mxfp8_tdm_store_split_n2(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1,
        OUTPUT_CGA_LAYOUT, cta_m, cta_n)


@gluon.jit
def bf16_kernelc_2pf_bk128_fused_cluster_4x4_gfx1250(
        a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk,
        stride_bn, stride_cm, stride_cn, GRID_MN: gl.constexpr,
        SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr,
        WMMA_LAYOUT: gl.constexpr):
    block_m: gl.constexpr = 1024
    block_n: gl.constexpr = 1024
    block_k: gl.constexpr = 128
    gl.static_assert(
        a_ptr.type.element_ty.is_bf16() and b_ptr.type.element_ty.is_bf16())
    gl.static_assert(gl.num_ctas() == 16)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
        M, N, block_m, block_n, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 8)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * block_m * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(block_m, block_k),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * block_n * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(block_n, block_k),
        layout=SHARED_LAYOUT_B)
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, block_m, block_k], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, block_n, block_k], SHARED_LAYOUT_B)
    # Only slot 0 is prefetched; tile t's refill supplies tile t + 1.
    a_desc, b_desc = bf16_issue_loads(
        a_desc, b_desc, a_buf.index(0), b_buf.index(0), block_k)
    acc = gl.zeros((block_m, block_n), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, block_k)
    gl.assume(iter_max >= 2)
    gl.assume((iter_max % 2) == 0)
    # Tiles 0 .. iter_max - 2 refill; the unrolled body walks them in pairs and
    # the last refilling tile is peeled so the final tile can skip its refill.
    for _ in range(0, (iter_max - 2) // 2):
        a_desc, b_desc, acc = bf16_cluster_consume_and_refill(
            a_buf, b_buf, 0, 1, a_desc, b_desc, acc, dot_a, dot_b, block_k)
        a_desc, b_desc, acc = bf16_cluster_consume_and_refill(
            a_buf, b_buf, 1, 0, a_desc, b_desc, acc, dot_a, dot_b, block_k)
    a_desc, b_desc, acc = bf16_cluster_consume_and_refill(
        a_buf, b_buf, 0, 1, a_desc, b_desc, acc, dot_a, dot_b, block_k)
    tdm.async_wait(0)
    acc = bf16_consume_slot(
        a_buf, b_buf, 1, acc, dot_a, dot_b, block_k)
    snapshot_tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        WMMA_LAYOUT, block_m, block_n, 4)


def build_bf16_layouts():
    """Build matching distributed LDS and accumulator layouts for BF16.

    Layout construction is intentionally two-pass. First derive the local
    256x256-per-CTA WMMA ownership, then graft its operand CGA bases onto the
    padded shared layouts. This makes every CTA load the LDS partition consumed
    by its accumulator partition while retaining a cluster-wide 1024x1024 view.
    """
    cga_layout_c = make_cga_layout([4, 4], [4, 4], [0, 1])
    slice_m = BLOCK_M // 4
    slice_n = BLOCK_N // 4
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[BF16_BLOCK_K, 8]], [BLOCK_M, BF16_BLOCK_K], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[BF16_BLOCK_K, 8]], [BLOCK_N, BF16_BLOCK_K], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        BLOCK_M, BLOCK_N, local_a, local_b, NUM_WARPS, [16, 16, 32],
        a_transposed=False, b_transposed=True, slice_m=slice_m,
        slice_n=slice_n, transposed=True)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga_layout_c)
    dot_a = gl.DotOperandLayout(0, wmma, 8)
    dot_b = gl.DotOperandLayout(1, wmma, 8)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[BF16_BLOCK_K, 8]], [BLOCK_M, BF16_BLOCK_K], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[BF16_BLOCK_K, 8]], [BLOCK_N, BF16_BLOCK_K], [1, 0], cga_b)
    # One group per physical partition is the retained partitioned layout.
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        slice_m, slice_n, padded_a, padded_b, NUM_WARPS, [16, 16, 32],
        a_transposed=False, b_transposed=True, slice_m=slice_m,
        slice_n=slice_n, transposed=True)
    return shared_a, shared_b, wmma


def build_bf16_8stage_layouts():
    """Build retained operand layouts plus a 128-column WMMA partition."""
    cga_c = make_cga_layout([4, 4], [4, 4], [0, 1])
    output_cga = tuple(tuple(basis) + (0, 0) for basis in cga_c)
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[BF16_BLOCK_K, 8]], [BLOCK_M, BF16_BLOCK_K], [1, 0])
    half_b = gl.PaddedSharedLayout.with_identity_for(
        [[BF16_BLOCK_K, 8]], [BLOCK_N // 2, BF16_BLOCK_K], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        BLOCK_M, BLOCK_N // 2, local_a, half_b, NUM_WARPS, [16, 16, 32],
        a_transposed=False, b_transposed=True, slice_m=256,
        slice_n=128, transposed=True)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga_c)
    # The full-width B load adds the local-N=128 register basis to the
    # half-width compute layout. Slicing that basis away is register-only and
    # leaves exactly the compute DotOperandLayout.
    load_reg_bases = tuple(local_wmma.reg_bases) + ((0, 8),)
    load_wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        load_reg_bases, local_wmma.instr_shape, cga_c)
    dot_a = gl.DotOperandLayout(0, load_wmma, 8)
    dot_b = gl.DotOperandLayout(1, load_wmma, 8)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[BF16_BLOCK_K, 8]], [BLOCK_M, BF16_BLOCK_K], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[BF16_BLOCK_K, 8]], [BLOCK_N, BF16_BLOCK_K], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        256, 256, padded_a, padded_b, NUM_WARPS, [16, 16, 32],
        a_transposed=False, b_transposed=True, slice_m=256,
        slice_n=256, transposed=True)
    return shared_a, shared_b, wmma, load_wmma, output_cga


# ---------------------------------------------------------------------------
# MXFP8 x MXFP8.
#
# Aggregate tile: 1024x1024x256 across a 4x4 cluster.
# Encoding: E4M3 values with one E8M0 scale per 32 logical K elements.
# Pipeline: two BK256 slots, each consumed as two K128 scaled WMMAs. A/B data
# use separate copies over warps 0-3; A/B scales share a fused copy split
# across warps 0-1 and 2-3.
# Tile t refills the opposite slot drained by tile t-1, then reads tile t's
# slot. The refill and first local-load regions remain separate pipeline stages
# so the phase-shifted trailing group cannot still read the refill target.
# The final BK256 slot retains explicit named warp-pipeline stages while the
# TDM ring drains.
# The epilogue serializes two 128-column stores. Besides halving the output
# staging tile, its 128-column accumulator partition removes the old spill.
#
# Slot count. A third slot would restore the lead this schedule gives up, but
# it needs 430,456 bytes of LDS against a 327,680 limit, so two is forced.
MXFP8_NUM_SLOTS = gl.constexpr(2)
# Descriptors a tile's refill issues: A data, B data, and a fused scale pair.
MXFP8_TDM_PER_TILE = gl.constexpr(3)


@gluon.jit
def mxfp8_load_scale(
        scale_buffer, slot, start_k: gl.constexpr, LAYOUT: gl.constexpr,
        BLOCK_NONK: gl.constexpr, BK_SCALE: gl.constexpr,
        SUBTILE_SCALE_K: gl.constexpr):
    scale_slice = scale_buffer.index(slot).reshape(
        (BLOCK_NONK // 128, BK_SCALE // 4, 32, 4, 4)
    ).permute((0, 3, 2, 1, 4)).reshape((BLOCK_NONK, BK_SCALE))
    return scale_slice.slice(0, BLOCK_NONK, 0).slice(
        start_k, SUBTILE_SCALE_K, 1).load(layout=LAYOUT)


@gluon.jit
def mxfp8_issue_leading4_data(
        a_desc, b_desc, a_buf, b_buf, slot):
    # Both standalone copies use all four leading warps. Narrowing A to two
    # warps is legal, but the two-descriptor experiment that fused A with the
    # scales measured slower than this three-descriptor schedule.
    tdm.async_load(
        a_desc, dest=a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(
        b_desc, dest=b_buf.index(slot), warp_used_hint=0b00001111)
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 256])
    return a_desc, b_desc


@gluon.jit
def mxfp8_issue_leading4_scale(
        as_desc, bs_desc, as_buf, bs_buf, slot):
    tdm.async_load_fused([
        (as_desc, as_buf.index(slot), 0b00000011),
        (bs_desc, bs_buf.index(slot), 0b00001100),
    ])
    as_desc = tdm.update_tensor_descriptor(
        as_desc, add_offsets=[0, 1024])
    bs_desc = tdm.update_tensor_descriptor(
        bs_desc, add_offsets=[0, 1024])
    return as_desc, bs_desc


@gluon.jit
def mxfp8_consume_tile(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, DOT_A: gl.constexpr,
        DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr):
    a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
        0, 128, 1).load(layout=DOT_A)
    as0 = mxfp8_load_scale(
        as_buf, slot, 0, SCALE_A_LAYOUT, BLOCK_M, 8, 4)
    b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
        0, 128, 1).permute([1, 0]).load(layout=DOT_B)
    bs0 = mxfp8_load_scale(
        bs_buf, slot, 0, SCALE_B_LAYOUT, BLOCK_N, 8, 4)
    acc = gl.amd.gfx1250.wmma_scaled(
        a0, as0, "e4m3", b0, bs0, "e4m3", acc)
    a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
        128, 128, 1).load(layout=DOT_A)
    as1 = mxfp8_load_scale(
        as_buf, slot, 4, SCALE_A_LAYOUT, BLOCK_M, 8, 4)
    b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
        128, 128, 1).permute([1, 0]).load(layout=DOT_B)
    bs1 = mxfp8_load_scale(
        bs_buf, slot, 4, SCALE_B_LAYOUT, BLOCK_N, 8, 4)
    return gl.amd.gfx1250.wmma_scaled(
        a1, as1, "e4m3", b1, bs1, "e4m3", acc)


@gluon.jit
def mxfp8_consume_pipelined_tail(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, DOT_A: gl.constexpr,
        DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr):
    with gl.amd.warp_pipeline_stage("tail_load_low", priority=0):
        a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        as0 = mxfp8_load_scale(
            as_buf, slot, 0, SCALE_A_LAYOUT, BLOCK_M, 8, 4)
        b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs0 = mxfp8_load_scale(
            bs_buf, slot, 0, SCALE_B_LAYOUT, BLOCK_N, 8, 4)
    with gl.amd.warp_pipeline_stage("tail_compute_low", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0, bs0, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("tail_load_high", priority=0):
        a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        as1 = mxfp8_load_scale(
            as_buf, slot, 4, SCALE_A_LAYOUT, BLOCK_M, 8, 4)
        b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs1 = mxfp8_load_scale(
            bs_buf, slot, 4, SCALE_B_LAYOUT, BLOCK_N, 8, 4)
    with gl.amd.warp_pipeline_stage("tail_compute_high", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1, bs1, "e4m3", acc)
    return acc


@gluon.jit
def mxfp8_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, slot, refill_slot, a_desc, b_desc,
        as_desc, bs_desc, acc, DOT_A: gl.constexpr,
        DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, RING_WAIT: gl.constexpr):
    with gl.amd.warp_pipeline_stage("refill", priority=0):
        # ``refill_slot`` was drained by the previous tile, including the
        # trailing wave group, so this write cannot overlap a live LDS read.
        gl.amd.gfx1250.cluster.arrive()
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc = mxfp8_issue_leading4_data(
            a_desc, b_desc, a_buf, b_buf, refill_slot)
        as_desc, bs_desc = mxfp8_issue_leading4_scale(
            as_desc, bs_desc, as_buf, bs_buf, refill_slot)
    tdm.async_wait(RING_WAIT)
    with gl.amd.warp_pipeline_stage("load_low", priority=0):
        a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        as0 = mxfp8_load_scale(
            as_buf, slot, 0, SCALE_A_LAYOUT, BLOCK_M, 8, 4)
        b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs0 = mxfp8_load_scale(
            bs_buf, slot, 0, SCALE_B_LAYOUT, BLOCK_N, 8, 4)
    with gl.amd.warp_pipeline_stage("compute_low", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0, bs0, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("load_high", priority=0):
        a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        as1 = mxfp8_load_scale(
            as_buf, slot, 4, SCALE_A_LAYOUT, BLOCK_M, 8, 4)
        b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs1 = mxfp8_load_scale(
            bs_buf, slot, 4, SCALE_B_LAYOUT, BLOCK_N, 8, 4)
    with gl.amd.warp_pipeline_stage("compute_high", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1, bs1, "e4m3", acc)
    return a_desc, b_desc, as_desc, bs_desc, acc


@gluon.jit
def fp8_scaled_cluster_bf16_style_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
        OUTPUT_CGA_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, CTA_M: gl.constexpr, CTA_N: gl.constexpr):
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    gl.static_assert(BLOCK_M // CTA_M == 256)
    gl.static_assert(BLOCK_N // CTA_N == 256)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
        M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [BLOCK_M, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [BLOCK_N, 4])
    slots: gl.constexpr = MXFP8_NUM_SLOTS
    # The wait count is in descriptors, and a tile issues three: A data, B
    # data, and the fused scale pair. Because the wait sits after the issue,
    # the S - 1 tiles that are in flight but not yet read may stay outstanding.
    ring_wait: gl.constexpr = MXFP8_TDM_PER_TILE * (slots - 1)
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [slots, BLOCK_M, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [slots, BLOCK_N, 256], SHARED_LAYOUT_B)
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty,
        [slots, BLOCK_M // 128, 1024], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty,
        [slots, BLOCK_N // 128, 1024], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, 256),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, 256),
        layout=SHARED_LAYOUT_B)
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + (pid_m * BLOCK_M) // 128 * stride_scale,
        shape=(M // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(BLOCK_M // 128, 1024), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * BLOCK_N) // 128 * stride_scale,
        shape=(N // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(BLOCK_N // 128, 1024), layout=SHARED_SCALE_B)
    # Only tile 0 is prefetched. Tile t refills the slot drained by tile t-1
    # with tile t+1, so the refill never targets the slot being consumed.
    a_desc, b_desc = mxfp8_issue_leading4_data(
        a_desc, b_desc, a_buf, b_buf, 0)
    as_desc, bs_desc = mxfp8_issue_leading4_scale(
        as_desc, bs_desc, as_buf, bs_buf, 0)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 256)
    gl.assume(iter_max >= 2)
    for tile_idx in range(0, iter_max - 1):
        slot = tile_idx % 2
        a_desc, b_desc, as_desc, bs_desc, acc = mxfp8_consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, slot, (tile_idx + 1) % 2,
            a_desc, b_desc, as_desc, bs_desc, acc, dot_a, dot_b,
            scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N, ring_wait)

    # The final tile has no successor refill.
    tdm.async_wait(0)
    snapshot_cluster_wait()
    last_slot = (iter_max - 1) % 2
    acc = mxfp8_consume_pipelined_tail(
        a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N)
    # The serial output stage aliases cluster-distributed operand LDS. All CTAs
    # must finish the final operand reads before any CTA repurposes that arena.
    snapshot_cluster_wait()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    mxfp8_tdm_store_serial_n2(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        OUTPUT_CGA_LAYOUT, CTA_M, CTA_N)


@gluon.jit
def mxfp8_bk256_opt_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, slot, refill_slot, a_desc, b_desc,
        as_desc, bs_desc, acc0, acc1, DOT_A: gl.constexpr,
        DOT_B: gl.constexpr, DOT_B_LOAD: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        SCALE_B_LOAD: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, CTA_N: gl.constexpr):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    tdm.async_wait(3)
    with gl.amd.warp_pipeline_stage(
            "stage0_load_k0", priority=0, phase_gap=2):
        a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        as0 = mxfp8_load_scale(
            as_buf, slot, 0, SCALE_A_LAYOUT, BLOCK_M, 8, 4)
        b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        bs0 = mxfp8_load_scale(
            bs_buf, slot, 0, SCALE_B_LOAD, BLOCK_N, 8, 4)
        b0_grid = b0.reshape((128, CTA_N, cta_n))
        b0_lo = gl.convert_layout(gl.amd.slice(
            b0_grid, [128, CTA_N, half_n], [0, 0, 0]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        b0_hi = gl.convert_layout(gl.amd.slice(
            b0_grid, [128, CTA_N, half_n], [0, 0, half_n]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        bs0_grid = bs0.reshape((CTA_N, cta_n, 4))
        bs0_lo = gl.convert_layout(gl.amd.slice(
            bs0_grid, [CTA_N, half_n, 4], [0, 0, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        bs0_hi = gl.convert_layout(gl.amd.slice(
            bs0_grid, [CTA_N, half_n, 4], [0, half_n, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
    with gl.amd.warp_pipeline_stage("stage1_bubble"):
        pass
    with gl.amd.warp_pipeline_stage("stage2_compute_k0_n0", priority=1):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0_lo, bs0_lo, "e4m3", acc0)
    with gl.amd.warp_pipeline_stage("stage3_compute_k0_n1", priority=1):
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0_hi, bs0_hi, "e4m3", acc1)
    with gl.amd.warp_pipeline_stage("stage4_load_k1", priority=0):
        a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        as1 = mxfp8_load_scale(
            as_buf, slot, 4, SCALE_A_LAYOUT, BLOCK_M, 8, 4)
        b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        bs1 = mxfp8_load_scale(
            bs_buf, slot, 4, SCALE_B_LOAD, BLOCK_N, 8, 4)
        b1_grid = b1.reshape((128, CTA_N, cta_n))
        b1_lo = gl.convert_layout(gl.amd.slice(
            b1_grid, [128, CTA_N, half_n], [0, 0, 0]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        b1_hi = gl.convert_layout(gl.amd.slice(
            b1_grid, [128, CTA_N, half_n], [0, 0, half_n]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        bs1_grid = bs1.reshape((CTA_N, cta_n, 4))
        bs1_lo = gl.convert_layout(gl.amd.slice(
            bs1_grid, [CTA_N, half_n, 4], [0, 0, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        bs1_hi = gl.convert_layout(gl.amd.slice(
            bs1_grid, [CTA_N, half_n, 4], [0, half_n, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
    with gl.amd.warp_pipeline_stage("stage5_bubble"):
        pass
    with gl.amd.warp_pipeline_stage("stage6_compute_k1_n0", priority=1):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1_lo, bs1_lo, "e4m3", acc0)
    with gl.amd.warp_pipeline_stage(
            "stage7_arrive_compute_k1_n1_wait_refill", priority=1):
        gl.amd.gfx1250.cluster.arrive()
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1_hi, bs1_hi, "e4m3", acc1)
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc = mxfp8_issue_leading4_data(
            a_desc, b_desc, a_buf, b_buf, refill_slot)
        as_desc, bs_desc = mxfp8_issue_leading4_scale(
            as_desc, bs_desc, as_buf, bs_buf, refill_slot)
    return a_desc, b_desc, as_desc, bs_desc, acc0, acc1


@gluon.jit
def mxfp8_bk256_opt_consume_tail_split_n(
        a_buf, b_buf, as_buf, bs_buf, slot, acc0, acc1,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        DOT_B_LOAD: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr, SCALE_B_LOAD: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_N: gl.constexpr):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    with gl.amd.warp_pipeline_stage("gap2_tail_load_k0", priority=0):
        a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        as0 = mxfp8_load_scale(
            as_buf, slot, 0, SCALE_A_LAYOUT, BLOCK_M, 8, 4)
        b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        bs0 = mxfp8_load_scale(
            bs_buf, slot, 0, SCALE_B_LOAD, BLOCK_N, 8, 4)
        b0_grid = b0.reshape((128, CTA_N, cta_n))
        b0_lo = gl.convert_layout(gl.amd.slice(
            b0_grid, [128, CTA_N, half_n], [0, 0, 0]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        b0_hi = gl.convert_layout(gl.amd.slice(
            b0_grid, [128, CTA_N, half_n], [0, 0, half_n]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        bs0_grid = bs0.reshape((CTA_N, cta_n, 4))
        bs0_lo = gl.convert_layout(gl.amd.slice(
            bs0_grid, [CTA_N, half_n, 4], [0, 0, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        bs0_hi = gl.convert_layout(gl.amd.slice(
            bs0_grid, [CTA_N, half_n, 4], [0, half_n, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
    with gl.amd.warp_pipeline_stage("gap2_tail_compute_k0", priority=1):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0_lo, bs0_lo, "e4m3", acc0)
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0_hi, bs0_hi, "e4m3", acc1)
    with gl.amd.warp_pipeline_stage("gap2_tail_load_k1", priority=0):
        a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        as1 = mxfp8_load_scale(
            as_buf, slot, 4, SCALE_A_LAYOUT, BLOCK_M, 8, 4)
        b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B_LOAD)
        bs1 = mxfp8_load_scale(
            bs_buf, slot, 4, SCALE_B_LOAD, BLOCK_N, 8, 4)
        b1_grid = b1.reshape((128, CTA_N, cta_n))
        b1_lo = gl.convert_layout(gl.amd.slice(
            b1_grid, [128, CTA_N, half_n], [0, 0, 0]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        b1_hi = gl.convert_layout(gl.amd.slice(
            b1_grid, [128, CTA_N, half_n], [0, 0, half_n]
        ).reshape((128, packed_n)), DOT_B, assert_trivial=True)
        bs1_grid = bs1.reshape((CTA_N, cta_n, 4))
        bs1_lo = gl.convert_layout(gl.amd.slice(
            bs1_grid, [CTA_N, half_n, 4], [0, 0, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
        bs1_hi = gl.convert_layout(gl.amd.slice(
            bs1_grid, [CTA_N, half_n, 4], [0, half_n, 0]
        ).reshape((packed_n, 4)), SCALE_B_LAYOUT, assert_trivial=True)
    with gl.amd.warp_pipeline_stage("gap2_tail_compute_k1", priority=1):
        acc0 = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1_lo, bs1_lo, "e4m3", acc0)
        acc1 = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1_hi, bs1_hi, "e4m3", acc1)
    return acc0, acc1


@gluon.jit
def fp8_scaled_cluster_bk256_opt_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
        LOAD_WMMA_LAYOUT: gl.constexpr, OUTPUT_CGA_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_M: gl.constexpr, CTA_N: gl.constexpr):
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    gl.static_assert(BLOCK_M // CTA_M == 256)
    gl.static_assert(BLOCK_N // CTA_N == 256)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
        M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    dot_b_load: gl.constexpr = gl.DotOperandLayout(
        1, LOAD_WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [BLOCK_M, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [BLOCK_N // 2, 4])
    scale_b_load: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b_load, [BLOCK_N, 4])
    slots: gl.constexpr = MXFP8_NUM_SLOTS
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [slots, BLOCK_M, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [slots, BLOCK_N, 256], SHARED_LAYOUT_B)
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty,
        [slots, BLOCK_M // 128, 1024], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty,
        [slots, BLOCK_N // 128, 1024], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, 256),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, 256),
        layout=SHARED_LAYOUT_B)
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + (pid_m * BLOCK_M) // 128 * stride_scale,
        shape=(M // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(BLOCK_M // 128, 1024), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * BLOCK_N) // 128 * stride_scale,
        shape=(N // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(BLOCK_N // 128, 1024), layout=SHARED_SCALE_B)
    for prefetch_idx in gl.static_range(2):
        a_desc, b_desc = mxfp8_issue_leading4_data(
            a_desc, b_desc, a_buf, b_buf, prefetch_idx)
        as_desc, bs_desc = mxfp8_issue_leading4_scale(
            as_desc, bs_desc, as_buf, bs_buf, prefetch_idx)
    acc0 = gl.zeros(
        (BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    acc1 = gl.zeros(
        (BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 256)
    gl.assume(iter_max >= 3)
    for tile_idx in range(0, iter_max - 2):
        slot = tile_idx % 2
        a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
            mxfp8_bk256_opt_consume_and_refill(
                a_buf, b_buf, as_buf, bs_buf, slot, slot,
                a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
                dot_b_load, scale_a_layout, scale_b_layout, scale_b_load,
                BLOCK_M, BLOCK_N, CTA_N))

    tdm.async_wait(3)
    penultimate_slot = (iter_max - 2) % 2
    acc0, acc1 = mxfp8_bk256_opt_consume_tail_split_n(
        a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc0, acc1,
        dot_a, dot_b, dot_b_load, scale_a_layout, scale_b_layout,
        scale_b_load, BLOCK_M, BLOCK_N, CTA_N)
    tdm.async_wait(0)
    last_slot = (iter_max - 1) % 2
    acc0, acc1 = mxfp8_bk256_opt_consume_tail_split_n(
        a_buf, b_buf, as_buf, bs_buf, last_slot, acc0, acc1, dot_a, dot_b,
        dot_b_load, scale_a_layout, scale_b_layout, scale_b_load,
        BLOCK_M, BLOCK_N, CTA_N)
    snapshot_cluster_wait()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    mxfp8_tdm_store_split_n2(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc0, acc1,
        OUTPUT_CGA_LAYOUT, CTA_M, CTA_N)


@gluon.jit
def mxfp8_bk128_issue_refill(
        a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
        slot):
    tdm.async_load(
        a_desc, dest=a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(
        b_desc, dest=b_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load_fused([
        (as_desc, as_buf.index(slot), 0b00000011),
        (bs_desc, bs_buf.index(slot), 0b00001100),
    ])
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 128])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 128])
    as_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, 512])
    bs_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, 512])
    return a_desc, b_desc, as_desc, bs_desc


@gluon.jit
def mxfp8_bk128_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, slot, refill_slot,
        a_desc, b_desc, as_desc, bs_desc, acc,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        REFILL_SCHEDULE: gl.constexpr):
    # Two prefetched tiles are outstanding. Retire the older one while leaving
    # the next tile in flight, then refill the third, disjoint slot.
    tdm.async_wait(3)
    if REFILL_SCHEDULE == 0:
        with gl.amd.warp_pipeline_stage(
                "bk128_refill_then_load", priority=0):
            gl.amd.gfx1250.cluster.arrive()
            gl.amd.gfx1250.cluster.wait()
            a_desc, b_desc, as_desc, bs_desc = mxfp8_bk128_issue_refill(
                a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                bs_buf, refill_slot)
            a = a_buf.index(slot).load(layout=DOT_A)
            as_ = mxfp8_load_scale(
                as_buf, slot, 0, SCALE_A_LAYOUT, BLOCK_M, 4, 4)
            b = b_buf.index(slot).permute([1, 0]).load(layout=DOT_B)
            bs = mxfp8_load_scale(
                bs_buf, slot, 0, SCALE_B_LAYOUT, BLOCK_N, 4, 4)
    elif REFILL_SCHEDULE == 1:
        with gl.amd.warp_pipeline_stage("bk128_load", priority=0):
            a = a_buf.index(slot).load(layout=DOT_A)
            as_ = mxfp8_load_scale(
                as_buf, slot, 0, SCALE_A_LAYOUT, BLOCK_M, 4, 4)
            b = b_buf.index(slot).permute([1, 0]).load(layout=DOT_B)
            bs = mxfp8_load_scale(
                bs_buf, slot, 0, SCALE_B_LAYOUT, BLOCK_N, 4, 4)
    else:
        with gl.amd.warp_pipeline_stage(
                "bk128_arrive_load_wait_refill", priority=0):
            gl.amd.gfx1250.cluster.arrive()
            a = a_buf.index(slot).load(layout=DOT_A)
            as_ = mxfp8_load_scale(
                as_buf, slot, 0, SCALE_A_LAYOUT, BLOCK_M, 4, 4)
            b = b_buf.index(slot).permute([1, 0]).load(layout=DOT_B)
            bs = mxfp8_load_scale(
                bs_buf, slot, 0, SCALE_B_LAYOUT, BLOCK_N, 4, 4)
            gl.amd.gfx1250.cluster.wait()
            a_desc, b_desc, as_desc, bs_desc = mxfp8_bk128_issue_refill(
                a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                bs_buf, refill_slot)
    with gl.amd.warp_pipeline_stage("bk128_compute", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a, as_, "e4m3", b, bs, "e4m3", acc)
    if REFILL_SCHEDULE == 1:
        with gl.amd.warp_pipeline_stage("bk128_refill_after", priority=0):
            gl.amd.gfx1250.cluster.arrive()
            gl.amd.gfx1250.cluster.wait()
            a_desc, b_desc, as_desc, bs_desc = mxfp8_bk128_issue_refill(
                a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                bs_buf, refill_slot)
    return a_desc, b_desc, as_desc, bs_desc, acc


@gluon.jit
def mxfp8_bk128_consume_tail(
        a_buf, b_buf, as_buf, bs_buf, slot, acc,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    with gl.amd.warp_pipeline_stage("bk128_tail_load", priority=0):
        a = a_buf.index(slot).load(layout=DOT_A)
        as_ = mxfp8_load_scale(
            as_buf, slot, 0, SCALE_A_LAYOUT, BLOCK_M, 4, 4)
        b = b_buf.index(slot).permute([1, 0]).load(layout=DOT_B)
        bs = mxfp8_load_scale(
            bs_buf, slot, 0, SCALE_B_LAYOUT, BLOCK_N, 4, 4)
    with gl.amd.warp_pipeline_stage("bk128_tail_compute", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a, as_, "e4m3", b, bs, "e4m3", acc)
    return acc


@gluon.jit
def mxfp8_bk128_single_cta_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, slot, refill_slot,
        a_desc, b_desc, as_desc, bs_desc, acc,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    tdm.async_wait(3)
    with gl.amd.warp_pipeline_stage(
            "skinny_bk128_refill_load", priority=0):
        a_desc, b_desc, as_desc, bs_desc = mxfp8_bk128_issue_refill(
            a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
            refill_slot)
        a = a_buf.index(slot).load(layout=DOT_A)
        as_ = mxfp8_load_scale(
            as_buf, slot, 0, SCALE_A_LAYOUT, BLOCK_M, 4, 4)
        b = b_buf.index(slot).permute([1, 0]).load(layout=DOT_B)
        bs = mxfp8_load_scale(
            bs_buf, slot, 0, SCALE_B_LAYOUT, BLOCK_N, 4, 4)
    with gl.amd.warp_pipeline_stage(
            "skinny_bk128_compute", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a, as_, "e4m3", b, bs, "e4m3", acc)
    return a_desc, b_desc, as_desc, bs_desc, acc


@gluon.jit
def fp8_scaled_single_cta_512x128x128_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr):
    block_m: gl.constexpr = 512
    block_n: gl.constexpr = 128
    gl.static_assert(gl.num_ctas() == 1)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
        M, N, block_m, block_n, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [block_m, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [block_n, 4])
    slots: gl.constexpr = 3
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [slots, block_m, 128], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [slots, block_n, 128], SHARED_LAYOUT_B)
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty, [slots, 4, 512], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty, [slots, 1, 512], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * block_m * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(block_m, 128),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * block_n * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(block_n, 128),
        layout=SHARED_LAYOUT_B)
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + (pid_m * block_m) // 128 * stride_scale,
        shape=(M // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(4, 512), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * block_n) // 128 * stride_scale,
        shape=(N // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(1, 512), layout=SHARED_SCALE_B)
    for prefetch_idx in gl.static_range(2):
        a_desc, b_desc, as_desc, bs_desc = mxfp8_bk128_issue_refill(
            a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
            prefetch_idx)
    acc = gl.zeros(
        (block_m, block_n), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 128)
    gl.assume(iter_max >= 3)
    for tile_idx in range(0, iter_max - 2):
        slot = tile_idx % 3
        refill_slot = (tile_idx + 2) % 3
        a_desc, b_desc, as_desc, bs_desc, acc = (
            mxfp8_bk128_single_cta_consume_and_refill(
                a_buf, b_buf, as_buf, bs_buf, slot, refill_slot,
                a_desc, b_desc, as_desc, bs_desc, acc, dot_a, dot_b,
                scale_a_layout, scale_b_layout, block_m, block_n))

    tdm.async_wait(3)
    penultimate_slot = (iter_max - 2) % 3
    acc = mxfp8_bk128_consume_tail(
        a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout, block_m, block_n)
    tdm.async_wait(0)
    last_slot = (iter_max - 1) % 3
    acc = mxfp8_bk128_consume_tail(
        a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout, block_m, block_n)
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    snapshot_tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        WMMA_LAYOUT, block_m, block_n, 1)


@gluon.jit
def mxfp8_64_bk512_issue_refill(
        a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
        slot):
    tdm.async_load(
        a_desc, dest=a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(
        b_desc, dest=b_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load_fused([
        (as_desc, as_buf.index(slot), 0b00000011),
        (bs_desc, bs_buf.index(slot), 0b00001100),
    ])
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 512])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 512])
    as_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, 16])
    bs_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, 16])
    return a_desc, b_desc, as_desc, bs_desc


@gluon.jit
def mxfp8_64_load_scale(
        scale_buf, slot, start_k: gl.constexpr, LAYOUT: gl.constexpr):
    return scale_buf.index(slot).slice(
        start_k, 4, 1).load(layout=LAYOUT)


@gluon.jit
def mxfp8_64_bk512_consume_tail(
        a_buf, b_buf, as_buf, bs_buf, slot, acc,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr):
    with gl.amd.warp_pipeline_stage("tile64_tail_load_k0", priority=0):
        a0 = a_buf.index(slot).slice(0, 128, 1).load(layout=DOT_A)
        b0 = b_buf.index(slot).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B)
        as0 = mxfp8_64_load_scale(
            as_buf, slot, 0, SCALE_A_LAYOUT)
        bs0 = mxfp8_64_load_scale(
            bs_buf, slot, 0, SCALE_B_LAYOUT)
    with gl.amd.warp_pipeline_stage("tile64_tail_compute_k0", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0, bs0, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("tile64_tail_load_k1", priority=0):
        a1 = a_buf.index(slot).slice(128, 128, 1).load(layout=DOT_A)
        b1 = b_buf.index(slot).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B)
        as1 = mxfp8_64_load_scale(
            as_buf, slot, 4, SCALE_A_LAYOUT)
        bs1 = mxfp8_64_load_scale(
            bs_buf, slot, 4, SCALE_B_LAYOUT)
    with gl.amd.warp_pipeline_stage("tile64_tail_compute_k1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1, bs1, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("tile64_tail_load_k2", priority=0):
        a2 = a_buf.index(slot).slice(256, 128, 1).load(layout=DOT_A)
        b2 = b_buf.index(slot).slice(
            256, 128, 1).permute([1, 0]).load(layout=DOT_B)
        as2 = mxfp8_64_load_scale(
            as_buf, slot, 8, SCALE_A_LAYOUT)
        bs2 = mxfp8_64_load_scale(
            bs_buf, slot, 8, SCALE_B_LAYOUT)
    with gl.amd.warp_pipeline_stage("tile64_tail_compute_k2", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a2, as2, "e4m3", b2, bs2, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("tile64_tail_load_k3", priority=0):
        a3 = a_buf.index(slot).slice(384, 128, 1).load(layout=DOT_A)
        b3 = b_buf.index(slot).slice(
            384, 128, 1).permute([1, 0]).load(layout=DOT_B)
        as3 = mxfp8_64_load_scale(
            as_buf, slot, 12, SCALE_A_LAYOUT)
        bs3 = mxfp8_64_load_scale(
            bs_buf, slot, 12, SCALE_B_LAYOUT)
    with gl.amd.warp_pipeline_stage("tile64_tail_compute_k3", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a3, as3, "e4m3", b3, bs3, "e4m3", acc)
    return acc


@gluon.jit
def fp8_scaled_cluster_cta_bk512_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_M: gl.constexpr, CTA_N: gl.constexpr,
        CTA_TILE_M: gl.constexpr, CTA_TILE_N: gl.constexpr,
        CLUSTERED: gl.constexpr):
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    gl.static_assert(BLOCK_M // CTA_M == CTA_TILE_M)
    gl.static_assert(BLOCK_N // CTA_N == CTA_TILE_N)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
        M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [BLOCK_M, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [BLOCK_N, 4])
    slots: gl.constexpr = 2
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [slots, BLOCK_M, 512], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [slots, BLOCK_N, 512], SHARED_LAYOUT_B)
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty, [slots, BLOCK_M, 16], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty, [slots, BLOCK_N, 16], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, 512),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, 512),
        layout=SHARED_LAYOUT_B)
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + pid_m * BLOCK_M * stride_scale,
        shape=(M, K // 32), strides=(stride_scale, 1),
        block_shape=(BLOCK_M, 16), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + pid_n * BLOCK_N * stride_scale,
        shape=(N, K // 32), strides=(stride_scale, 1),
        block_shape=(BLOCK_N, 16), layout=SHARED_SCALE_B)
    for prefetch_idx in gl.static_range(2):
        a_desc, b_desc, as_desc, bs_desc = mxfp8_64_bk512_issue_refill(
            a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
            prefetch_idx)
    acc = gl.zeros(
        (BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 512)
    gl.assume(iter_max >= 3)
    for tile_idx in range(0, iter_max - 2):
        slot = tile_idx % 2
        tdm.async_wait(3)
        with gl.amd.warp_pipeline_stage(
                "tile64_load_k0", priority=0, phase_gap=2):
            a0 = a_buf.index(slot).slice(0, 128, 1).load(layout=dot_a)
            b0 = b_buf.index(slot).slice(
                0, 128, 1).permute([1, 0]).load(layout=dot_b)
            as0 = mxfp8_64_load_scale(
                as_buf, slot, 0, scale_a_layout)
            bs0 = mxfp8_64_load_scale(
                bs_buf, slot, 0, scale_b_layout)
        with gl.amd.warp_pipeline_stage("tile64_compute_k0", priority=1):
            acc = gl.amd.gfx1250.wmma_scaled(
                a0, as0, "e4m3", b0, bs0, "e4m3", acc)
        with gl.amd.warp_pipeline_stage("tile64_load_k1", priority=0):
            a1 = a_buf.index(slot).slice(128, 128, 1).load(layout=dot_a)
            b1 = b_buf.index(slot).slice(
                128, 128, 1).permute([1, 0]).load(layout=dot_b)
            as1 = mxfp8_64_load_scale(
                as_buf, slot, 4, scale_a_layout)
            bs1 = mxfp8_64_load_scale(
                bs_buf, slot, 4, scale_b_layout)
        with gl.amd.warp_pipeline_stage("tile64_compute_k1", priority=1):
            acc = gl.amd.gfx1250.wmma_scaled(
                a1, as1, "e4m3", b1, bs1, "e4m3", acc)
        with gl.amd.warp_pipeline_stage("tile64_load_k23", priority=0):
            a2 = a_buf.index(slot).slice(256, 128, 1).load(layout=dot_a)
            b2 = b_buf.index(slot).slice(
                256, 128, 1).permute([1, 0]).load(layout=dot_b)
            as2 = mxfp8_64_load_scale(
                as_buf, slot, 8, scale_a_layout)
            bs2 = mxfp8_64_load_scale(
                bs_buf, slot, 8, scale_b_layout)
            a3 = a_buf.index(slot).slice(384, 128, 1).load(layout=dot_a)
            b3 = b_buf.index(slot).slice(
                384, 128, 1).permute([1, 0]).load(layout=dot_b)
            as3 = mxfp8_64_load_scale(
                as_buf, slot, 12, scale_a_layout)
            bs3 = mxfp8_64_load_scale(
                bs_buf, slot, 12, scale_b_layout)
        with gl.amd.warp_pipeline_stage("tile64_bubble"):
            pass
        with gl.amd.warp_pipeline_stage("tile64_compute_k2", priority=1):
            acc = gl.amd.gfx1250.wmma_scaled(
                a2, as2, "e4m3", b2, bs2, "e4m3", acc)
        with gl.amd.warp_pipeline_stage(
                "tile64_compute_k3_refill", priority=1):
            acc = gl.amd.gfx1250.wmma_scaled(
                a3, as3, "e4m3", b3, bs3, "e4m3", acc)
            if CLUSTERED:
                gl.amd.gfx1250.cluster.arrive()
                gl.amd.gfx1250.cluster.wait()
            a_desc, b_desc, as_desc, bs_desc = mxfp8_64_bk512_issue_refill(
                a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                bs_buf, slot)

    tdm.async_wait(3)
    penultimate_slot = (iter_max - 2) % 2
    acc = mxfp8_64_bk512_consume_tail(
        a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout)
    tdm.async_wait(0)
    last_slot = (iter_max - 1) % 2
    acc = mxfp8_64_bk512_consume_tail(
        a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout)
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    snapshot_tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        WMMA_LAYOUT, BLOCK_M, BLOCK_N, CTA_N)


@gluon.jit
def mxfp8_64_bk1024_issue_refill(
        a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
        slot):
    tdm.async_load(
        a_desc, dest=a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(
        b_desc, dest=b_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load_fused([
        (as_desc, as_buf.index(slot), 0b00000011),
        (bs_desc, bs_buf.index(slot), 0b00001100),
    ])
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 1024])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 1024])
    as_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, 32])
    bs_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, 32])
    return a_desc, b_desc, as_desc, bs_desc


@gluon.jit
def mxfp8_64_bk1024_consume(
        a_buf, b_buf, as_buf, bs_buf, slot, refill_slot,
        a_desc, b_desc, as_desc, bs_desc, acc,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        DO_REFILL: gl.constexpr, CLUSTERED: gl.constexpr):
    tdm.async_wait(0)
    with gl.amd.warp_pipeline_stage("tile64_bk1024_load_k0_3", priority=0):
        a0 = a_buf.index(slot).slice(0, 128, 1).load(layout=DOT_A)
        b0 = b_buf.index(slot).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B)
        as0 = mxfp8_64_load_scale(as_buf, slot, 0, SCALE_A_LAYOUT)
        bs0 = mxfp8_64_load_scale(bs_buf, slot, 0, SCALE_B_LAYOUT)
        a1 = a_buf.index(slot).slice(128, 128, 1).load(layout=DOT_A)
        b1 = b_buf.index(slot).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B)
        as1 = mxfp8_64_load_scale(as_buf, slot, 4, SCALE_A_LAYOUT)
        bs1 = mxfp8_64_load_scale(bs_buf, slot, 4, SCALE_B_LAYOUT)
        a2 = a_buf.index(slot).slice(256, 128, 1).load(layout=DOT_A)
        b2 = b_buf.index(slot).slice(
            256, 128, 1).permute([1, 0]).load(layout=DOT_B)
        as2 = mxfp8_64_load_scale(as_buf, slot, 8, SCALE_A_LAYOUT)
        bs2 = mxfp8_64_load_scale(bs_buf, slot, 8, SCALE_B_LAYOUT)
        a3 = a_buf.index(slot).slice(384, 128, 1).load(layout=DOT_A)
        b3 = b_buf.index(slot).slice(
            384, 128, 1).permute([1, 0]).load(layout=DOT_B)
        as3 = mxfp8_64_load_scale(as_buf, slot, 12, SCALE_A_LAYOUT)
        bs3 = mxfp8_64_load_scale(bs_buf, slot, 12, SCALE_B_LAYOUT)
    with gl.amd.warp_pipeline_stage("tile64_bk1024_compute_k0_3", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0, bs0, "e4m3", acc)
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1, bs1, "e4m3", acc)
        acc = gl.amd.gfx1250.wmma_scaled(
            a2, as2, "e4m3", b2, bs2, "e4m3", acc)
        acc = gl.amd.gfx1250.wmma_scaled(
            a3, as3, "e4m3", b3, bs3, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("tile64_bk1024_load_k4_7", priority=0):
        a4 = a_buf.index(slot).slice(512, 128, 1).load(layout=DOT_A)
        b4 = b_buf.index(slot).slice(
            512, 128, 1).permute([1, 0]).load(layout=DOT_B)
        as4 = mxfp8_64_load_scale(as_buf, slot, 16, SCALE_A_LAYOUT)
        bs4 = mxfp8_64_load_scale(bs_buf, slot, 16, SCALE_B_LAYOUT)
        a5 = a_buf.index(slot).slice(640, 128, 1).load(layout=DOT_A)
        b5 = b_buf.index(slot).slice(
            640, 128, 1).permute([1, 0]).load(layout=DOT_B)
        as5 = mxfp8_64_load_scale(as_buf, slot, 20, SCALE_A_LAYOUT)
        bs5 = mxfp8_64_load_scale(bs_buf, slot, 20, SCALE_B_LAYOUT)
        a6 = a_buf.index(slot).slice(768, 128, 1).load(layout=DOT_A)
        b6 = b_buf.index(slot).slice(
            768, 128, 1).permute([1, 0]).load(layout=DOT_B)
        as6 = mxfp8_64_load_scale(as_buf, slot, 24, SCALE_A_LAYOUT)
        bs6 = mxfp8_64_load_scale(bs_buf, slot, 24, SCALE_B_LAYOUT)
        a7 = a_buf.index(slot).slice(896, 128, 1).load(layout=DOT_A)
        b7 = b_buf.index(slot).slice(
            896, 128, 1).permute([1, 0]).load(layout=DOT_B)
        as7 = mxfp8_64_load_scale(as_buf, slot, 28, SCALE_A_LAYOUT)
        bs7 = mxfp8_64_load_scale(bs_buf, slot, 28, SCALE_B_LAYOUT)
    with gl.amd.warp_pipeline_stage(
            "tile64_bk1024_compute_k4_7_refill", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a4, as4, "e4m3", b4, bs4, "e4m3", acc)
        acc = gl.amd.gfx1250.wmma_scaled(
            a5, as5, "e4m3", b5, bs5, "e4m3", acc)
        acc = gl.amd.gfx1250.wmma_scaled(
            a6, as6, "e4m3", b6, bs6, "e4m3", acc)
        acc = gl.amd.gfx1250.wmma_scaled(
            a7, as7, "e4m3", b7, bs7, "e4m3", acc)
        if DO_REFILL:
            if CLUSTERED:
                gl.amd.gfx1250.cluster.arrive()
                gl.amd.gfx1250.cluster.wait()
            a_desc, b_desc, as_desc, bs_desc = mxfp8_64_bk1024_issue_refill(
                a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                bs_buf, refill_slot)
    return a_desc, b_desc, as_desc, bs_desc, acc


@gluon.jit
def fp8_scaled_cluster_64x64x1024_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_M: gl.constexpr, CTA_N: gl.constexpr):
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    gl.static_assert(BLOCK_M // CTA_M == 64)
    gl.static_assert(BLOCK_N // CTA_N == 64)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
        M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [BLOCK_M, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [BLOCK_N, 4])
    slots: gl.constexpr = 2
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [slots, BLOCK_M, 1024], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [slots, BLOCK_N, 1024], SHARED_LAYOUT_B)
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty, [slots, BLOCK_M, 32], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty, [slots, BLOCK_N, 32], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, 1024),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, 1024),
        layout=SHARED_LAYOUT_B)
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + pid_m * BLOCK_M * stride_scale,
        shape=(M, K // 32), strides=(stride_scale, 1),
        block_shape=(BLOCK_M, 32), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + pid_n * BLOCK_N * stride_scale,
        shape=(N, K // 32), strides=(stride_scale, 1),
        block_shape=(BLOCK_N, 32), layout=SHARED_SCALE_B)
    a_desc, b_desc, as_desc, bs_desc = mxfp8_64_bk1024_issue_refill(
        a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, 0)
    acc = gl.zeros(
        (BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 1024)
    gl.assume(iter_max >= 2)
    for tile_idx in range(0, iter_max - 1):
        slot = tile_idx % 2
        refill_slot = (tile_idx + 1) % 2
        a_desc, b_desc, as_desc, bs_desc, acc = mxfp8_64_bk1024_consume(
            a_buf, b_buf, as_buf, bs_buf, slot, refill_slot,
            a_desc, b_desc, as_desc, bs_desc, acc, dot_a, dot_b,
            scale_a_layout, scale_b_layout, True, CTA_M * CTA_N > 1)
    last_slot = (iter_max - 1) % 2
    a_desc, b_desc, as_desc, bs_desc, acc = mxfp8_64_bk1024_consume(
        a_buf, b_buf, as_buf, bs_buf, last_slot, 0,
        a_desc, b_desc, as_desc, bs_desc, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout, False, CTA_M * CTA_N > 1)
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    snapshot_tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        WMMA_LAYOUT, BLOCK_M, BLOCK_N, CTA_N)


@gluon.jit
def mxfp8_raw_bk256_issue_refill(
        a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
        slot):
    tdm.async_load(
        a_desc, dest=a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(
        b_desc, dest=b_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load_fused([
        (as_desc, as_buf.index(slot), 0b00000011),
        (bs_desc, bs_buf.index(slot), 0b00001100),
    ])
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 256])
    as_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, 8])
    bs_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, 8])
    return a_desc, b_desc, as_desc, bs_desc


@gluon.jit
def mxfp8_raw_bk256_consume_refill(
        a_buf, b_buf, as_buf, bs_buf, slot, refill_slot,
        a_desc, b_desc, as_desc, bs_desc, acc,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr):
    tdm.async_wait(3)
    with gl.amd.warp_pipeline_stage("raw_bk256_load_k0", priority=0):
        a0 = a_buf.index(slot).slice(0, 128, 1).load(layout=DOT_A)
        b0 = b_buf.index(slot).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B)
        as0 = mxfp8_64_load_scale(as_buf, slot, 0, SCALE_A_LAYOUT)
        bs0 = mxfp8_64_load_scale(bs_buf, slot, 0, SCALE_B_LAYOUT)
    with gl.amd.warp_pipeline_stage("raw_bk256_compute_k0", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0, bs0, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("raw_bk256_load_k1", priority=0):
        a1 = a_buf.index(slot).slice(128, 128, 1).load(layout=DOT_A)
        b1 = b_buf.index(slot).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B)
        as1 = mxfp8_64_load_scale(as_buf, slot, 4, SCALE_A_LAYOUT)
        bs1 = mxfp8_64_load_scale(bs_buf, slot, 4, SCALE_B_LAYOUT)
    with gl.amd.warp_pipeline_stage(
            "raw_bk256_compute_k1_refill", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1, bs1, "e4m3", acc)
        gl.amd.gfx1250.cluster.arrive()
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc, as_desc, bs_desc = mxfp8_raw_bk256_issue_refill(
            a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
            refill_slot)
    return a_desc, b_desc, as_desc, bs_desc, acc


@gluon.jit
def mxfp8_raw_bk256_consume_tail(
        a_buf, b_buf, as_buf, bs_buf, slot, acc,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr):
    with gl.amd.warp_pipeline_stage("raw_bk256_tail_load_k0", priority=0):
        a0 = a_buf.index(slot).slice(0, 128, 1).load(layout=DOT_A)
        b0 = b_buf.index(slot).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B)
        as0 = mxfp8_64_load_scale(as_buf, slot, 0, SCALE_A_LAYOUT)
        bs0 = mxfp8_64_load_scale(bs_buf, slot, 0, SCALE_B_LAYOUT)
    with gl.amd.warp_pipeline_stage("raw_bk256_tail_compute_k0", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0, bs0, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("raw_bk256_tail_load_k1", priority=0):
        a1 = a_buf.index(slot).slice(128, 128, 1).load(layout=DOT_A)
        b1 = b_buf.index(slot).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B)
        as1 = mxfp8_64_load_scale(as_buf, slot, 4, SCALE_A_LAYOUT)
        bs1 = mxfp8_64_load_scale(bs_buf, slot, 4, SCALE_B_LAYOUT)
    with gl.amd.warp_pipeline_stage("raw_bk256_tail_compute_k1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1, bs1, "e4m3", acc)
    return acc


@gluon.jit
def fp8_scaled_cluster_128x128x256_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr):
    block_m: gl.constexpr = 512
    block_n: gl.constexpr = 512
    cta_n: gl.constexpr = 4
    gl.static_assert(gl.num_ctas() == 16)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
        M, N, block_m, block_n, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [block_m, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [block_n, 4])
    slots: gl.constexpr = 3
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [slots, block_m, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [slots, block_n, 256], SHARED_LAYOUT_B)
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty, [slots, block_m, 8], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty, [slots, block_n, 8], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * block_m * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(block_m, 256),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * block_n * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(block_n, 256),
        layout=SHARED_LAYOUT_B)
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + pid_m * block_m * stride_scale,
        shape=(M, K // 32), strides=(stride_scale, 1),
        block_shape=(block_m, 8), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + pid_n * block_n * stride_scale,
        shape=(N, K // 32), strides=(stride_scale, 1),
        block_shape=(block_n, 8), layout=SHARED_SCALE_B)
    for prefetch_idx in gl.static_range(2):
        a_desc, b_desc, as_desc, bs_desc = mxfp8_raw_bk256_issue_refill(
            a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
            prefetch_idx)
    acc = gl.zeros(
        (block_m, block_n), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 256)
    gl.assume(iter_max >= 3)
    for tile_idx in range(0, iter_max - 2):
        slot = tile_idx % 3
        refill_slot = (tile_idx + 2) % 3
        a_desc, b_desc, as_desc, bs_desc, acc = (
            mxfp8_raw_bk256_consume_refill(
                a_buf, b_buf, as_buf, bs_buf, slot, refill_slot,
                a_desc, b_desc, as_desc, bs_desc, acc, dot_a, dot_b,
                scale_a_layout, scale_b_layout))
    tdm.async_wait(3)
    penultimate_slot = (iter_max - 2) % 3
    acc = mxfp8_raw_bk256_consume_tail(
        a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout)
    tdm.async_wait(0)
    last_slot = (iter_max - 1) % 3
    acc = mxfp8_raw_bk256_consume_tail(
        a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout)
    snapshot_cluster_wait()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    snapshot_tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        WMMA_LAYOUT, block_m, block_n, cta_n)


@gluon.jit
def mxfp8_bk512_issue_refill(
        a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
        slot):
    tdm.async_load(
        a_desc, dest=a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(
        b_desc, dest=b_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load_fused([
        (as_desc, as_buf.index(slot), 0b00000011),
        (bs_desc, bs_buf.index(slot), 0b00001100),
    ])
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 512])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 512])
    as_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, 2048])
    bs_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, 2048])
    return a_desc, b_desc, as_desc, bs_desc


@gluon.jit
def mxfp8_bk512_consume(
        a_buf, b_buf, as_buf, bs_buf, slot, acc,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    with gl.amd.warp_pipeline_stage("bk512_load_0", priority=0):
        a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        as0 = mxfp8_load_scale(
            as_buf, slot, 0, SCALE_A_LAYOUT, BLOCK_M, 16, 4)
        b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs0 = mxfp8_load_scale(
            bs_buf, slot, 0, SCALE_B_LAYOUT, BLOCK_N, 16, 4)
    with gl.amd.warp_pipeline_stage("bk512_compute_0", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0, bs0, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("bk512_load_1", priority=0):
        a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        as1 = mxfp8_load_scale(
            as_buf, slot, 4, SCALE_A_LAYOUT, BLOCK_M, 16, 4)
        b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs1 = mxfp8_load_scale(
            bs_buf, slot, 4, SCALE_B_LAYOUT, BLOCK_N, 16, 4)
    with gl.amd.warp_pipeline_stage("bk512_compute_1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1, bs1, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("bk512_load_2", priority=0):
        a2 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            256, 128, 1).load(layout=DOT_A)
        as2 = mxfp8_load_scale(
            as_buf, slot, 8, SCALE_A_LAYOUT, BLOCK_M, 16, 4)
        b2 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            256, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs2 = mxfp8_load_scale(
            bs_buf, slot, 8, SCALE_B_LAYOUT, BLOCK_N, 16, 4)
    with gl.amd.warp_pipeline_stage("bk512_compute_2", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a2, as2, "e4m3", b2, bs2, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("bk512_load_3", priority=0):
        a3 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            384, 128, 1).load(layout=DOT_A)
        as3 = mxfp8_load_scale(
            as_buf, slot, 12, SCALE_A_LAYOUT, BLOCK_M, 16, 4)
        b3 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            384, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs3 = mxfp8_load_scale(
            bs_buf, slot, 12, SCALE_B_LAYOUT, BLOCK_N, 16, 4)
    with gl.amd.warp_pipeline_stage("bk512_compute_3", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a3, as3, "e4m3", b3, bs3, "e4m3", acc)
    return acc


@gluon.jit
def fp8_scaled_cluster_bk512_cta128_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        CTA_M: gl.constexpr, CTA_N: gl.constexpr):
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    gl.static_assert(BLOCK_M // CTA_M == 128)
    gl.static_assert(BLOCK_N // CTA_N == 128)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
        M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [BLOCK_M, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [BLOCK_N, 4])
    slots: gl.constexpr = 2
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [slots, BLOCK_M, 512], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [slots, BLOCK_N, 512], SHARED_LAYOUT_B)
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty,
        [slots, BLOCK_M // 128, 2048], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty,
        [slots, BLOCK_N // 128, 2048], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, 512),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, 512),
        layout=SHARED_LAYOUT_B)
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + (pid_m * BLOCK_M) // 128 * stride_scale,
        shape=(M // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(BLOCK_M // 128, 2048), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * BLOCK_N) // 128 * stride_scale,
        shape=(N // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(BLOCK_N // 128, 2048), layout=SHARED_SCALE_B)
    a_desc, b_desc, as_desc, bs_desc = mxfp8_bk512_issue_refill(
        a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, 0)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 512)
    gl.assume(iter_max >= 2)
    for tile_idx in range(0, iter_max - 1):
        slot = tile_idx % 2
        refill_slot = (tile_idx + 1) % 2
        with gl.amd.warp_pipeline_stage("bk512_refill", priority=0):
            gl.amd.gfx1250.cluster.arrive()
            gl.amd.gfx1250.cluster.wait()
            a_desc, b_desc, as_desc, bs_desc = mxfp8_bk512_issue_refill(
                a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                bs_buf, refill_slot)
        tdm.async_wait(3)
        acc = mxfp8_bk512_consume(
            a_buf, b_buf, as_buf, bs_buf, slot, acc, dot_a, dot_b,
            scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N)

    tdm.async_wait(0)
    snapshot_cluster_wait()
    last_slot = (iter_max - 1) % 2
    acc = mxfp8_bk512_consume(
        a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N)
    snapshot_cluster_wait()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    snapshot_tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        WMMA_LAYOUT, BLOCK_M, BLOCK_N, CTA_N)


@gluon.jit
def fp8_scaled_cluster_bk128_three_slot_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
        OUTPUT_CGA_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, CTA_M: gl.constexpr, CTA_N: gl.constexpr,
        REFILL_SCHEDULE: gl.constexpr):
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    gl.static_assert(BLOCK_M // CTA_M == 256)
    gl.static_assert(BLOCK_N // CTA_N == 256)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
        M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [BLOCK_M, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [BLOCK_N, 4])
    slots: gl.constexpr = 3
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [slots, BLOCK_M, 128], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [slots, BLOCK_N, 128], SHARED_LAYOUT_B)
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty,
        [slots, BLOCK_M // 128, 512], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty,
        [slots, BLOCK_N // 128, 512], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, 128),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, 128),
        layout=SHARED_LAYOUT_B)
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + (pid_m * BLOCK_M) // 128 * stride_scale,
        shape=(M // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(BLOCK_M // 128, 512), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * BLOCK_N) // 128 * stride_scale,
        shape=(N // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(BLOCK_N // 128, 512), layout=SHARED_SCALE_B)

    for prefetch_idx in gl.static_range(2):
        a_desc, b_desc, as_desc, bs_desc = mxfp8_bk128_issue_refill(
            a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
            prefetch_idx)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 128)
    gl.assume(iter_max >= 3)
    for tile_idx in range(0, iter_max - 2):
        slot = tile_idx % 3
        refill_slot = (tile_idx + 2) % 3
        a_desc, b_desc, as_desc, bs_desc, acc = (
            mxfp8_bk128_consume_and_refill(
                a_buf, b_buf, as_buf, bs_buf, slot, refill_slot,
                a_desc, b_desc, as_desc, bs_desc, acc, dot_a, dot_b,
                scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N,
                REFILL_SCHEDULE))

    tdm.async_wait(3)
    snapshot_cluster_wait()
    penultimate_slot = (iter_max - 2) % 3
    acc = mxfp8_bk128_consume_tail(
        a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N)
    tdm.async_wait(0)
    snapshot_cluster_wait()
    last_slot = (iter_max - 1) % 3
    acc = mxfp8_bk128_consume_tail(
        a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N)
    snapshot_cluster_wait()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    mxfp8_tdm_store_serial_n2(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        OUTPUT_CGA_LAYOUT, CTA_M, CTA_N)


def build_mxfp8_layouts(cluster_m=4, cluster_n=None):
    """Build cluster data, scale, and WMMA layouts for MXFP8.

    Data operands use the K128 scaled-WMMA instruction shape. Scale layouts
    reuse the corresponding A/B CGA bases, so data and its block-32 scales are
    owned by the same CTA partition. The 128-column N slice matches the serial
    epilogue and reduces accumulator register pressure enough to avoid spills.
    """
    cluster_n = cluster_m if cluster_n is None else cluster_n
    block_m = 256 * cluster_m
    block_n = 256 * cluster_n
    cga_c = make_cga_layout(
        [cluster_m, cluster_n], [cluster_m, cluster_n], [0, 1])
    output_cga = tuple(tuple(basis) + (0, 0) for basis in cga_c)
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_m, 256], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_n, 256], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, block_n, local_a, local_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=128)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga_c)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_m, 256], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_n, 256], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        256, 256, padded_a, padded_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=128)
    shared_as = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_m // 128, 1024], [1, 0], cga_a)
    shared_bs = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_n // 128, 1024], [1, 0], cga_b)
    return shared_a, shared_b, shared_as, shared_bs, wmma, output_cga


def build_mxfp8_bk256_opt_layouts(cluster_m=4, cluster_n=None):
    """Build BK256 operand layouts with a 128-column accumulator partition."""
    cluster_n = cluster_m if cluster_n is None else cluster_n
    block_m = 256 * cluster_m
    packed_n = 128 * cluster_n
    shared_a, shared_b, shared_as, shared_bs, load_wmma, output_cga = (
        build_mxfp8_layouts(cluster_m, cluster_n))
    cga_c = make_cga_layout(
        [cluster_m, cluster_n], [cluster_m, cluster_n], [0, 1])
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_m, 256], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [packed_n, 256], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, packed_n, local_a, local_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=128)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga_c)
    return (shared_a, shared_b, shared_as, shared_bs, wmma, load_wmma,
            output_cga)


def build_mxfp8_bk128_layouts(cluster_m=4, cluster_n=None):
    """Build three-slot BK128 MXFP8 layouts."""
    cluster_n = cluster_m if cluster_n is None else cluster_n
    block_m = 256 * cluster_m
    block_n = 256 * cluster_n
    cga_c = make_cga_layout(
        [cluster_m, cluster_n], [cluster_m, cluster_n], [0, 1])
    output_cga = tuple(tuple(basis) + (0, 0) for basis in cga_c)
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[128, 16]], [block_m, 128], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[128, 16]], [block_n, 128], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, block_n, local_a, local_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=128)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga_c)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[128, 16]], [block_m, 128], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[128, 16]], [block_n, 128], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        256, 256, padded_a, padded_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=128)
    shared_as = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_m // 128, 512], [1, 0], cga_a)
    shared_bs = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_n // 128, 512], [1, 0], cga_b)
    return shared_a, shared_b, shared_as, shared_bs, wmma, output_cga


def build_mxfp8_512x128x128_cga1_layouts():
    """Build three-slot layouts for a 512x128x128 single-CTA tile."""
    block_m = 512
    block_n = 128
    cga_c = make_cga_layout([1, 1], [1, 1], [0, 1])
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[128, 16]], [block_m, 128], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[128, 16]], [block_n, 128], [1, 0])
    shared_a, shared_b, local_wmma = (
        gl.amd.gfx1250.make_partitioned_dot_layouts(
            block_m, block_n, local_a, local_b, 8, [16, 16, 128],
            a_transposed=False, b_transposed=True,
            slice_m=block_m, slice_n=block_n))
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga_c)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    shared_as = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [4, 512], [1, 0], cga_a)
    shared_bs = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [1, 512], [1, 0], cga_b)
    return shared_a, shared_b, shared_as, shared_bs, wmma


def build_mxfp8_64x64x512_layouts(
        cluster_m=1, cluster_n=None, cta_m=64, cta_n=64):
    """Build two-slot BK512 layouts for a configurable CTA output tile."""
    cluster_n = cluster_m if cluster_n is None else cluster_n
    block_m = cta_m * cluster_m
    block_n = cta_n * cluster_n
    cga_c = make_cga_layout(
        [cluster_m, cluster_n], [cluster_m, cluster_n], [0, 1])
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [block_m, 512], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [block_n, 512], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, block_n, local_a, local_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True,
        slice_m=cta_m, slice_n=cta_n)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga_c)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [block_m, 512], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [block_n, 512], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        cta_m, cta_n, padded_a, padded_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True,
        slice_m=cta_m, slice_n=cta_n)
    shared_as = gl.PaddedSharedLayout.with_identity_for(
        [[16, 8]], [block_m, 16], [1, 0], cga_a)
    shared_bs = gl.PaddedSharedLayout.with_identity_for(
        [[16, 8]], [block_n, 16], [1, 0], cga_b)
    return shared_a, shared_b, shared_as, shared_bs, wmma


def build_mxfp8_64x64x1024_layouts(cluster_m=2, cluster_n=None):
    """Build two-slot layouts with a 64x64x1024 per-CTA tile."""
    cluster_n = cluster_m if cluster_n is None else cluster_n
    block_m = 64 * cluster_m
    block_n = 64 * cluster_n
    cga_c = make_cga_layout(
        [cluster_m, cluster_n], [cluster_m, cluster_n], [0, 1])
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[1024, 16]], [block_m, 1024], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[1024, 16]], [block_n, 1024], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, block_n, local_a, local_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=64, slice_n=64)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga_c)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[1024, 16]], [block_m, 1024], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[1024, 16]], [block_n, 1024], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        64, 64, padded_a, padded_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=64, slice_n=64)
    shared_as = gl.PaddedSharedLayout.with_identity_for(
        [[32, 8]], [block_m, 32], [1, 0], cga_a)
    shared_bs = gl.PaddedSharedLayout.with_identity_for(
        [[32, 8]], [block_n, 32], [1, 0], cga_b)
    return shared_a, shared_b, shared_as, shared_bs, wmma


def build_mxfp8_128x128x256_layouts():
    """Build three-slot layouts for a 128x128x256 CTA tile in a 4x4 CGA."""
    block_m = block_n = 512
    cga_c = make_cga_layout([4, 4], [4, 4], [0, 1])
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_m, 256], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_n, 256], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, block_n, local_a, local_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=128, slice_n=128)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga_c)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_m, 256], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_n, 256], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        128, 128, padded_a, padded_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=128, slice_n=128)
    shared_as = gl.PaddedSharedLayout.with_identity_for(
        [[8, 8]], [block_m, 8], [1, 0], cga_a)
    shared_bs = gl.PaddedSharedLayout.with_identity_for(
        [[8, 8]], [block_n, 8], [1, 0], cga_b)
    return shared_a, shared_b, shared_as, shared_bs, wmma


def build_mxfp8_bk512_cta128_layouts(cluster_m=4, cluster_n=None):
    """Build two-slot BK512 layouts with a 128x128 per-CTA output tile."""
    cluster_n = cluster_m if cluster_n is None else cluster_n
    block_m = 128 * cluster_m
    block_n = 128 * cluster_n
    cga_c = make_cga_layout(
        [cluster_m, cluster_n], [cluster_m, cluster_n], [0, 1])
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [block_m, 512], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [block_n, 512], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, block_n, local_a, local_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=128, slice_n=128)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga_c)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [block_m, 512], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [block_n, 512], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        128, 128, padded_a, padded_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=128, slice_n=128)
    shared_as = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_m // 128, 2048], [1, 0], cga_a)
    shared_bs = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_n // 128, 2048], [1, 0], cga_b)
    return shared_a, shared_b, shared_as, shared_bs, wmma


# ---------------------------------------------------------------------------
# FP8 x MXFP4.
#
# A is ordinary E4M3, so its scale operand is ``None`` (mathematically one).
# B is packed E2M1 with block-32 E8M0 scales. One BK256 logical tile therefore
# occupies 256 A columns but only 128 packed B bytes.
#
# The default launch uses a three-slot TDM ring, a single-slot prologue, a
# warp-pipelined drain, and A-reusing output-tile order. Unlike the other peak
# paths, cluster width, ring depth, and reuse direction remain CLI-selectable.

@gluon.jit
def fp8_mxfp4_load_scale(
        scale_buffer, slot, start_k: gl.constexpr, LAYOUT: gl.constexpr,
        BLOCK_N: gl.constexpr):
    scale_slice = scale_buffer.index(slot).reshape(
        (BLOCK_N // 128, 2, 32, 4, 4)
    ).permute((0, 3, 2, 1, 4)).reshape((BLOCK_N, 8))
    return scale_slice.slice(0, BLOCK_N, 0).slice(
        start_k, 4, 1).load(layout=LAYOUT)


@gluon.jit
def fp8_mxfp4_issue_leading4_data(
        a_desc, b_desc, a_buf, b_buf, slot, B_WARP_HINT: gl.constexpr):
    tdm.async_load_fused([
        (a_desc, a_buf.index(slot), 0b00000011),
        (b_desc, b_buf.index(slot), B_WARP_HINT),
    ])
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 128])
    return a_desc, b_desc


@gluon.jit
def fp8_mxfp4_issue_leading4_scale(
        bs_desc, bs_buf, slot):
    # Only B is quantized here, so this copy has no A-scale partner to be
    # disjoint from. Standalone copies stay separate by contract, so it takes
    # all four of the warps a TDM copy is allowed rather than the warps 2 and
    # 3 a fused pair would have left it.
    tdm.async_load(
        bs_desc, dest=bs_buf.index(slot), warp_used_hint=0b00001111)
    return tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, 1024])


@gluon.jit
def fp8_mxfp4_consume_tile(
        a_buf, b_buf, bs_buf, slot, acc, DOT_A: gl.constexpr,
        DOT_B: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
        0, 128, 1).load(layout=DOT_A)
    b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
        0, 64, 1).permute([1, 0]).load(layout=DOT_B)
    bs0 = fp8_mxfp4_load_scale(
        bs_buf, slot, 0, SCALE_B_LAYOUT, BLOCK_N)
    acc = gl.amd.gfx1250.wmma_scaled(
        a0, None, "e4m3", b0, bs0, "e2m1", acc)
    a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
        128, 128, 1).load(layout=DOT_A)
    b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
        64, 64, 1).permute([1, 0]).load(layout=DOT_B)
    bs1 = fp8_mxfp4_load_scale(
        bs_buf, slot, 4, SCALE_B_LAYOUT, BLOCK_N)
    return gl.amd.gfx1250.wmma_scaled(
        a1, None, "e4m3", b1, bs1, "e2m1", acc)


@gluon.jit
def fp8_mxfp4_consume_pipelined_tile(
        a_buf, b_buf, bs_buf, slot, acc, DOT_A: gl.constexpr,
        DOT_B: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    with gl.amd.warp_pipeline_stage("stage0_tail", priority=0):
        a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            0, 64, 1).permute([1, 0]).load(layout=DOT_B)
        bs0 = fp8_mxfp4_load_scale(
            bs_buf, slot, 0, SCALE_B_LAYOUT, BLOCK_N)
    with gl.amd.warp_pipeline_stage("stage1_tail", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, None, "e4m3", b0, bs0, "e2m1", acc)
    with gl.amd.warp_pipeline_stage("stage0_tail", priority=0):
        a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            64, 64, 1).permute([1, 0]).load(layout=DOT_B)
        bs1 = fp8_mxfp4_load_scale(
            bs_buf, slot, 4, SCALE_B_LAYOUT, BLOCK_N)
    with gl.amd.warp_pipeline_stage("stage1_tail", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, None, "e4m3", b1, bs1, "e2m1", acc)
    return acc


@gluon.jit
def fp8_mxfp4_consume_and_refill(
        a_buf, b_buf, bs_buf, slot, refill_slot, a_desc, b_desc, bs_desc,
        acc, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr, B_WARP_HINT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        CLUSTERED: gl.constexpr, RING_WAIT: gl.constexpr):
    tdm.async_wait(RING_WAIT)
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        # ``refill_slot`` was drained by the previous tile.
        if CLUSTERED:
            gl.amd.gfx1250.cluster.arrive()
            gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc = fp8_mxfp4_issue_leading4_data(
            a_desc, b_desc, a_buf, b_buf, refill_slot, B_WARP_HINT)
        bs_desc = fp8_mxfp4_issue_leading4_scale(
            bs_desc, bs_buf, refill_slot)
        a0 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        b0 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            0, 64, 1).permute([1, 0]).load(layout=DOT_B)
        bs0 = fp8_mxfp4_load_scale(
            bs_buf, slot, 0, SCALE_B_LAYOUT, BLOCK_N)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, None, "e4m3", b0, bs0, "e2m1", acc)
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a1 = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        b1 = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
            64, 64, 1).permute([1, 0]).load(layout=DOT_B)
        bs1 = fp8_mxfp4_load_scale(
            bs_buf, slot, 4, SCALE_B_LAYOUT, BLOCK_N)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, None, "e4m3", b1, bs1, "e2m1", acc)
    return a_desc, b_desc, bs_desc, acc


@gluon.jit
def fp8_mxfp4_cluster_gemm_gfx1250(
        a_ptr, b_ptr, c_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr,
        SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
        PIPELINE_TAIL: gl.constexpr, SINGLE_SLOT_PROLOGUE: gl.constexpr,
        NUM_BUFFERS: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, CTA_M: gl.constexpr, CTA_N: gl.constexpr,
        REUSE_A: gl.constexpr, GROUP_SIZE: gl.constexpr):
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    gl.static_assert(BLOCK_M // CTA_M == 256)
    gl.static_assert(BLOCK_N // CTA_N == 256)
    clustered: gl.constexpr = CTA_M * CTA_N > 1
    if REUSE_A:
        pid_m, pid_n = snapshot_get_xcd_swizzled_pids_reuse_a(
            M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, GROUP_SIZE)
    else:
        pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
            M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, GROUP_SIZE)
    wmma_packed: gl.constexpr = gl.amd.AMDWMMALayout(
        3, WMMA_LAYOUT.transposed, WMMA_LAYOUT.warp_bases,
        WMMA_LAYOUT.reg_bases, [16, 16, 64], WMMA_LAYOUT.cga_layout)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma_packed, 16)
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [BLOCK_N, 4])

    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [NUM_BUFFERS, BLOCK_M, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [NUM_BUFFERS, BLOCK_N, 128], SHARED_LAYOUT_B)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_N // 128, 1024], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, 256),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K // 2),
        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, 128),
        layout=SHARED_LAYOUT_B)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * BLOCK_N) // 128 * stride_scale,
        shape=(N // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(BLOCK_N // 128, 1024), layout=SHARED_SCALE_B)
    b_warp_hint: gl.constexpr = 0b00001100

    # Two descriptors per tile: the fused A/B data pair, and the lone B scale
    # copy. Only B is quantized, so there is no A scale.
    #
    # The prologue fills NUM_BUFFERS - 1 slots; tile t refills the slot tile
    # t - 1 consumed, supplying tile t + NUM_BUFFERS - 1. At the top of a tile
    # the refill it needs was issued NUM_BUFFERS - 1 tiles earlier, so
    # NUM_BUFFERS - 2 later refills may still be in flight.
    #
    # The wait stays ahead of the issue here, unlike MXFP8 and MXFP4. At the
    # default three slots this leaves one later tile in flight. Moving the wait
    # after the issue preserves two tiles (count 4), but the extra stage region
    # measured 0.43% slower; one retained tile already saturates the engine.
    # The split is only worthwhile on two-slot rings, where an ahead-of-issue
    # wait would pin the count to 0 and cancel all prefetch.
    ring_wait: gl.constexpr = 2 * (NUM_BUFFERS - 2)
    if SINGLE_SLOT_PROLOGUE:
        a_desc, b_desc = fp8_mxfp4_issue_leading4_data(
            a_desc, b_desc, a_buf, b_buf, 0, b_warp_hint)
        bs_desc = fp8_mxfp4_issue_leading4_scale(bs_desc, bs_buf, 0)
        tdm.async_wait(0)
        if clustered:
            snapshot_cluster_wait()
        for prefetch_idx in gl.static_range(1, NUM_BUFFERS - 1):
            a_desc, b_desc = fp8_mxfp4_issue_leading4_data(
                a_desc, b_desc, a_buf, b_buf, prefetch_idx, b_warp_hint)
            bs_desc = fp8_mxfp4_issue_leading4_scale(
                bs_desc, bs_buf, prefetch_idx)
    else:
        for prefetch_idx in gl.static_range(NUM_BUFFERS - 1):
            a_desc, b_desc = fp8_mxfp4_issue_leading4_data(
                a_desc, b_desc, a_buf, b_buf, prefetch_idx, b_warp_hint)
            bs_desc = fp8_mxfp4_issue_leading4_scale(
                bs_desc, bs_buf, prefetch_idx)
        tdm.async_wait(ring_wait)
        if clustered:
            snapshot_cluster_wait()

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 256)
    gl.assume(iter_max >= NUM_BUFFERS)
    for tile_idx in range(0, iter_max - NUM_BUFFERS + 1):
        slot = tile_idx % NUM_BUFFERS
        a_desc, b_desc, bs_desc, acc = fp8_mxfp4_consume_and_refill(
            a_buf, b_buf, bs_buf, slot, (tile_idx + NUM_BUFFERS - 1) %
            NUM_BUFFERS, a_desc, b_desc, bs_desc, acc, dot_a, dot_b,
            scale_b_layout,
            b_warp_hint, BLOCK_M, BLOCK_N, clustered, ring_wait)

    # One fewer drain tile than before, since the loop now covers one more.
    # Each drain tile needs its own rendezvous because it has no successor. The
    # first wait/barrier is hoisted out: the pass rejects a wait that falls
    # between two pipelined tail regions without a preceding one.
    tdm.async_wait(2 * (NUM_BUFFERS - 2))
    if clustered:
        snapshot_cluster_wait()
    for drain_idx in gl.static_range(NUM_BUFFERS - 1):
        slot = (iter_max - NUM_BUFFERS + 1 + drain_idx) % NUM_BUFFERS
        if PIPELINE_TAIL:
            acc = fp8_mxfp4_consume_pipelined_tile(
                a_buf, b_buf, bs_buf, slot, acc, dot_a, dot_b,
                scale_b_layout, BLOCK_M, BLOCK_N)
        else:
            acc = fp8_mxfp4_consume_tile(
                a_buf, b_buf, bs_buf, slot, acc, dot_a, dot_b,
                scale_b_layout, BLOCK_M, BLOCK_N)
        if drain_idx < NUM_BUFFERS - 2:
            tdm.async_wait(2 * (NUM_BUFFERS - 3 - drain_idx))
            if clustered:
                snapshot_cluster_wait()
    if clustered:
        snapshot_cluster_wait()
    a_buf._keep_alive()
    b_buf._keep_alive()
    bs_buf._keep_alive()
    snapshot_tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        WMMA_LAYOUT, BLOCK_M, BLOCK_N, CTA_N)


def build_fp8_mxfp4_layouts(cluster_width=4):
    """Build mixed-format layouts for a square CTA cluster.

    The E4M3 A tile has 256 stored K elements while packed E2M1 B has 128
    stored bytes for the same logical K extent. ``wmma_packed`` changes only
    B's instruction K shape; output ownership remains compatible with A.
    """
    block_m = 256 * cluster_width
    block_n = 256 * cluster_width
    cga = make_cga_layout(
        [cluster_width, cluster_width], [cluster_width, cluster_width], [0, 1])
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_m, 256], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[128, 16]], [block_n, 128], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, block_n, local_a, local_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=256)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga)
    wmma_packed = gl.amd.AMDWMMALayout(
        3, wmma.transposed, wmma.warp_bases, wmma.reg_bases,
        [16, 16, 64], cga)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma_packed, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [block_m, 256], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[128, 16]], [block_n, 128], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        256, 256, padded_a, padded_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=256)
    shared_bs = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [block_n // 128, 1024], [1, 0], cga_b)
    return shared_a, shared_b, shared_bs, wmma


# ---------------------------------------------------------------------------
# MXFP4 x MXFP4.
#
# Aggregate tile: 1024x1024x512 logical values across a 4x4 cluster. Packed A/B
# storage reduces that K extent to 256 bytes. Each slot is consumed as two K256
# groups, and each group contains two K128 scaled WMMAs.
#
# The refill now targets the slot the previous tile consumed, issued behind an
# adjacent arrive/wait pair at the top of the tile. A third slot would restore
# the lead this gives up, but it needs 443,128 bytes of LDS against a 327,680
# limit, so two is forced.
MXFP4_NUM_SLOTS = gl.constexpr(2)
MXFP4_TDM_PER_TILE = gl.constexpr(2)


@gluon.jit
def mxfp4_load_scale(
        scale_buffer, slot, start_k: gl.constexpr, LAYOUT: gl.constexpr):
    scale_slice = scale_buffer.index(slot).reshape(
        (8, 4, 32, 4, 4)).permute((0, 3, 2, 1, 4)).reshape((1024, 16))
    return scale_slice.slice(0, 1024, 0).slice(
        start_k, 4, 1).load(layout=LAYOUT)


@gluon.jit
def mxfp4_issue_refill(
        a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
        slot):
    # MXFP4 A and B each have two logical partition pieces, so two issuing
    # warps cover either full operand. This differs from MXFP8 B, whose two
    # partitions and two groups require four warps and force separate A/B
    # copies. Here the disjoint 0-1/2-3 hints are both sufficient and keep the
    # fused operation entirely on the leading wave group.
    tdm.async_load_fused([
        (a_desc, a_buf.index(slot), 0b00000011),
        (b_desc, b_buf.index(slot), 0b00001100),
    ])
    tdm.async_load_fused([
        (as_desc, as_buf.index(slot), 0b00000011),
        (bs_desc, bs_buf.index(slot), 0b00001100),
    ])
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 256])
    as_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, 2048])
    bs_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, 2048])
    return a_desc, b_desc, as_desc, bs_desc


@gluon.jit
def mxfp4_consume_k256(
        a_buf, b_buf, as_buf, bs_buf, slot, acc,
        START_K_PACKED: gl.constexpr, START_SCALE_K: gl.constexpr,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr):
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a0 = a_buf.index(slot).slice(0, 1024, 0).slice(
            START_K_PACKED, 64, 1).load(layout=DOT_A)
        as0 = mxfp4_load_scale(
            as_buf, slot, START_SCALE_K, SCALE_A_LAYOUT)
        b0 = b_buf.index(slot).slice(0, 1024, 0).slice(
            START_K_PACKED, 64, 1).permute([1, 0]).load(layout=DOT_B)
        bs0 = mxfp4_load_scale(
            bs_buf, slot, START_SCALE_K, SCALE_B_LAYOUT)
        a1 = a_buf.index(slot).slice(0, 1024, 0).slice(
            START_K_PACKED + 64, 64, 1).load(layout=DOT_A)
        as1 = mxfp4_load_scale(
            as_buf, slot, START_SCALE_K + 4, SCALE_A_LAYOUT)
        b1 = b_buf.index(slot).slice(0, 1024, 0).slice(
            START_K_PACKED + 64, 64, 1).permute([1, 0]).load(layout=DOT_B)
        bs1 = mxfp4_load_scale(
            bs_buf, slot, START_SCALE_K + 4, SCALE_B_LAYOUT)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e2m1", b0, bs0, "e2m1", acc)
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e2m1", b1, bs1, "e2m1", acc)
    return acc


@gluon.jit
def mxfp4_consume_tile(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, DOT_A: gl.constexpr,
        DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr):
    # The retained path deliberately keeps the original K256 grouping/order.
    acc = mxfp4_consume_k256(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, 0, 0, DOT_A, DOT_B,
        SCALE_A_LAYOUT, SCALE_B_LAYOUT)
    return mxfp4_consume_k256(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, 128, 8, DOT_A, DOT_B,
        SCALE_A_LAYOUT, SCALE_B_LAYOUT)


@gluon.jit
def mxfp4_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, a_desc, b_desc, as_desc, bs_desc,
        acc, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        SLOT: gl.constexpr, REFILL_SLOT: gl.constexpr,
        RING_WAIT: gl.constexpr):
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        gl.amd.gfx1250.cluster.arrive()
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc, as_desc, bs_desc = mxfp4_issue_refill(
            a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
            REFILL_SLOT)
    tdm.async_wait(RING_WAIT)
    acc = mxfp4_consume_tile(
        a_buf, b_buf, as_buf, bs_buf, SLOT, acc, DOT_A, DOT_B,
        SCALE_A_LAYOUT, SCALE_B_LAYOUT)
    return a_desc, b_desc, as_desc, bs_desc, acc


@gluon.jit
def mxfp4_bk512_warp_pipeline_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr):
    gl.static_assert(gl.num_ctas() == 16)
    pid_m, pid_n = snapshot_get_xcd_swizzled_pids(
        M, N, 1024, 1024, GRID_MN, 8, 4)
    wmma_packed: gl.constexpr = gl.amd.AMDWMMALayout(
        3, WMMA_LAYOUT.transposed, WMMA_LAYOUT.warp_bases,
        WMMA_LAYOUT.reg_bases, [32, 16, 64], WMMA_LAYOUT.cga_layout)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma_packed, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma_packed, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [1024, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [1024, 4])
    slots: gl.constexpr = MXFP4_NUM_SLOTS
    # Two descriptors per tile: one fused data pair and one fused scale pair.
    # The wait follows the issue, so one tile stays in flight.
    ring_wait: gl.constexpr = MXFP4_TDM_PER_TILE * (slots - 1)
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [slots, 1024, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [slots, 1024, 256], SHARED_LAYOUT_B)
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty, [slots, 8, 2048], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty, [slots, 8, 2048], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * 1024 * stride_am, shape=(M, K // 2),
        strides=(stride_am, stride_ak), block_shape=(1024, 256),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * 1024 * stride_bn, shape=(N, K // 2),
        strides=(stride_bn, stride_bk), block_shape=(1024, 256),
        layout=SHARED_LAYOUT_B)
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + (pid_m * 1024) // 128 * stride_scale,
        shape=(M // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(8, 2048), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * 1024) // 128 * stride_scale,
        shape=(N // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(8, 2048), layout=SHARED_SCALE_B)
    # Only slot 0 is prefetched; tile t's refill supplies tile t + 1.
    a_desc, b_desc, as_desc, bs_desc = mxfp4_issue_refill(
        a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, 0)
    acc = gl.zeros((1024, 1024), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 512)
    gl.assume(iter_max >= 2)
    gl.assume((iter_max % 2) == 0)
    # Fixed slot indices let buffer analysis prove each refill disjoint from
    # the tile being read and remove the dynamic modulo/address selection.
    for _ in range(0, (iter_max - 2) // 2):
        a_desc, b_desc, as_desc, bs_desc, acc = mxfp4_consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, a_desc, b_desc, as_desc, bs_desc,
            acc, dot_a, dot_b, scale_a_layout, scale_b_layout,
            0, 1, ring_wait)
        a_desc, b_desc, as_desc, bs_desc, acc = mxfp4_consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, a_desc, b_desc, as_desc, bs_desc,
            acc, dot_a, dot_b, scale_a_layout, scale_b_layout,
            1, 0, ring_wait)
    a_desc, b_desc, as_desc, bs_desc, acc = mxfp4_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, a_desc, b_desc, as_desc, bs_desc,
        acc, dot_a, dot_b, scale_a_layout, scale_b_layout,
        0, 1, ring_wait)
    # ``ring_wait`` belongs to the main loop only. The tail is the last
    # consumer, so it drains to 0.
    tdm.async_wait(0)
    snapshot_cluster_wait()
    acc = mxfp4_consume_tile(
        a_buf, b_buf, as_buf, bs_buf, 1, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout)
    # Output LDS aliases the cluster-distributed operand arena. Match MXFP8's
    # final rendezvous so no CTA can repurpose its local partition while a peer
    # CTA is still reading the last operand slot.
    snapshot_cluster_wait()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    snapshot_tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        WMMA_LAYOUT, 1024, 1024, 4)


def build_mxfp4_layouts():
    """Build packed-data and scale layouts for the BK512 MXFP4 kernel.

    Both operands use packed K64 fragments and block-32 scales. Their CGA bases
    are mirrored because B is loaded in [N,K] order and transposed for WMMA.
    """
    cga = make_cga_layout([4, 4], [4, 4], [0, 1])
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        1024, 1024, local_a, local_b, 8, [32, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=256)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        256, 256, padded_a, padded_b, 8, [32, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=256)
    shared_as = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [8, 2048], [1, 0], cga_a)
    shared_bs = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [8, 2048], [1, 0], cga_b)
    return shared_a, shared_b, shared_as, shared_bs, wmma


# ---------------------------------------------------------------------------
# Host-side input, launch, correctness, and benchmark glue.
#
# Storage conventions:
# * B is passed to kernels as an [N, K] contiguous transpose.
# * MXFP4 operands are packed along K before transfer to the GPU.
# * E8M0 scales are preshuffled in groups of 128 non-K values so LDS fragments
#   match ``get_wmma_scale_layout`` without a runtime transpose.
# * Every retained kernel accumulates in FP32 and writes BF16 by default.
#
# Benchmarking performs an event-timed probe, captures repeated launches in a
# CUDA/HIP graph, and reports the average latency over all graph replays.
# ``gpu-lock`` only serializes access; it does not pin clocks or disable DVFS.

def make_trig_tensor(shape, dtype, cosine):
    """Create a CPU tensor filled with sin(i) or cos(i) in flat-index order."""
    result = torch.empty(shape, dtype=dtype)
    flat = result.view(-1)
    chunk = 8 * 1024 * 1024
    for begin in range(0, flat.numel(), chunk):
        end = min(begin + chunk, flat.numel())
        angle = torch.arange(begin, end, dtype=torch.float64)
        values = torch.cos(angle) if cosine else torch.sin(angle)
        flat[begin:end].copy_(values.float())
    return result


def pack_scale(scale, scale_kwidth):
    """Preshuffle [non-K, K-block] E8M0 scales for scaled-WMMA LDS loads.

    The factor-128 permutation groups the four scale lanes consumed by each
    WMMA scale fragment. The returned 2D tensor is the global-memory format
    expected by the scale TDM descriptors in this module.
    """
    non_k, k_scale = scale.shape
    scale = scale.view(
        non_k // 128, 4, 32, k_scale // scale_kwidth, scale_kwidth)
    scale = scale.permute(0, 3, 2, 1, 4).contiguous()
    return scale.view(non_k // 128, k_scale * 128)


def make_bf16_case(args, eight_stage=False):
    """Build BF16 launch/check closures while keeping their GPU tensors alive."""
    if args.M % 1024 or args.N % 1024 or args.K % 128:
        raise ValueError("BF16 requires M/N divisible by 1024 and K by 128")
    if args.K // 128 < 2 or (args.K // 128) % 2:
        raise ValueError("BF16 requires an even number of at least two BK128 tiles")
    mode = args.input_mode or "trig"
    if mode == "random":
        torch.manual_seed(args.seed)
        a = torch.randn((args.M, args.K), dtype=torch.bfloat16)
        b = torch.randn((args.K, args.N), dtype=torch.bfloat16).T.contiguous()
        a = a.cuda()
        b = b.cuda()
    else:
        device = torch.device("cuda")
        a = torch.empty(
            (args.M, args.K), dtype=torch.bfloat16, device=device)
        b = torch.empty(
            (args.N, args.K), dtype=torch.bfloat16, device=device)
        chunk = 8 * 1024 * 1024
        for output, cosine in ((a, False), (b, True)):
            flat = output.view(-1)
            for begin in range(0, flat.numel(), chunk):
                end = min(begin + chunk, flat.numel())
                angle = torch.arange(
                    begin, end, dtype=torch.float64, device=device)
                value = torch.cos(angle) if cosine else torch.sin(angle)
                flat[begin:end].copy_(value.float())
    output_dtype = torch.bfloat16 if args.bf16_output else torch.float32
    c = torch.zeros((args.M, args.N), dtype=output_dtype, device="cuda")
    layouts = build_bf16_8stage_layouts() if eight_stage else build_bf16_layouts()
    grid = (triton.cdiv(args.M, 1024) * triton.cdiv(args.N, 1024), 1)

    def launch():
        if eight_stage:
            shared_a, shared_b, wmma, load_wmma, output_cga = layouts
            return bf16_bk128_8stage_cluster_4x4_gfx1250[grid](
                a, b, c, args.M, args.N, args.K,
                a.stride(0), a.stride(1), b.stride(1), b.stride(0),
                c.stride(0), c.stride(1), GRID_MN=grid[0],
                SHARED_LAYOUT_A=shared_a, SHARED_LAYOUT_B=shared_b,
                WMMA_LAYOUT=wmma, LOAD_WMMA_LAYOUT=load_wmma,
                OUTPUT_CGA_LAYOUT=output_cga,
                num_warps=8, waves_per_eu=2, num_ctas=16)
        shared_a, shared_b, wmma = layouts
        return bf16_kernelc_2pf_bk128_fused_cluster_4x4_gfx1250[grid](
            a, b, c, args.M, args.N, args.K, a.stride(0), a.stride(1),
            b.stride(1), b.stride(0), c.stride(0), c.stride(1),
            GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, WMMA_LAYOUT=wmma,
            num_warps=8, waves_per_eu=2, num_ctas=16)

    def check():
        c.zero_()
        launch()
        torch.cuda.synchronize()
        reference = (a.cpu().float() @ b.cpu().T.float()).to(output_dtype)
        rtol = 1e-2 if output_dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(c.cpu(), reference, rtol=rtol, atol=1e-2)
        print("result verified", flush=True)

    return launch, check


def make_bf16_8stage_case(args):
    return make_bf16_case(args, eight_stage=True)


def make_mxfp8_case(
        args, block_k=256, cta_tile=256, experimental_bk256=False,
        single_cta_skinny=False, cta64_bk512_cluster=None,
        cta64_bk1024_cluster=None, cta128x64_bk512_cluster=None,
        cta64x128_bk512_cluster=None, raw_128_bk256=False):
    """Build the E4M3/E8M0 block-32 cluster benchmark case."""
    if raw_128_bk256:
        cluster_m = cluster_n = 4
        block_m = block_n = 512
    elif cta64x128_bk512_cluster is not None:
        cluster_m = cluster_n = cta64x128_bk512_cluster
        block_m = 64 * cluster_m
        block_n = 128 * cluster_n
    elif cta128x64_bk512_cluster is not None:
        cluster_m = cluster_n = cta128x64_bk512_cluster
        block_m = 128 * cluster_m
        block_n = 64 * cluster_n
    elif cta64_bk1024_cluster is not None:
        cluster_m = cluster_n = cta64_bk1024_cluster
        block_m = 64 * cluster_m
        block_n = 64 * cluster_n
    elif cta64_bk512_cluster is not None:
        cluster_m = cluster_n = cta64_bk512_cluster
        block_m = 64 * cluster_m
        block_n = 64 * cluster_n
    elif single_cta_skinny:
        cluster_m = cluster_n = 1
        block_m, block_n = 512, 128
    else:
        cluster_shape = getattr(args, "mxfp8_cluster_shape", None)
        if cluster_shape is None:
            cluster_m = cluster_n = args.mxfp8_cluster_width
        else:
            cluster_m, cluster_n = cluster_shape
        block_m = cta_tile * cluster_m
        block_n = cta_tile * cluster_n
    if args.M % block_m or args.N % block_n or args.K % block_k:
        raise ValueError(
            f"MXFP8 requires M/N divisible by {block_m} and K by {block_k}")
    min_tiles = 3 if (
        block_k == 128 or experimental_bk256 or
        cta64_bk512_cluster is not None or
        cta128x64_bk512_cluster is not None or
        cta64x128_bk512_cluster is not None or raw_128_bk256
    ) else 2
    if args.K // block_k < min_tiles:
        raise ValueError(
            f"MXFP8 BK{block_k} requires at least {min_tiles} tiles")
    raw_scale_layout = (
        cta64_bk512_cluster is not None or
        cta64_bk1024_cluster is not None or
        cta128x64_bk512_cluster is not None or
        cta64x128_bk512_cluster is not None or raw_128_bk256)
    required_mn_alignment = 64 if raw_scale_layout else 128
    if (args.M % required_mn_alignment or
            args.N % required_mn_alignment or (args.K // 32) % 4):
        raise ValueError("MXFP8 dimensions are incompatible with scale packing")
    torch.manual_seed(args.seed)
    mode = args.input_mode or "trig"
    if mode == "trig":
        a = make_trig_tensor(
            (args.M, args.K), torch.float8_e4m3fn, False)
        b = make_trig_tensor(
            (args.K, args.N), torch.float8_e4m3fn, True)
    else:
        a = init_data("float8_e4m3", args.M, args.K)
        b = init_data("float8_e4m3", args.K, args.N)
    scale_k = args.K // 32
    a_scale_obj = MXScaleTensor(
        size=(args.M, scale_k)).random(low=1.0, high=32.0)
    b_scale_obj = MXScaleTensor(
        size=(args.N, scale_k)).random(low=1.0, high=32.0)
    c_ref = None
    if args.check:
        c_ref = torch_gemm_mxfp(
            a, b, a_scale_obj, b_scale_obj, 32,
            args.M, args.N, args.K).to(torch.bfloat16)
    if raw_scale_layout:
        a_scale = a_scale_obj.data.contiguous()
        b_scale = b_scale_obj.data.contiguous()
    else:
        a_scale = pack_scale(a_scale_obj.data, 4)
        b_scale = pack_scale(b_scale_obj.data, 4)
    a_d = a.contiguous().cuda()
    b_d = b.T.contiguous().cuda()
    as_d = a_scale.cuda()
    bs_d = b_scale.cuda()
    c_d = torch.empty(
        (args.M, args.N), dtype=torch.bfloat16, device="cuda")
    if raw_128_bk256:
        layouts = build_mxfp8_128x128x256_layouts()
    elif cta64x128_bk512_cluster is not None:
        layouts = build_mxfp8_64x64x512_layouts(
            cluster_m, cluster_n, cta_m=64, cta_n=128)
    elif cta128x64_bk512_cluster is not None:
        layouts = build_mxfp8_64x64x512_layouts(
            cluster_m, cluster_n, cta_m=128, cta_n=64)
    elif cta64_bk1024_cluster is not None:
        layouts = build_mxfp8_64x64x1024_layouts(cluster_m, cluster_n)
    elif cta64_bk512_cluster is not None:
        layouts = build_mxfp8_64x64x512_layouts(cluster_m, cluster_n)
    elif single_cta_skinny:
        layouts = build_mxfp8_512x128x128_cga1_layouts()
    elif block_k == 128:
        layouts = build_mxfp8_bk128_layouts(cluster_m, cluster_n)
    elif experimental_bk256:
        layouts = build_mxfp8_bk256_opt_layouts(cluster_m, cluster_n)
    elif block_k == 512 and cta_tile == 128:
        layouts = build_mxfp8_bk512_cta128_layouts(cluster_m, cluster_n)
    else:
        layouts = build_mxfp8_layouts(cluster_m, cluster_n)
    grid = (
        triton.cdiv(args.M, block_m) * triton.cdiv(args.N, block_n), 1)
    refill_schedule = {
        "refill-load": 0,
        "post-wmma": 1,
        "arrive-load-wait-refill": 2,
    }[getattr(args, "mxfp8_bk128_schedule", "refill-load")]

    def launch():
        if raw_128_bk256:
            shared_a, shared_b, shared_as, shared_bs, wmma = layouts
            return fp8_scaled_cluster_128x128x256_kernel_gfx1250[grid](
                a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
                a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
                c_d.stride(0), c_d.stride(1), as_d.stride(0),
                GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
                SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
                SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
                num_warps=8, waves_per_eu=2, num_ctas=16,
                llvm_fn_attrs=AGPR_ATTRS)
        if cta64x128_bk512_cluster is not None:
            shared_a, shared_b, shared_as, shared_bs, wmma = layouts
            return fp8_scaled_cluster_cta_bk512_kernel_gfx1250[grid](
                a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
                a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
                c_d.stride(0), c_d.stride(1), as_d.stride(0),
                GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
                SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
                SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
                BLOCK_M=block_m, BLOCK_N=block_n,
                CTA_M=cluster_m, CTA_N=cluster_n,
                CTA_TILE_M=64, CTA_TILE_N=128, CLUSTERED=True,
                num_warps=8, waves_per_eu=2,
                num_ctas=cluster_m * cluster_n,
                llvm_fn_attrs=AGPR_ATTRS)
        if cta128x64_bk512_cluster is not None:
            shared_a, shared_b, shared_as, shared_bs, wmma = layouts
            return fp8_scaled_cluster_cta_bk512_kernel_gfx1250[grid](
                a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
                a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
                c_d.stride(0), c_d.stride(1), as_d.stride(0),
                GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
                SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
                SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
                BLOCK_M=block_m, BLOCK_N=block_n,
                CTA_M=cluster_m, CTA_N=cluster_n,
                CTA_TILE_M=128, CTA_TILE_N=64, CLUSTERED=True,
                num_warps=8, waves_per_eu=2,
                num_ctas=cluster_m * cluster_n,
                llvm_fn_attrs=AGPR_ATTRS)
        if cta64_bk1024_cluster is not None:
            shared_a, shared_b, shared_as, shared_bs, wmma = layouts
            return fp8_scaled_cluster_64x64x1024_kernel_gfx1250[grid](
                a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
                a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
                c_d.stride(0), c_d.stride(1), as_d.stride(0),
                GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
                SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
                SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
                BLOCK_M=block_m, BLOCK_N=block_n,
                CTA_M=cluster_m, CTA_N=cluster_n,
                num_warps=8, waves_per_eu=2,
                num_ctas=cluster_m * cluster_n,
                llvm_fn_attrs=AGPR_ATTRS)
        if cta64_bk512_cluster is not None:
            shared_a, shared_b, shared_as, shared_bs, wmma = layouts
            return fp8_scaled_cluster_cta_bk512_kernel_gfx1250[grid](
                a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
                a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
                c_d.stride(0), c_d.stride(1), as_d.stride(0),
                GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
                SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
                SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
                BLOCK_M=block_m, BLOCK_N=block_n,
                CTA_M=cluster_m, CTA_N=cluster_n,
                CTA_TILE_M=64, CTA_TILE_N=64,
                CLUSTERED=cta64_bk512_cluster > 1,
                num_warps=8, waves_per_eu=2,
                num_ctas=cluster_m * cluster_n,
                llvm_fn_attrs=AGPR_ATTRS)
        if single_cta_skinny:
            shared_a, shared_b, shared_as, shared_bs, wmma = layouts
            return fp8_scaled_single_cta_512x128x128_kernel_gfx1250[grid](
                a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
                a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
                c_d.stride(0), c_d.stride(1), bs_d.stride(0),
                GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
                SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
                SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
                num_warps=8, waves_per_eu=2, num_ctas=1,
                llvm_fn_attrs=AGPR_ATTRS)
        if block_k == 512 and cta_tile == 128:
            shared_a, shared_b, shared_as, shared_bs, wmma = layouts
            return fp8_scaled_cluster_bk512_cta128_kernel_gfx1250[grid](
                a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
                a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
                c_d.stride(0), c_d.stride(1), bs_d.stride(0),
                GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
                SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
                SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
                BLOCK_M=block_m, BLOCK_N=block_n,
                CTA_M=cluster_m, CTA_N=cluster_n,
                num_warps=8, waves_per_eu=2,
                num_ctas=cluster_m * cluster_n,
                llvm_fn_attrs=AGPR_ATTRS)
        if experimental_bk256:
            (shared_a, shared_b, shared_as, shared_bs, wmma, load_wmma,
             output_cga) = layouts
            return fp8_scaled_cluster_bk256_opt_kernel_gfx1250[grid](
                a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
                a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
                c_d.stride(0), c_d.stride(1), bs_d.stride(0),
                GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
                SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
                SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
                LOAD_WMMA_LAYOUT=load_wmma, OUTPUT_CGA_LAYOUT=output_cga,
                BLOCK_M=block_m, BLOCK_N=block_n,
                CTA_M=cluster_m, CTA_N=cluster_n,
                num_warps=8, waves_per_eu=2,
                num_ctas=cluster_m * cluster_n,
                llvm_fn_attrs=AGPR_ATTRS)
        shared_a, shared_b, shared_as, shared_bs, wmma, output_cga = layouts
        if block_k == 128:
            kernel = fp8_scaled_cluster_bk128_three_slot_kernel_gfx1250
        else:
            kernel = fp8_scaled_cluster_bf16_style_kernel_gfx1250
        return kernel[grid](
            a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
            a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1), bs_d.stride(0),
            GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
            SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
            OUTPUT_CGA_LAYOUT=output_cga,
            BLOCK_M=block_m, BLOCK_N=block_n,
            CTA_M=cluster_m, CTA_N=cluster_n,
            num_warps=8, waves_per_eu=2,
            num_ctas=cluster_m * cluster_n,
            llvm_fn_attrs=AGPR_ATTRS,
            **({"REFILL_SCHEDULE": refill_schedule}
               if block_k == 128 else {}))

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            c_d.cpu(), c_ref, rtol=1e-2, atol=5e-1)
        print("result verified", flush=True)

    return launch, check


def make_mxfp8_bk128_case(args):
    return make_mxfp8_case(args, block_k=128)


def make_mxfp8_bk256_opt_case(args):
    return make_mxfp8_case(args, experimental_bk256=True)


def make_mxfp8_bk512_cta128_case(args):
    return make_mxfp8_case(args, block_k=512, cta_tile=128)


def make_mxfp8_512x128x128_cga1_case(args):
    return make_mxfp8_case(
        args, block_k=128, single_cta_skinny=True)


def make_mxfp8_64x64x512_cga1_case(args):
    return make_mxfp8_case(
        args, block_k=512, cta64_bk512_cluster=1)


def make_mxfp8_64x64x512_cga2x2_case(args):
    return make_mxfp8_case(
        args, block_k=512, cta64_bk512_cluster=2)


def make_mxfp8_64x64x512_cga4x4_case(args):
    return make_mxfp8_case(
        args, block_k=512, cta64_bk512_cluster=4)


def make_mxfp8_128x64x512_cga4x4_case(args):
    return make_mxfp8_case(
        args, block_k=512, cta128x64_bk512_cluster=4)


def make_mxfp8_64x128x512_cga4x4_case(args):
    return make_mxfp8_case(
        args, block_k=512, cta64x128_bk512_cluster=4)


def make_mxfp8_64x64x1024_cga2x2_case(args):
    return make_mxfp8_case(
        args, block_k=1024, cta64_bk1024_cluster=2)


def make_mxfp8_128x128x256_cga4x4_case(args):
    return make_mxfp8_case(
        args, block_k=256, raw_128_bk256=True)


def make_fp8_mxfp4_case(args):
    """Build the configurable E4M3 x packed-E2M1 benchmark case.

    Only B is block-scaled. The scaled-WMMA A-scale operand is ``None``, which
    represents a unit scale for the ordinary FP8 A matrix.
    """
    cluster_width = args.fp8_mxfp4_cluster_width
    block_m = 256 * cluster_width
    block_n = 256 * cluster_width
    if args.M % block_m or args.N % block_n or args.K % 256:
        raise ValueError(
            f"FP8xMXFP4 requires M/N divisible by {block_m} and K by 256")
    if args.K // 256 < args.fp8_mxfp4_num_buffers:
        raise ValueError(
            "FP8xMXFP4 requires at least one BK256 tile per buffer")
    if args.M % 128 or args.N % 128 or (args.K // 32) % 4:
        raise ValueError(
            "FP8xMXFP4 dimensions are incompatible with scale packing")
    torch.manual_seed(args.seed)
    mode = args.input_mode or "random"
    if mode == "trig":
        a = make_trig_tensor(
            (args.M, args.K), torch.float8_e4m3fn, False)
        b = MXFP4Tensor(size=(args.K, args.N))
        encoded = torch.empty(args.K * args.N, dtype=torch.uint8)
        chunk = 8 * 1024 * 1024
        for begin in range(0, encoded.numel(), chunk):
            end = min(begin + chunk, encoded.numel())
            angle = torch.arange(begin, end, dtype=torch.float64)
            encoded[begin:end].copy_(
                MXFP4Tensor(data=torch.cos(angle).float()).data)
        b.data = encoded.view(args.K, args.N)
    else:
        a = init_data("float8_e4m3", args.M, args.K)
        b = init_data("float4", args.K, args.N)

    scale_k = args.K // 32
    b_scale_obj = MXScaleTensor(
        size=(args.N, scale_k)).random(low=1.0, high=32.0)
    c_ref = None
    if args.check:
        a_scale_obj = MXScaleTensor(data=torch.ones(
            (args.M, scale_k), dtype=torch.float32))
        c_ref = torch_gemm_mxfp(
            a, b, a_scale_obj, b_scale_obj, 32,
            args.M, args.N, args.K).to(torch.bfloat16)

    a_d = a.contiguous().cuda()
    b_d = b.to_packed_tensor(dim=0).data.T.contiguous().cuda()
    bs_d = pack_scale(b_scale_obj.data, 4).cuda()
    c_d = torch.empty(
        (args.M, args.N), dtype=torch.bfloat16, device="cuda")
    layouts = build_fp8_mxfp4_layouts(cluster_width)
    grid = (
        triton.cdiv(args.M, block_m) * triton.cdiv(args.N, block_n), 1)
    group_size = args.fp8_mxfp4_group_size
    if group_size is None:
        group_size = 8 if args.K <= 8192 else 4

    def launch():
        shared_a, shared_b, shared_bs, wmma = layouts
        return fp8_mxfp4_cluster_gemm_gfx1250[grid](
            a_d, b_d, c_d, bs_d, args.M, args.N, args.K,
            a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1), bs_d.stride(0),
            GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, SHARED_SCALE_B=shared_bs,
            WMMA_LAYOUT=wmma,
            PIPELINE_TAIL=True, SINGLE_SLOT_PROLOGUE=True,
            NUM_BUFFERS=args.fp8_mxfp4_num_buffers,
            BLOCK_M=block_m, BLOCK_N=block_n,
            CTA_M=cluster_width, CTA_N=cluster_width,
            REUSE_A=args.fp8_mxfp4_reuse_order == "a",
            GROUP_SIZE=group_size,
            num_warps=8, waves_per_eu=2,
            num_ctas=cluster_width * cluster_width,
            llvm_fn_attrs=AGPR_ATTRS)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            c_d.cpu(), c_ref, rtol=1e-2, atol=5e-1)
        print("result verified", flush=True)

    return launch, check


def make_mxfp4_case(args):
    """Build the fixed packed-E2M1/E8M0 block-32 4x4-cluster case."""
    if args.M % 1024 or args.N % 1024 or args.K % 512:
        raise ValueError("MXFP4 requires M/N divisible by 1024 and K by 512")
    if args.K // 512 < 2 or (args.K // 512) % 2:
        raise ValueError(
            "MXFP4 requires an even number of at least two BK512 tiles")
    if args.M % 128 or args.N % 128 or (args.K // 32) % 4:
        raise ValueError("MXFP4 dimensions are incompatible with scale packing")
    torch.manual_seed(args.seed)
    mode = args.input_mode or "random"
    if mode == "trig":
        def make_mxfp4_trig(shape, cosine):
            result = MXFP4Tensor(size=shape)
            encoded = torch.empty(shape[0] * shape[1], dtype=torch.uint8)
            chunk = 8 * 1024 * 1024
            for begin in range(0, encoded.numel(), chunk):
                end = min(begin + chunk, encoded.numel())
                angle = torch.arange(begin, end, dtype=torch.float64)
                values = torch.cos(angle) if cosine else torch.sin(angle)
                encoded[begin:end].copy_(
                    MXFP4Tensor(data=values.float()).data)
            result.data = encoded.view(shape)
            return result

        a = make_mxfp4_trig((args.M, args.K), False)
        b = make_mxfp4_trig((args.K, args.N), True)
    else:
        a = init_data("float4", args.M, args.K)
        b = init_data("float4", args.K, args.N)
    scale_k = args.K // 32
    a_scale_obj = MXScaleTensor(
        size=(args.M, scale_k)).random(low=1.0, high=32.0)
    b_scale_obj = MXScaleTensor(
        size=(args.N, scale_k)).random(low=1.0, high=32.0)
    c_ref = None
    if args.check:
        c_ref = torch_gemm_mxfp(
            a, b, a_scale_obj, b_scale_obj, 32,
            args.M, args.N, args.K).to(torch.bfloat16)
    as_d = pack_scale(a_scale_obj.data, 4).cuda()
    bs_d = pack_scale(b_scale_obj.data, 4).cuda()
    a_d = a.to_packed_tensor(dim=1).data.contiguous().cuda()
    b_d = b.to_packed_tensor(dim=0).data.T.contiguous().cuda()
    c_d = torch.empty(
        (args.M, args.N), dtype=torch.bfloat16, device="cuda")
    layouts = build_mxfp4_layouts()
    grid = (triton.cdiv(args.M, 1024) * triton.cdiv(args.N, 1024), 1)

    def launch():
        shared_a, shared_b, shared_as, shared_bs, wmma = layouts
        return mxfp4_bk512_warp_pipeline_gfx1250[grid](
            a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
            a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1), bs_d.stride(0),
            GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
            SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
            num_warps=8, waves_per_eu=2, num_ctas=16,
            llvm_fn_attrs=AGPR_ATTRS)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            c_d.cpu(), c_ref.cpu(), rtol=1e-2, atol=5e-1)
        print("result verified", flush=True)

    return launch, check


def event_probe(fn, iters):
    """Measure a short per-launch GPU-event latency used to size the graph."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def capture_graph(fn, n_per_graph):
    """Capture ``n_per_graph`` launches after warming a side stream."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
        side.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            for _ in range(n_per_graph):
                fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    return graph


def run_benchmark(launch, M, N, K, args):
    """Time graph replays and report nominal GEMM throughput.

    Throughput uses ``2*M*N*K`` operations for every format. Packing and scale
    preparation happen before timing; only the GEMM launch is captured.
    """
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    probe_ms = event_probe(launch, args.probe_iters)
    n_per_graph = (
        max(1, int(args.graph_ms / max(probe_ms, 1e-6)))
        if args.iters_per_graph is None else args.iters_per_graph)
    if n_per_graph <= 0:
        raise ValueError("--iters-per-graph must be positive")
    graph = capture_graph(launch, n_per_graph)
    total_iters = args.replays * n_per_graph
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.replays):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    per_iter = elapsed / total_iters
    print(f"probe per-iter   : {probe_ms * 1e3:.2f} us")
    print(f"iters per graph  : {n_per_graph}")
    print(f"replays          : {args.replays}")
    print(f"total iters      : {total_iters}")
    print(f"total elapsed    : {elapsed:.6f} s")
    print(f"per-iter         : {per_iter * 1e6:.2f} us")
    print(f"TFLOPS           : {(2 * M * N * K) / per_iter / 1e12:.3f}")


def run_one(name, args):
    """Create, optionally verify, benchmark, and release one selected case."""
    print(f"\n=== {name.upper()} ===", flush=True)
    makers = {
        "bf16": make_bf16_case,
        "bf16_8stage": make_bf16_8stage_case,
        "mxfp8": make_mxfp8_case,
        "mxfp8_bk128": make_mxfp8_bk128_case,
        "mxfp8_bk256_opt": make_mxfp8_bk256_opt_case,
        "mxfp8_bk512_128": make_mxfp8_bk512_cta128_case,
        "mxfp8_512x128x128_cga1": make_mxfp8_512x128x128_cga1_case,
        "mxfp8_64x64x512_cga1": make_mxfp8_64x64x512_cga1_case,
        "mxfp8_64x64x512_cga2x2": make_mxfp8_64x64x512_cga2x2_case,
        "mxfp8_64x64x512_cga4x4": make_mxfp8_64x64x512_cga4x4_case,
        "mxfp8_128x64x512_cga4x4": make_mxfp8_128x64x512_cga4x4_case,
        "mxfp8_64x128x512_cga4x4": make_mxfp8_64x128x512_cga4x4_case,
        "mxfp8_64x64x1024_cga2x2": make_mxfp8_64x64x1024_cga2x2_case,
        "mxfp8_128x128x256_cga4x4": make_mxfp8_128x128x256_cga4x4_case,
        "fp8_mxfp4": make_fp8_mxfp4_case,
        "mxfp4": make_mxfp4_case,
    }
    launch, check = makers[name](args)
    if args.check:
        check()
    if args.benchmark:
        run_benchmark(launch, args.M, args.N, args.K, args)
    del launch, check
    gc.collect()
    torch.cuda.empty_cache()


def parse_args():
    """Parse the standalone correctness/benchmark command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run the four canonical performance-retained GFX1250 GEMMs")
    parser.add_argument(
        "--kernel",
        choices=(*KERNEL_NAMES, *EXPERIMENTAL_KERNEL_NAMES, "all"),
        default="all")
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-K", type=int, default=65536)
    parser.add_argument(
        "--input-mode", choices=("random", "trig"),
        help=("input value pattern; defaults to trig for BF16/MXFP8 and random "
              "for FP8xMXFP4/MXFP4"))
    parser.add_argument(
        "--bf16-fp32-output", dest="bf16_output", action="store_false",
        help="Store BF16 kernel results as FP32 instead of the BF16 default")
    parser.set_defaults(bf16_output=True)
    parser.add_argument(
        "--mxfp8-cluster-width", type=int, choices=(2, 4), default=4,
        help="square MXFP8 CTA-cluster width")
    parser.add_argument(
        "--mxfp8-cluster-shape", type=int, nargs=2, choices=(2, 4),
        metavar=("CTA_M", "CTA_N"),
        help="rectangular MXFP8 CTA-cluster shape; overrides cluster width")
    parser.add_argument(
        "--mxfp8-bk128-schedule",
        choices=("refill-load", "post-wmma", "arrive-load-wait-refill"),
        default="refill-load",
        help="experimental three-slot BK128 refill placement")
    parser.add_argument(
        "--fp8-mxfp4-num-buffers", type=int, choices=(2, 3), default=3,
        help="number of FP8xMXFP4 TDM ring slots")
    parser.add_argument(
        "--fp8-mxfp4-cluster-width", type=int, choices=(1, 2, 4), default=4,
        help="square FP8xMXFP4 CTA-cluster width")
    parser.add_argument(
        "--fp8-mxfp4-reuse-order", choices=("a", "b"), default="a",
        help="group output tiles to reuse A or B between cluster waves")
    parser.add_argument(
        "--fp8-mxfp4-group-size", type=int, choices=(1, 2, 4, 8),
        help="same-operand PID group size (default: 8 for K<=8192, else 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--probe-iters", type=int, default=20)
    parser.add_argument("--graph-ms", type=float, default=100.0)
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--iters-per-graph", type=int, default=50)
    args = parser.parse_args()
    if not args.check and not args.benchmark:
        parser.error("select --check and/or --benchmark")
    if args.probe_iters <= 0 or args.replays <= 0 or args.warmup < 0:
        parser.error("benchmark iteration counts must be positive")
    return args


def main():
    args = parse_args()
    names = KERNEL_NAMES if args.kernel == "all" else (args.kernel,)
    for name in names:
        run_one(name, args)


if __name__ == "__main__":
    main()
