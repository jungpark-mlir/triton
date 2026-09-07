"""Verified GFX1250 Gluon GEMMs: one proven schedule per data type.

This module is a review-oriented extraction of ``kernels.py``. It carries
exactly one kernel per data type -- the fastest schedule that is also proven
free of run-to-run nondeterminism -- with the variant flags, dead branches, and
single-use device helpers removed so each kernel reads top to bottom.

    bf16       BF16 operands, FP32 accumulate, BF16 output
    mxfp8      E4M3 operands, block-32 E8M0 scales on both sides
    fp8_mxfp4  unscaled E4M3 A, packed E2M1 B with block-32 E8M0 scales
    mxfp4      packed E2M1 operands, block-32 E8M0 scales on both sides

Every kernel launches eight warps over a 4x4 CTA cluster. A cluster computes one
1024x1024 output tile, so each CTA owns a 256x256 partition. Operand tiles and
scales live in cluster-distributed LDS described by CGA-aware layouts, and TDM
moves global tiles into and out of that distributed storage.


Correctness protocol
--------------------
``--check`` compares against a CPU reference with ``atol=5e-1``, which is only
meaningful at small K; at K=65536 the accumulated magnitudes make it report
failures that are pure rounding. Data races were instead found by running the
same kernel repeatedly on identical inputs and comparing the outputs bitwise.
Inputs here are deterministic (fixed seed, trig fill) and no kernel splits K,
so every output element is produced by one CTA accumulating in a fixed order:
any run-to-run difference is a race. ``--determinism N`` runs that check.

Results at 4096x4096x65536 on gfx1250, and the per-iteration latency of each
schedule under ``gpu-lock``:

    bf16       150+ runs bit-identical     686 us   3204 TFLOP/s
    mxfp8       20+ runs bit-identical     269 us   8180 TFLOP/s
    fp8_mxfp4  see caveat below            226 us   9738 TFLOP/s
    mxfp4      150  runs bit-identical     145 us  15149 TFLOP/s

Caveat on fp8_mxfp4: this family shows a rare nondeterminism, roughly one run
in forty, disturbing about 170 of 16.7M output elements. It reproduces on the
schedule this file does not use (arrive at the end of the read stage) and was
not observed in 140 runs of the schedule below, but the rate is too low for
those 140 runs to prove the difference. Its cause is unidentified. Treat this
one kernel as unproven.


Two hardware rules this file encodes
------------------------------------
1. A TDM ``warp_used_hint`` must select only warps 0-3. Any hint reaching warps
   4-7 silently corrupts results: at 4096x4096x65536, hints touching warps 6-7
   perturbed 2-5% of the output with up to 79% relative error and varied run to
   run. The MLIR verifier accepts such hints, so nothing catches this at compile
   time -- it must be maintained by hand. This is what forbids fusing MXFP8's A
   and B copies; see the note in the MXFP8 refill.

2. ``tdm.async_wait(n)`` must be ``descriptors_per_tile * (slots - 1)``. The
   wait counts TDM operations, so a count below that retires part of the tile
   that was just issued and the loop blocks on a fresh global load every
   iteration. MXFP8 issues three descriptors per tile with two slots and wants
   3; waiting 2 there costs 102 us of the kernel's 269 us. FP8xMXFP4 issues two
   per tile with three slots and wants 4. Whenever the descriptor count in a
   refill changes, the matching wait must change with it.

Beyond those two rules, statement order around ``warp_pipeline_stage``, the
cluster barriers, TDM issue, and ``tdm.async_wait`` is a performance contract
rather than style. Before reordering any of it, re-check correctness,
determinism, register and scratch use, and locked benchmark medians.


Why the cluster barriers sit where they do
------------------------------------------
Waves 0-3 and 4-7 run the same tile offset by one pipeline stage, so waves 4-7
trail by a full stage. ``cluster.arrive()`` is signalled by warp 0 alone and
tells the peer CTAs that this CTA is done reading the slot they are about to
refill by multicast. If warp 0 signals while waves 4-7 are still reading that
slot, a peer's refill can land on live data. Each kernel therefore keeps the
arrive downstream of the workgroup barrier that ends the read stage, and every
refill downstream of the matching ``cluster.wait()``. The per-family comments
record the specific placement and what it cost to get there.
"""

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
# Per-tile K depth differs by format and each kernel spells its own out: 128 for
# BF16, 256 for MXFP8 and FP8xMXFP4, 512 logical (256 stored) for MXFP4. Packed
# E2M1 holds two values per byte, and scaled WMMA spans a different logical K
# than ordinary BF16 WMMA. Only the BF16 depth is needed outside its kernel,
# by the layout builder.
BF16_BLOCK_K = 128
NUM_WARPS = 8
CTA_M = 4
CTA_N = 4
# Every kernel here accumulates in VGPRs; denying AGPRs keeps the allocator from
# spilling the accumulator into the slower bank.
AGPR_ATTRS = (("amdgpu-agpr-alloc", "0,0"),)
KERNEL_NAMES = ("bf16", "mxfp8", "fp8_mxfp4", "mxfp4")

# Constants read from inside @gluon.jit must be gl.constexpr; plain Python
# globals are rejected by the frontend.
NUM_XCDS = gl.constexpr(8)
GROUP_SIZE_M = gl.constexpr(4)

# The only legal TDM warp hints (see rule 1 in the module docstring). Two-warp
# hints suit operands with two logical LDS pieces; the four-warp hint suits four
# pieces, or a single copy that no longer has to leave room for a fused partner.
TDM_WARPS_01 = gl.constexpr(0b00000011)
TDM_WARPS_23 = gl.constexpr(0b00001100)
TDM_WARPS_0123 = gl.constexpr(0b00001111)


# ---------------------------------------------------------------------------
# Shared grid mapping and output epilogues.
# ---------------------------------------------------------------------------
# Programs are spread across the eight XCDs first, then grouped along one output
# dimension so consecutive clusters reuse an operand. The default form groups
# along M and reuses B; the ``reuse_a`` form swaps the grouping axis and is what
# the mixed FP8xMXFP4 kernel uses.

@gluon.jit
def get_xcd_swizzled_pids(
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
def get_xcd_swizzled_pids_reuse_a(
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
def cluster_barrier():
    """Full cluster rendezvous: every CTA arrives, then every CTA waits.

    Used at pipeline boundaries where there is no read/refill overlap to
    exploit. Inside the steady-state loops the arrive and the wait are placed
    separately so useful work can sit between them.
    """
    gl.amd.gfx1250.cluster.arrive()
    gl.amd.gfx1250.cluster.wait()


@gluon.jit
def tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        STORE_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, CTA_N: gl.constexpr):
    """Convert the FP32 accumulator in LDS and store the tile in one TDM op.

    Eight elements of row padding avoid the bank conflicts an unpadded identity
    layout would produce.
    """
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


@gluon.jit
def mxfp8_tdm_store_serial_n2(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc):
    """Store the MXFP8 tile as two serial 128-column halves.

    MXFP8 partitions its accumulator 128 columns wide rather than 256. That
    halves the output staging tile and drops enough accumulator register
    pressure to remove the spill the 256-column form had (504 VGPRs and 68
    bytes/thread of scratch, against 470 and none).
    """
    cta_m: gl.constexpr = 256
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    cga_layout: gl.constexpr = (
        (1, 0, 0, 0), (2, 0, 0, 0),
        (0, 1, 0, 0), (0, 2, 0, 0))
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[half_n, 8]], [4, 4, cta_m, half_n],
        [3, 2, 1, 0], cga_layout)
    shared = gl.allocate_shared_memory(
        c_ptr.type.element_ty, [4, 4, cta_m, half_n], shared_layout)
    acc4 = acc.reshape((4, cta_m, 4, cta_n)).permute((0, 2, 1, 3))
    output0 = gl.amd.slice(
        acc4, [4, 4, cta_m, half_n], [0, 0, 0, 0])
    output1 = gl.amd.slice(
        acc4, [4, 4, cta_m, half_n], [0, 0, 0, half_n])
    desc = tdm.make_tensor_descriptor(
        base=c_ptr,
        shape=(M // cta_m, N // cta_n, cta_m, cta_n),
        strides=(cta_m * stride_cm, cta_n * stride_cn, stride_cm, stride_cn),
        block_shape=(4, 4, cta_m, half_n), layout=shared_layout)
    base_m = pid_m * 4
    base_n = pid_n * 4
    shared.store(output0.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, 0], shared)
    tdm.async_wait(0)
    shared.store(output1.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, half_n], shared)
    tdm.async_wait(0)


# ---------------------------------------------------------------------------
# BF16 x BF16.
# ---------------------------------------------------------------------------
# Aggregate tile: 1024x1024x128 across the cluster. Two BK128 LDS slots, each
# consumed as two K64 halves.
#
# This family uses a *deferred* refill: tile t refills the slot that tile t-1
# drained, not the slot t is reading. That is what makes the arrive safe for
# free. The alternative repair -- keeping the same-slot refill and moving the
# arrive next to the wait -- also removes the race but costs 1.9% (702 us
# against 689 us), because the arrive then lands inside the tile it is guarding.
# Deferring instead puts the arrive/wait pair at the very top of the tile, ahead
# of any read, with a full tile of WMMA between consecutive barriers. Measured
# at parity with the racy original, within 0.02%.

@gluon.jit
def bf16_deferred_consume_and_refill(
        a_buf, b_buf, slot, refill_slot, a_desc, b_desc, acc,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr, BLOCK_K: gl.constexpr,
        RING_WAIT: gl.constexpr):
    """Consume ``slot`` as two K64 WMMAs and refill ``refill_slot``.

    ``refill_slot`` was drained by the previous tile, so the write into it needs
    no in-tile guard against this tile's own reads.
    """
    half_k: gl.constexpr = BLOCK_K // 2
    # A wait op may not appear inside a warp_pipeline_stage region, so this is
    # the latest point it can sit: ahead of the issue below rather than between
    # the issue and the reads.
    #
    # RING_WAIT is 0 here, which is the optimum rather than a drained pipeline:
    # the deferred lead is one tile, so only one tile's refill is ever in
    # flight and the wait that collects it has nothing else to preserve. The
    # refill still gets a full tile of WMMA to land in. Reaching the usual
    # descriptors_per_tile * (slots - 1) depth of 2 would need the wait behind
    # the issue, which costs an extra stage region and measured 712 us against
    # 689, or a third slot, which needs 393,216 bytes of LDS against the 327,680
    # available.
    tdm.async_wait(RING_WAIT)
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        # The arrive sits directly ahead of the wait and ahead of every read in
        # this tile. Hoisting it into the previous tile would have warp 0 assert
        # drainage while waves 4-7, one stage behind, were still reading the
        # slot the peers then refill.
        gl.amd.gfx1250.cluster.arrive()
        gl.amd.gfx1250.cluster.wait()
        # Refill: two descriptors, A and B, each issued from warps 0-3. The
        # descriptors advance by one BK128 tile so the next call refills the
        # next tile of K.
        tdm.async_load(
            a_desc, [0, 0], a_buf.index(refill_slot),
            warp_used_hint=TDM_WARPS_0123)
        tdm.async_load(
            b_desc, [0, 0], b_buf.index(refill_slot),
            warp_used_hint=TDM_WARPS_0123)
        a_desc = tdm.update_tensor_descriptor(
            a_desc, add_offsets=[0, BLOCK_K])
        b_desc = tdm.update_tensor_descriptor(
            b_desc, add_offsets=[0, BLOCK_K])
        # First K64 of this tile's own slot.
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
        # The same-slot schedule split this second K64 into two K32 WMMAs only
        # so the refill could sit between them. The refill has moved to the top
        # of the tile, so the split has no purpose and collapses into one group.
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    return a_desc, b_desc, acc


@gluon.jit
def bf16_cluster_gemm_gfx1250(
        a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk,
        stride_bn, stride_cm, stride_cn, GRID_MN: gl.constexpr,
        SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr,
        WMMA_LAYOUT: gl.constexpr):
    block_m: gl.constexpr = 1024
    block_n: gl.constexpr = 1024
    block_k: gl.constexpr = 128
    half_k: gl.constexpr = block_k // 2
    gl.static_assert(
        a_ptr.type.element_ty.is_bf16() and b_ptr.type.element_ty.is_bf16())
    gl.static_assert(gl.num_ctas() == 16)
    pid_m, pid_n = get_xcd_swizzled_pids(
        M, N, block_m, block_n, GRID_MN, NUM_XCDS, GROUP_SIZE_M)
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
    acc = gl.zeros((block_m, block_n), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, block_k)
    gl.assume(iter_max >= 2)
    # The body is unrolled in pairs so ``slot`` and ``refill_slot`` are
    # compile-time constants, which requires an even tile count.
    gl.assume((iter_max % 2) == 0)
    ring_wait: gl.constexpr = 0

    # Deferred refill means only slot 0 is prefetched: tile t supplies tile
    # t + 1, so slot 1 is filled by tile 0 rather than by the prologue.
    tdm.async_load(
        a_desc, [0, 0], a_buf.index(0), warp_used_hint=TDM_WARPS_0123)
    tdm.async_load(
        b_desc, [0, 0], b_buf.index(0), warp_used_hint=TDM_WARPS_0123)
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, block_k])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, block_k])

    # Tiles 0 .. iter_max - 2 refill. The pair loop walks them two at a time and
    # the last refilling tile is peeled so the final tile skips its refill.
    for _ in range(0, (iter_max - 2) // 2):
        a_desc, b_desc, acc = bf16_deferred_consume_and_refill(
            a_buf, b_buf, 0, 1, a_desc, b_desc, acc, dot_a, dot_b, block_k,
            ring_wait)
        a_desc, b_desc, acc = bf16_deferred_consume_and_refill(
            a_buf, b_buf, 1, 0, a_desc, b_desc, acc, dot_a, dot_b, block_k,
            ring_wait)
    a_desc, b_desc, acc = bf16_deferred_consume_and_refill(
        a_buf, b_buf, 0, 1, a_desc, b_desc, acc, dot_a, dot_b, block_k,
        ring_wait)

    # Final tile: drain the ring, then consume slot 1 with no refill and no
    # cluster barrier, since no peer will touch this LDS again.
    tdm.async_wait(0)
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(1).slice(0, half_k, 1).load(layout=dot_a)
        b = b_buf.index(1).slice(
            0, half_k, 1).permute([1, 0]).load(layout=dot_b)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a = a_buf.index(1).slice(half_k, half_k, 1).load(layout=dot_a)
        b = b_buf.index(1).slice(
            half_k, half_k, 1).permute([1, 0]).load(layout=dot_b)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma(a, b, acc)

    tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        WMMA_LAYOUT, block_m, block_n, 4)


def build_bf16_layouts():
    """Build matching distributed LDS and accumulator layouts for BF16.

    Construction is deliberately two-pass. The first pass derives the local
    256x256-per-CTA WMMA ownership; the second grafts that layout's operand CGA
    bases onto the padded shared layouts. The result is that each CTA loads
    exactly the LDS partition its accumulator partition consumes, while the
    cluster still presents a single 1024x1024 view.
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
    # B is loaded in [N, K] order and transposed for WMMA, so its CGA bases are
    # mirrored relative to A's.
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[BF16_BLOCK_K, 8]], [BLOCK_M, BF16_BLOCK_K], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[BF16_BLOCK_K, 8]], [BLOCK_N, BF16_BLOCK_K], [1, 0], cga_b)
    # slice == block gives one group per physical partition, hence two logical
    # LDS pieces per operand.
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        slice_m, slice_n, padded_a, padded_b, NUM_WARPS, [16, 16, 32],
        a_transposed=False, b_transposed=True, slice_m=slice_m,
        slice_n=slice_n, transposed=True)
    return shared_a, shared_b, wmma


# ---------------------------------------------------------------------------
# MXFP8 x MXFP8.
# ---------------------------------------------------------------------------
# Aggregate tile: 1024x1024x256 across the cluster. E4M3 values with one E8M0
# scale per 32 logical K elements. Two BK256 slots, each consumed as two K128
# scaled WMMAs, with a same-slot refill: tile t reads slot t%2 and refills it
# with tile t+2.
#
# The steady-state order inside the second stage is fixed by measurement:
# arrive, wait, data refill, scale refill, and only then the final K128 WMMA.
#
# This family is bit-stable with the arrive in either position once the TDM
# hints stay inside warps 0-3, so the placement below is the conservative one
# rather than a fix for a demonstrated race: it keeps the arrive downstream of
# the workgroup barrier that ends the read stage. It costs about 1.4%.

# Descriptors issued per BK256 tile: A data, B data, and the fused scale pair.
# With two slots the steady-state ring keeps exactly one tile in flight, so this
# is also the main-loop wait count (rule 2 in the module docstring).
MXFP8_TDM_PER_TILE = gl.constexpr(3)


@gluon.jit
def mxfp8_load_scale(
        scale_buffer, slot, start_k: gl.constexpr, LAYOUT: gl.constexpr,
        BLOCK_NONK: gl.constexpr, BK_SCALE: gl.constexpr,
        SUBTILE_SCALE_K: gl.constexpr):
    """Undo the host-side factor-128 scale preshuffle for one K subtile.

    The reshape/permute pair is pure index arithmetic on the LDS view; it costs
    no instructions and exists so the scale fragment matches what
    ``get_wmma_scale_layout`` expects without a runtime transpose.
    """
    scale_slice = scale_buffer.index(slot).reshape(
        (BLOCK_NONK // 128, BK_SCALE // 4, 32, 4, 4)
    ).permute((0, 3, 2, 1, 4)).reshape((BLOCK_NONK, BK_SCALE))
    return scale_slice.slice(0, BLOCK_NONK, 0).slice(
        start_k, SUBTILE_SCALE_K, 1).load(layout=LAYOUT)


@gluon.jit
def mxfp8_issue_refill(
        a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
        tile_idx, slot):
    """Issue the three descriptors that refill one BK256 slot.

    A and B data must stay unfused, and every hint must stay inside warps 0-3.
    B's 128-column accumulator partition gives it four logical LDS pieces, so
    its hint has to select four warps; a fused pair additionally needs its
    members disjoint, which would force one member onto warps 4-7. Measured at
    4096x4096x65536: the fused pair (A=0b00000011, B=0b11001100) and the
    equivalent unfused pair both corrupt 2-5% of the output and vary run to run,
    while the four-warp hints below are bit-stable over 20 runs. The MLIR
    verifier accepts the warps-4-7 hints, so this is not enforced anywhere but
    here.

    Unfused, the two copies no longer need disjoint warps, so each takes all
    four leading warps. The cost is a third in-flight descriptor per tile
    alongside the fused scale pair, which MXFP8_TDM_PER_TILE accounts for.
    """
    tile_k = tile_idx * 256
    a_load = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, tile_k])
    b_load = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, tile_k])
    tdm.async_load(
        a_load, [0, 0], a_buf.index(slot), warp_used_hint=TDM_WARPS_0123)
    tdm.async_load(
        b_load, [0, 0], b_buf.index(slot), warp_used_hint=TDM_WARPS_0123)
    # The scales are small enough for two logical pieces each, so they do fuse,
    # with A on warps 0-1 and B on warps 2-3.
    scale_k = tile_idx * (256 // 32) * 128
    as_load = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, scale_k])
    bs_load = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, scale_k])
    tdm.async_load_fused([
        (as_load, as_buf.index(slot), TDM_WARPS_01),
        (bs_load, bs_buf.index(slot), TDM_WARPS_23),
    ])


@gluon.jit
def mxfp8_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, slot, a_desc, b_desc, as_desc,
        bs_desc, refill_idx, acc, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr):
    """Consume ``slot`` as two K128 scaled WMMAs, then refill it in place."""
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a0 = a_buf.index(slot).slice(0, 1024, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        as0 = mxfp8_load_scale(
            as_buf, slot, 0, SCALE_A_LAYOUT, 1024, 8, 4)
        b0 = b_buf.index(slot).slice(0, 1024, 0).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs0 = mxfp8_load_scale(
            bs_buf, slot, 0, SCALE_B_LAYOUT, 1024, 8, 4)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0, bs0, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
        a1 = a_buf.index(slot).slice(0, 1024, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        as1 = mxfp8_load_scale(
            as_buf, slot, 4, SCALE_A_LAYOUT, 1024, 8, 4)
        b1 = b_buf.index(slot).slice(0, 1024, 0).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs1 = mxfp8_load_scale(
            bs_buf, slot, 4, SCALE_B_LAYOUT, 1024, 8, 4)
    with gl.amd.warp_pipeline_stage("stage1", priority=1):
        # The arrive is here, not at the end of the read stage above, so that
        # the workgroup barrier closing that stage has already put all eight
        # waves past their reads of this slot before warp 0 releases the peers.
        gl.amd.gfx1250.cluster.arrive()
        gl.amd.gfx1250.cluster.wait()
        mxfp8_issue_refill(
            a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
            refill_idx, slot)
        # a1/b1 were read before the refill was issued; consuming them here
        # gives the refill's global latency something to hide behind.
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1, bs1, "e4m3", acc)
    return acc


@gluon.jit
def mxfp8_consume_tail_tile(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, DOT_A: gl.constexpr,
        DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr):
    """Consume one slot with no refill, keeping the warp pipeline annotated.

    The last two slots still want the stage split so the two wave groups stay
    offset while the TDM ring drains; the stage names differ from the main loop
    so the pipeliner treats them as a separate region.
    """
    with gl.amd.warp_pipeline_stage("stage0_tail", priority=0):
        a0 = a_buf.index(slot).slice(0, 1024, 0).slice(
            0, 128, 1).load(layout=DOT_A)
        as0 = mxfp8_load_scale(
            as_buf, slot, 0, SCALE_A_LAYOUT, 1024, 8, 4)
        b0 = b_buf.index(slot).slice(0, 1024, 0).slice(
            0, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs0 = mxfp8_load_scale(
            bs_buf, slot, 0, SCALE_B_LAYOUT, 1024, 8, 4)
    with gl.amd.warp_pipeline_stage("stage1_tail", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a0, as0, "e4m3", b0, bs0, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("stage0_tail", priority=0):
        a1 = a_buf.index(slot).slice(0, 1024, 0).slice(
            128, 128, 1).load(layout=DOT_A)
        as1 = mxfp8_load_scale(
            as_buf, slot, 4, SCALE_A_LAYOUT, 1024, 8, 4)
        b1 = b_buf.index(slot).slice(0, 1024, 0).slice(
            128, 128, 1).permute([1, 0]).load(layout=DOT_B)
        bs1 = mxfp8_load_scale(
            bs_buf, slot, 4, SCALE_B_LAYOUT, 1024, 8, 4)
    with gl.amd.warp_pipeline_stage("stage1_tail", priority=1):
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e4m3", b1, bs1, "e4m3", acc)
    return acc


@gluon.jit
def mxfp8_cluster_gemm_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr):
    gl.static_assert(gl.num_ctas() == 16)
    pid_m, pid_n = get_xcd_swizzled_pids(
        M, N, 1024, 1024, GRID_MN, NUM_XCDS, GROUP_SIZE_M)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [1024, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [1024, 4])
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, 1024, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, 1024, 256], SHARED_LAYOUT_B)
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty, [2, 8, 1024], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty, [2, 8, 1024], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * 1024 * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(1024, 256),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * 1024 * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(1024, 256),
        layout=SHARED_LAYOUT_B)
    # Scales are addressed as [non-K/128, K/32*128] because of the host-side
    # factor-128 preshuffle; mxfp8_load_scale undoes it on the LDS side.
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + (pid_m * 1024) // 128 * stride_scale,
        shape=(M // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(8, 1024), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * 1024) // 128 * stride_scale,
        shape=(N // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(8, 1024), layout=SHARED_SCALE_B)

    # Prefetch both slots, then wait for tile 0 only: the three descriptors of
    # tile 1 stay in flight. Waiting 2 here instead would drain into tile 1.
    for prefetch_idx in gl.static_range(2):
        mxfp8_issue_refill(
            a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
            prefetch_idx, prefetch_idx)
    tdm.async_wait(MXFP8_TDM_PER_TILE)
    cluster_barrier()

    acc = gl.zeros((1024, 1024), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 256)
    gl.assume(iter_max >= 2)
    for tile_idx in range(0, iter_max - 2):
        slot = tile_idx % 2
        acc = mxfp8_consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, slot, a_desc, b_desc, as_desc,
            bs_desc, tile_idx + 2, acc, dot_a, dot_b, scale_a_layout,
            scale_b_layout)
        # Three descriptors per tile with two slots: keep one tile in flight.
        # Dropping this to 2 forces every iteration to block on the A load it
        # just issued and costs 102 us at 4096x4096x65536.
        tdm.async_wait(MXFP8_TDM_PER_TILE)

    # Two tiles remain resident and need no refill.
    penultimate_slot = (iter_max - 2) % 2
    acc = mxfp8_consume_tail_tile(
        a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout)
    tdm.async_wait(0)
    cluster_barrier()
    last_slot = (iter_max - 1) % 2
    acc = mxfp8_consume_tail_tile(
        a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b,
        scale_a_layout, scale_b_layout)

    # The serial output stage aliases the cluster-distributed operand arena, so
    # every CTA must finish its final operand reads before any CTA repurposes
    # that LDS. The keep-alive calls stop the allocator from reusing the
    # buffers before this barrier.
    cluster_barrier()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    mxfp8_tdm_store_serial_n2(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc)


def build_mxfp8_layouts():
    """Build the 4x4-cluster data, scale, and WMMA layouts for MXFP8.

    Data operands use the K128 scaled-WMMA instruction shape. Scale layouts
    reuse the corresponding A/B CGA bases so that data and its block-32 scales
    are owned by the same CTA partition.

    ``slice_n=128`` against a 256-wide block is what gives B four logical LDS
    pieces (two partitions x two groups) instead of two. That is deliberate: the
    128-column accumulator slice matches the serial epilogue and removes the
    spill. It is also the reason B needs a four-warp TDM hint, and therefore the
    reason A and B cannot be fused -- see mxfp8_issue_refill.
    """
    cga_c = make_cga_layout([4, 4], [4, 4], [0, 1])
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        1024, 1024, local_a, local_b, 8, [16, 16, 128],
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
        [[256, 16]], [1024, 256], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [1024, 256], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
        256, 256, padded_a, padded_b, 8, [16, 16, 128],
        a_transposed=False, b_transposed=True, slice_m=256, slice_n=128)
    shared_as = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [8, 1024], [1, 0], cga_a)
    shared_bs = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [8, 1024], [1, 0], cga_b)
    return shared_a, shared_b, shared_as, shared_bs, wmma


# ---------------------------------------------------------------------------
# FP8 x MXFP4.
# ---------------------------------------------------------------------------
# A is ordinary E4M3, so its scaled-WMMA scale operand is None, which the
# instruction reads as a unit scale. B is packed E2M1 with block-32 E8M0 scales,
# so one BK256 logical tile occupies 256 A columns but only 128 packed B bytes.
#
# Three ring slots, a single-slot prologue so the first cluster barrier happens
# with one tile resident, a warp-pipelined drain, and A-reusing tile order.
#
# NOTE: this is the one family in this file whose freedom from races is not
# established. See the caveat in the module docstring.

FP8_MXFP4_TDM_PER_TILE = gl.constexpr(2)
FP8_MXFP4_SLOTS = gl.constexpr(3)


@gluon.jit
def fp8_mxfp4_load_scale(
        scale_buffer, slot, start_k: gl.constexpr, LAYOUT: gl.constexpr,
        BLOCK_N: gl.constexpr):
    """Undo the factor-128 scale preshuffle for one K subtile of B."""
    scale_slice = scale_buffer.index(slot).reshape(
        (BLOCK_N // 128, 2, 32, 4, 4)
    ).permute((0, 3, 2, 1, 4)).reshape((BLOCK_N, 8))
    return scale_slice.slice(0, BLOCK_N, 0).slice(
        start_k, 4, 1).load(layout=LAYOUT)


@gluon.jit
def fp8_mxfp4_issue_refill(
        a_desc, b_desc, bs_desc, a_buf, b_buf, bs_buf, tile_idx, slot):
    """Issue the two descriptors that refill one slot: fused A+B, then scales.

    Both operands have two logical LDS pieces here (slice == block in
    build_fp8_mxfp4_layouts), so two-warp hints are legal and A and B fuse into
    a single instruction with A on warps 0-1 and B on warps 2-3. Every hint
    stays inside warps 0-3.
    """
    a_load = tdm.update_tensor_descriptor(
        a_desc, add_offsets=[0, tile_idx * 256])
    # B advances half as fast as A because two E2M1 values share a byte.
    b_load = tdm.update_tensor_descriptor(
        b_desc, add_offsets=[0, tile_idx * 128])
    tdm.async_load_fused([
        (a_load, a_buf.index(slot), TDM_WARPS_01),
        (b_load, b_buf.index(slot), TDM_WARPS_23),
    ])
    scale_k = tile_idx * (256 // 32) * 128
    bs_load = tdm.update_tensor_descriptor(
        bs_desc, add_offsets=[0, scale_k])
    tdm.async_load(
        bs_load, [0, 0], bs_buf.index(slot), warp_used_hint=TDM_WARPS_23)


@gluon.jit
def fp8_mxfp4_consume_and_refill(
        a_buf, b_buf, bs_buf, slot, a_desc, b_desc, bs_desc, refill_idx,
        acc, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr):
    """Consume ``slot`` as two K128 scaled WMMAs, then refill it in place."""
    with gl.amd.warp_pipeline_stage("stage0", priority=0):
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
        # As in MXFP8, the arrive is kept past the workgroup barrier that ends
        # the read stage rather than at the end of that stage.
        gl.amd.gfx1250.cluster.arrive()
        gl.amd.gfx1250.cluster.wait()
        fp8_mxfp4_issue_refill(
            a_desc, b_desc, bs_desc, a_buf, b_buf, bs_buf, refill_idx, slot)
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, None, "e4m3", b1, bs1, "e2m1", acc)
    return acc


@gluon.jit
def fp8_mxfp4_consume_tail_tile(
        a_buf, b_buf, bs_buf, slot, acc, DOT_A: gl.constexpr,
        DOT_B: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    """Consume one slot with no refill, keeping the warp pipeline annotated."""
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
def fp8_mxfp4_cluster_gemm_gfx1250(
        a_ptr, b_ptr, c_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr,
        SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
        GROUP_SIZE: gl.constexpr):
    block_m: gl.constexpr = 1024
    block_n: gl.constexpr = 1024
    slots: gl.constexpr = FP8_MXFP4_SLOTS
    # Two descriptors per tile across three slots keeps two tiles in flight.
    ring_wait: gl.constexpr = FP8_MXFP4_TDM_PER_TILE * (slots - 1)
    gl.static_assert(gl.num_ctas() == 16)
    # Group along N so consecutive clusters reuse A, which is the larger
    # operand here: unpacked E4M3 against packed E2M1.
    pid_m, pid_n = get_xcd_swizzled_pids_reuse_a(
        M, N, block_m, block_n, GRID_MN, NUM_XCDS, GROUP_SIZE)
    # Only B's instruction K shape changes for the packed operand; output
    # ownership stays compatible with A's.
    wmma_packed: gl.constexpr = gl.amd.AMDWMMALayout(
        3, WMMA_LAYOUT.transposed, WMMA_LAYOUT.warp_bases,
        WMMA_LAYOUT.reg_bases, [16, 16, 64], WMMA_LAYOUT.cga_layout)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma_packed, 16)
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [block_n, 4])

    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [slots, block_m, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [slots, block_n, 128], SHARED_LAYOUT_B)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty,
        [slots, block_n // 128, 1024], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * block_m * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(block_m, 256),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * block_n * stride_bn, shape=(N, K // 2),
        strides=(stride_bn, stride_bk), block_shape=(block_n, 128),
        layout=SHARED_LAYOUT_B)
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + (pid_n * block_n) // 128 * stride_scale,
        shape=(N // 128, K // 32 * 128), strides=(stride_scale, 1),
        block_shape=(block_n // 128, 1024), layout=SHARED_SCALE_B)

    # Single-slot prologue: fill slot 0, drain it fully, and take the first
    # cluster barrier with just that tile resident. The remaining slots are then
    # issued behind the barrier so their latency overlaps the first tiles.
    fp8_mxfp4_issue_refill(
        a_desc, b_desc, bs_desc, a_buf, b_buf, bs_buf, 0, 0)
    tdm.async_wait(0)
    cluster_barrier()
    for prefetch_idx in gl.static_range(1, slots):
        fp8_mxfp4_issue_refill(
            a_desc, b_desc, bs_desc, a_buf, b_buf, bs_buf,
            prefetch_idx, prefetch_idx)

    acc = gl.zeros((block_m, block_n), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 256)
    gl.assume(iter_max >= slots)
    for tile_idx in range(0, iter_max - slots):
        slot = tile_idx % slots
        acc = fp8_mxfp4_consume_and_refill(
            a_buf, b_buf, bs_buf, slot, a_desc, b_desc, bs_desc,
            tile_idx + slots, acc, dot_a, dot_b, scale_b_layout,
            block_m, block_n)
        tdm.async_wait(ring_wait)

    # Drain the ring. Each tile releases one slot's worth of outstanding
    # descriptors, so the wait count steps down as the ring empties, and a
    # cluster barrier separates tiles because peers are still refilling.
    for drain_idx in gl.static_range(slots):
        slot = (iter_max - slots + drain_idx) % slots
        acc = fp8_mxfp4_consume_tail_tile(
            a_buf, b_buf, bs_buf, slot, acc, dot_a, dot_b,
            scale_b_layout, block_m, block_n)
        if drain_idx < slots - 1:
            tdm.async_wait(
                FP8_MXFP4_TDM_PER_TILE * (slots - 2 - drain_idx))
            cluster_barrier()
    # With three slots the drain loop's last iteration skips its barrier, so
    # the operand arena still needs one before the output stage aliases it.
    cluster_barrier()
    a_buf._keep_alive()
    b_buf._keep_alive()
    bs_buf._keep_alive()
    tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        WMMA_LAYOUT, block_m, block_n, 4)


def build_fp8_mxfp4_layouts():
    """Build mixed-format layouts for the 4x4 cluster.

    The E4M3 A tile has 256 stored K elements while packed E2M1 B has 128 stored
    bytes over the same logical K extent. ``slice_n=256`` equals the block
    width, so both operands get two logical LDS pieces and both can use two-warp
    TDM hints -- which is why this family fuses A and B where MXFP8 cannot.
    """
    block_m = 256 * CTA_M
    block_n = 256 * CTA_N
    cga = make_cga_layout([CTA_M, CTA_N], [CTA_M, CTA_N], [0, 1])
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
# ---------------------------------------------------------------------------
# Aggregate tile: 1024x1024x512 logical values across the cluster, which packed
# A/B storage reduces to 256 bytes of K. Two slots; each slot is consumed as two
# K256 groups, and each group is two K128 scaled WMMAs.
#
# This family needed no repair, and it is the reference for how the barriers
# should sit. Only the second K256 group overlaps the cluster handshake, and it
# does so around a WMMA: arrive, WMMA, wait. By the time warp 0 arrives, all
# eight waves are past the stage0 workgroup barrier and a WMMA has already
# consumed the stage0 reads. The refill is then issued from its own "tdm" stage
# region after the whole tile returns, so peer writes are strictly downstream of
# the wait.
#
# It also satisfies both hardware rules by construction: slice_n equals the
# block width, so both operands have two logical LDS pieces and fuse inside
# warps 0-3, and its two descriptors per tile match its wait of 2.

MXFP4_TDM_PER_TILE = gl.constexpr(2)


@gluon.jit
def mxfp4_load_scale(
        scale_buffer, slot, start_k: gl.constexpr, LAYOUT: gl.constexpr):
    """Undo the factor-128 scale preshuffle for one K subtile."""
    scale_slice = scale_buffer.index(slot).reshape(
        (8, 4, 32, 4, 4)).permute((0, 3, 2, 1, 4)).reshape((1024, 16))
    return scale_slice.slice(0, 1024, 0).slice(
        start_k, 4, 1).load(layout=LAYOUT)


@gluon.jit
def mxfp4_issue_refill(
        a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
        tile_idx, slot):
    """Issue the two descriptors that refill one BK512 slot.

    Both data and scales fuse, A on warps 0-1 and B on warps 2-3, all inside
    warps 0-3.
    """
    packed_k = tile_idx * 256
    a_load = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, packed_k])
    b_load = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, packed_k])
    tdm.async_load_fused([
        (a_load, a_buf.index(slot), TDM_WARPS_01),
        (b_load, b_buf.index(slot), TDM_WARPS_23),
    ])
    scale_k = tile_idx * 2048
    as_load = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, scale_k])
    bs_load = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, scale_k])
    tdm.async_load_fused([
        (as_load, as_buf.index(slot), TDM_WARPS_01),
        (bs_load, bs_buf.index(slot), TDM_WARPS_23),
    ])


@gluon.jit
def mxfp4_consume_k256(
        a_buf, b_buf, as_buf, bs_buf, slot, acc,
        START_K_PACKED: gl.constexpr, START_SCALE_K: gl.constexpr,
        DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        OVERLAP_CLUSTER: gl.constexpr):
    """Consume one K256 group as two K128 scaled WMMAs.

    With ``OVERLAP_CLUSTER`` the cluster handshake is folded around the first of
    those two WMMAs, so the arrive lands after a WMMA has consumed this group's
    reads and the wait is separated from the arrive by real work.
    """
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
        if OVERLAP_CLUSTER:
            gl.amd.gfx1250.cluster.arrive()
        acc = gl.amd.gfx1250.wmma_scaled(
            a1, as1, "e2m1", b1, bs1, "e2m1", acc)
        if OVERLAP_CLUSTER:
            gl.amd.gfx1250.cluster.wait()
    return acc


@gluon.jit
def mxfp4_cluster_gemm_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr):
    gl.static_assert(gl.num_ctas() == 16)
    pid_m, pid_n = get_xcd_swizzled_pids(
        M, N, 1024, 1024, GRID_MN, NUM_XCDS, GROUP_SIZE_M)
    # Both operands are packed, so both dot layouts use the packed K64 shape.
    wmma_packed: gl.constexpr = gl.amd.AMDWMMALayout(
        3, WMMA_LAYOUT.transposed, WMMA_LAYOUT.warp_bases,
        WMMA_LAYOUT.reg_bases, [32, 16, 64], WMMA_LAYOUT.cga_layout)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, wmma_packed, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, wmma_packed, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [1024, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [1024, 4])
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, 1024, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, 1024, 256], SHARED_LAYOUT_B)
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty, [2, 8, 2048], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty, [2, 8, 2048], SHARED_SCALE_B)
    # K // 2 because two E2M1 values share a stored byte.
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

    # Prefetch both slots and wait for tile 0, leaving tile 1's two descriptors
    # in flight.
    for prefetch_idx in gl.static_range(2):
        mxfp4_issue_refill(
            a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf,
            prefetch_idx, prefetch_idx)
    tdm.async_wait(MXFP4_TDM_PER_TILE)
    cluster_barrier()

    acc = gl.zeros((1024, 1024), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 512)
    gl.assume(iter_max >= 2)
    for tile_idx in range(0, iter_max - 2):
        slot = tile_idx % 2
        # First K256 group: no cluster handshake.
        acc = mxfp4_consume_k256(
            a_buf, b_buf, as_buf, bs_buf, slot, acc, 0, 0, dot_a, dot_b,
            scale_a_layout, scale_b_layout, False)
        # Second K256 group: arrive, WMMA, wait.
        acc = mxfp4_consume_k256(
            a_buf, b_buf, as_buf, bs_buf, slot, acc, 128, 8, dot_a, dot_b,
            scale_a_layout, scale_b_layout, True)
        # The refill sits in its own stage region, after every read of this
        # slot and after the wait above released the peers.
        with gl.amd.warp_pipeline_stage("tdm", priority=1):
            mxfp4_issue_refill(
                a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                bs_buf, tile_idx + 2, slot)
        tdm.async_wait(MXFP4_TDM_PER_TILE)

    # Two tiles remain resident; neither refills, so neither overlaps a
    # handshake.
    penultimate_slot = (iter_max - 2) % 2
    acc = mxfp4_consume_k256(
        a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, 0, 0,
        dot_a, dot_b, scale_a_layout, scale_b_layout, False)
    acc = mxfp4_consume_k256(
        a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, 128, 8,
        dot_a, dot_b, scale_a_layout, scale_b_layout, False)
    tdm.async_wait(0)
    cluster_barrier()
    last_slot = (iter_max - 1) % 2
    acc = mxfp4_consume_k256(
        a_buf, b_buf, as_buf, bs_buf, last_slot, acc, 0, 0,
        dot_a, dot_b, scale_a_layout, scale_b_layout, False)
    acc = mxfp4_consume_k256(
        a_buf, b_buf, as_buf, bs_buf, last_slot, acc, 128, 8,
        dot_a, dot_b, scale_a_layout, scale_b_layout, False)

    tdm_store_full_tile(
        c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
        WMMA_LAYOUT, 1024, 1024, 4)


def build_mxfp4_layouts():
    """Build packed-data and scale layouts for the BK512 MXFP4 kernel.

    Both operands use packed K64 fragments and block-32 scales, and their CGA
    bases are mirrored because B is loaded in [N, K] order and transposed for
    WMMA.
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
# ---------------------------------------------------------------------------
# Storage conventions:
# * B is passed to kernels as an [N, K] contiguous transpose.
# * MXFP4 operands are packed along K before transfer to the GPU.
# * E8M0 scales are preshuffled in groups of 128 non-K values so LDS fragments
#   match get_wmma_scale_layout without a runtime transpose.
# * Every kernel accumulates in FP32 and writes BF16.
#
# Each case builder returns (launch, check, output) so --determinism can compare
# raw output tensors across runs.
#
# Benchmarking event-times a probe launch, captures repeated launches in a HIP
# graph, and reports the mean latency over all replays. gpu-lock only serializes
# access to the board; it does not pin clocks or disable DVFS, so compare
# medians of interleaved runs rather than absolute numbers across sessions.

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

    The factor-128 permutation groups the four scale lanes each WMMA scale
    fragment consumes. The returned 2D tensor is the global-memory format the
    scale TDM descriptors in this module expect.
    """
    non_k, k_scale = scale.shape
    scale = scale.view(
        non_k // 128, 4, 32, k_scale // scale_kwidth, scale_kwidth)
    scale = scale.permute(0, 3, 2, 1, 4).contiguous()
    return scale.view(non_k // 128, k_scale * 128)


def make_bf16_case(args):
    """Build BF16 launch/check closures while keeping their GPU tensors alive."""
    if args.M % 1024 or args.N % 1024 or args.K % 128:
        raise ValueError("BF16 requires M/N divisible by 1024 and K by 128")
    if args.K // 128 < 2 or (args.K // 128) % 2:
        raise ValueError(
            "BF16 requires an even number of at least two BK128 tiles")
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
    c = torch.zeros((args.M, args.N), dtype=torch.bfloat16, device="cuda")
    layouts = build_bf16_layouts()
    grid = (triton.cdiv(args.M, 1024) * triton.cdiv(args.N, 1024), 1)

    def launch():
        shared_a, shared_b, wmma = layouts
        return bf16_cluster_gemm_gfx1250[grid](
            a, b, c, args.M, args.N, args.K, a.stride(0), a.stride(1),
            b.stride(1), b.stride(0), c.stride(0), c.stride(1),
            GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, WMMA_LAYOUT=wmma,
            num_warps=NUM_WARPS, waves_per_eu=2, num_ctas=16)

    def check():
        c.zero_()
        launch()
        torch.cuda.synchronize()
        reference = (a.cpu().float() @ b.cpu().T.float()).to(torch.bfloat16)
        torch.testing.assert_close(
            c.cpu(), reference, rtol=1e-2, atol=1e-2)
        print("result verified", flush=True)

    return launch, check, c


def make_mxfp8_case(args):
    """Build the E4M3/E8M0 block-32 4x4-cluster case."""
    if args.M % 1024 or args.N % 1024 or args.K % 256:
        raise ValueError("MXFP8 requires M/N divisible by 1024 and K by 256")
    if args.K // 256 < 2:
        raise ValueError("MXFP8 requires at least two BK256 tiles")
    if args.M % 128 or args.N % 128 or (args.K // 32) % 4:
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
    a_scale = pack_scale(a_scale_obj.data, 4)
    b_scale = pack_scale(b_scale_obj.data, 4)
    a_d = a.contiguous().cuda()
    b_d = b.T.contiguous().cuda()
    as_d = a_scale.cuda()
    bs_d = b_scale.cuda()
    c_d = torch.empty(
        (args.M, args.N), dtype=torch.bfloat16, device="cuda")
    layouts = build_mxfp8_layouts()
    grid = (triton.cdiv(args.M, 1024) * triton.cdiv(args.N, 1024), 1)

    def launch():
        shared_a, shared_b, shared_as, shared_bs, wmma = layouts
        return mxfp8_cluster_gemm_gfx1250[grid](
            a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
            a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1), bs_d.stride(0),
            GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
            SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
            num_warps=NUM_WARPS, waves_per_eu=2, num_ctas=16,
            llvm_fn_attrs=AGPR_ATTRS)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            c_d.cpu(), c_ref, rtol=1e-2, atol=5e-1)
        print("result verified", flush=True)

    return launch, check, c_d


def make_fp8_mxfp4_case(args):
    """Build the E4M3 x packed-E2M1 case.

    Only B is block-scaled; the scaled-WMMA A-scale operand is None, which
    represents a unit scale for the ordinary FP8 A matrix.
    """
    block_m = 256 * CTA_M
    block_n = 256 * CTA_N
    if args.M % block_m or args.N % block_n or args.K % 256:
        raise ValueError(
            f"FP8xMXFP4 requires M/N divisible by {block_m} and K by 256")
    if args.K // 256 < int(FP8_MXFP4_SLOTS.value):
        raise ValueError(
            "FP8xMXFP4 requires at least one BK256 tile per ring slot")
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
    layouts = build_fp8_mxfp4_layouts()
    grid = (
        triton.cdiv(args.M, block_m) * triton.cdiv(args.N, block_n), 1)
    # Wider PID groups help while the working set still fits; past K=8192 the
    # reuse window is too large to hold and 4 measures better.
    group_size = 8 if args.K <= 8192 else 4

    def launch():
        shared_a, shared_b, shared_bs, wmma = layouts
        return fp8_mxfp4_cluster_gemm_gfx1250[grid](
            a_d, b_d, c_d, bs_d, args.M, args.N, args.K,
            a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1), bs_d.stride(0),
            GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, SHARED_SCALE_B=shared_bs,
            WMMA_LAYOUT=wmma, GROUP_SIZE=group_size,
            num_warps=NUM_WARPS, waves_per_eu=2, num_ctas=16,
            llvm_fn_attrs=AGPR_ATTRS)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            c_d.cpu(), c_ref, rtol=1e-2, atol=5e-1)
        print("result verified", flush=True)

    return launch, check, c_d


def make_mxfp4_case(args):
    """Build the packed-E2M1/E8M0 block-32 4x4-cluster case."""
    if args.M % 1024 or args.N % 1024 or args.K % 512:
        raise ValueError("MXFP4 requires M/N divisible by 1024 and K by 512")
    if args.K // 512 < 2:
        raise ValueError("MXFP4 requires at least two BK512 tiles")
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
        return mxfp4_cluster_gemm_gfx1250[grid](
            a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
            a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1), bs_d.stride(0),
            GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
            SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
            num_warps=NUM_WARPS, waves_per_eu=2, num_ctas=16,
            llvm_fn_attrs=AGPR_ATTRS)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            c_d.cpu(), c_ref.cpu(), rtol=1e-2, atol=5e-1)
        print("result verified", flush=True)

    return launch, check, c_d


def run_determinism(launch, out, reps):
    """Detect data races by re-running one kernel on identical inputs.

    Inputs are fixed by seed and no kernel splits K, so a correct kernel is
    bit-exact reproducible. Any difference between runs is nondeterminism in the
    kernel, which is a far sharper signal than the reference comparison in
    ``check`` -- that one drowns in accumulation error once K is large.
    """
    base = None
    identical = 0
    worst = 0.0
    differing = 0
    for _ in range(reps):
        out.zero_()
        launch()
        torch.cuda.synchronize()
        current = out.clone()
        if base is None:
            base = current
            identical += 1
            continue
        if torch.equal(base, current):
            identical += 1
        else:
            worst = max(
                worst, (base.float() - current.float()).abs().max().item())
            differing = max(differing, int((base != current).sum().item()))
    print(f"determinism      : {identical}/{reps} runs bit-identical")
    if identical != reps:
        scale = base.float().abs().max().item()
        print(f"worst abs diff   : {worst:.4g} (max |c| {scale:.4g}, "
              f"relative {worst / max(scale, 1e-30):.3g})")
        print(f"differing elems  : {differing} / {base.numel()} "
              f"({100.0 * differing / base.numel():.4f}%)")


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
        "mxfp8": make_mxfp8_case,
        "fp8_mxfp4": make_fp8_mxfp4_case,
        "mxfp4": make_mxfp4_case,
    }
    launch, check, out = makers[name](args)
    if args.check:
        check()
    if args.determinism:
        run_determinism(launch, out, args.determinism)
    if args.benchmark:
        run_benchmark(launch, args.M, args.N, args.K, args)
    del launch, check, out
    gc.collect()
    torch.cuda.empty_cache()


def parse_args():
    """Parse the standalone correctness/determinism/benchmark interface."""
    parser = argparse.ArgumentParser(
        description="Run the four verified GFX1250 cluster GEMMs")
    parser.add_argument(
        "--kernel", choices=(*KERNEL_NAMES, "all"), default="all")
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-K", type=int, default=65536)
    parser.add_argument(
        "--input-mode", choices=("random", "trig"),
        help=("input value pattern; defaults to trig for BF16/MXFP8 and random "
              "for FP8xMXFP4/MXFP4"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check", action="store_true",
        help=("compare against a CPU reference; only meaningful at small K, "
              "since atol=5e-1 is swamped by accumulation error by K=65536"))
    parser.add_argument(
        "--determinism", type=int, metavar="N", default=0,
        help=("run the kernel N times on identical inputs and report how many "
              "outputs are bit-identical; this is the race check, and it is "
              "valid at any K"))
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--probe-iters", type=int, default=20)
    parser.add_argument("--graph-ms", type=float, default=100.0)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--iters-per-graph", type=int)
    args = parser.parse_args()
    if not args.check and not args.benchmark and not args.determinism:
        parser.error("select --check, --determinism, and/or --benchmark")
    if args.probe_iters <= 0 or args.replays <= 0 or args.warmup < 0:
        parser.error("benchmark iteration counts must be positive")
    if args.determinism < 0:
        parser.error("--determinism must be non-negative")
    return args


def main():
    args = parse_args()
    names = KERNEL_NAMES if args.kernel == "all" else (args.kernel,)
    for name in names:
        run_one(name, args)


if __name__ == "__main__":
    main()
