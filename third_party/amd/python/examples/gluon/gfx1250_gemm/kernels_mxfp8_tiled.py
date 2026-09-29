"""Tile- and cluster-configurable MXFP8 cluster GEMM for GFX1250.

This module carries a single MXFP8 kernel derived from the promoted baseline,
``fp8_scaled_cluster_bf16_style_kernel_gfx1250`` in ``kernels_deferred.py``,
generalized over two axes that were previously fixed by kernel identity:

* CTA output tile and K depth: ``64x64x512``, ``128x128x256``,
  ``128x128x512``, ``256x256x256``.
* CTA cluster shape: ``2x2`` or ``4x4``.

At ``256x256x256`` with a ``4x4`` cluster the schedule, layouts, and epilogue
reduce to the promoted baseline, so that configuration is the reference point
rather than a new variant. ``kernels_deferred.py`` reaches the other tile
shapes through separate kernels with separate schedules; here they are one
kernel with constexpr parameters.

Why the tile size matters. The baseline was tuned at M=N=4096, K=65536, where a
4x4 cluster of 256x256 tiles fills the machine: 16 cluster tiles of 1024x1024,
256 CTAs. The cluster tile is also the launch granularity, so that
configuration requires M and N divisible by 1024 and cannot run any of the
skinny-M shapes below at all. Shrinking the CTA tile trades operand reuse for
parallelism, and on those shapes the trade is strongly favorable. Best
configuration against the largest tile that does run, two rounds each:

    512x2048x7168   64x64x512  4x4    8.80 vs 20.34 us   2.3x
    512x7168x3072  128x128x512 4x4    6.32 vs 10.97 us   1.7x
    512x8192x1536  128x128x512 4x4    4.41 vs  7.14 us   1.6x

The right column is 256x256x256 on a 2x2 cluster. Tile choice dominates
cluster choice, which moves these shapes by a few percent, and the best tile is
shape-dependent: 64x64x512 wins the K-heavy shape and loses the other two by
2.2-2.8x. Expect to sweep rather than to predict.

Schedule. The default two-slot path uses the baseline's race-free schedule:
one prefetched slot, then a deferred refill that targets the slot drained by
the *previous* tile, issued behind an adjacent cluster arrive/wait pair at the
top of the tile's first read stage, with ``tdm.async_wait`` placed *after* the
issue so both the in-flight and the about-to-be-read tile stay outstanding.

``--buffers 3`` instead prefetches two tiles. Each iteration retires the older
tile while leaving the next tile in flight, then refills the third, disjoint
slot before consuming the retired tile. Full-BLOCK_K triple buffering fits in
the 320-KiB LDS limit for ``64x64x512`` and ``128x128x256``. The larger
BLOCK_K configurations require 418-430 KiB and are rejected by the host before
launch. On 64x64x512, triple buffering improves the K-heavy workloads but
regresses the shortest-K one:

    layout       shape               two slots   three slots
    plain        512x2048x7168          9.23 us       8.56 us
    plain        512x7168x3072         17.95 us      16.73 us
    plain        512x8192x1536          9.84 us      10.27 us
    partitioned  512x2048x7168          8.83 us       7.97 us
    partitioned  512x7168x3072         17.97 us      16.38 us
    partitioned  512x8192x1536          9.49 us       9.78 us

For the 128x128 CTA tile, reducing BLOCK_K from 512 to 256 makes three slots
fit but does not beat the original 512-K double-buffer configuration:

    shape              BK256 two   BK256 three   BK512 two
    512x7168x3072          7.92 us       7.82 us      7.39 us
    512x8192x1536          5.09 us       5.09 us      4.95 us

See the ``kernels_deferred.py`` header for why the synchronization properties
are load-bearing. The default path deliberately does not reuse the 2-prefetch
same-slot schedule that ``kernels_deferred.py`` carries for its specialized
64x64x512 kernel.

Measured against those specialized kernels, two order-balanced rounds each:

    256x256x256 4x4  4096x4096x65536   274.0/276.1 vs 275.3/275.5 us
    128x128x512 4x4  512x7168x3072       6.39/6.41 vs   6.35/6.34 us
    128x128x512 4x4  512x8192x1536       4.42/4.41 vs   4.42/4.41 us
    64x64x512   4x4  512x2048x7168       8.82/8.81 vs   8.72/8.70 us

The first three are at parity; 64x64x512 pays about 1.3% for using the
baseline schedule instead of its specialized one.

Two axes are not free parameters, because the hardware layout forces them.

* Scale storage. Scales are preshuffled in groups of 128 along the non-K axis,
  which makes the scale tile ``[BLOCK/128, BLOCK_K/32*128]``. Distributing that
  across ``cluster_m`` CTAs needs ``BLOCK/128 >= cluster_m`` rows to hand out,
  which holds for a 128 or 256 CTA tile but not for 64, where the group of 128
  straddles two CTAs. A 64-wide tile therefore uses the unshuffled
  ``[BLOCK, BLOCK_K/32]`` form, and the host packs its scales differently.
  ``PRESHUFFLE_SCALES`` selects between them.

* Output epilogue. ``mxfp8_tiled_store_serial_n2`` halves the accumulator
  partition width and lets the output allocation reuse the dead operand arena,
  but it is written against a 256x256 CTA tile. Smaller tiles use the
  single-store ``mxfp8_tiled_store_full_tile`` instead, as their kernels in
  ``kernels_deferred.py`` do. ``SERIAL_N2_OUTPUT`` selects between them.

Both are derived from the tile shape in ``mxfp8_tiled_config``; neither is
exposed on the command line, because neither has a valid alternative setting.

Operand LDS form (``--shared-layout``). Operand A is split across the cluster
along M, because that is how the accumulator is split, and it is *also* LDS-
partitioned along M by ``make_partitioned_dot_layouts`` to avoid partition
conflicts on the WMMA operand reads. Those are necessarily the same axis, and
``PartitionedSharedLayout`` places its partition bit above the grafted CGA
bits in that dimension. A toolchain that derives ``shapePerCTA`` as
``shape / cgaSplit`` and requires every non-block basis to fit inside it will
reject the result: the partition bit sits outside the trimmed shape. Upstream
Triton does exactly this in ``maybeLinearToCGAEncodingAttr``, so every cluster
kernel here fails to lower there, in ``fillTDMDescriptor`` off ``async_load``.

``plain`` drops the partitioning pass and hands the CGA-grafted padded layouts
straight to TDM. That puts the CGA bits on top, which those toolchains accept,
at a cost that is real and not uniform:

    256x256x256 4x4  4096x4096x65536   273.1 vs 295.8 us    +8%
    128x128x512 4x4  512x7168x3072       6.34 vs  7.32 us   +15%
    64x64x512   4x4  512x2048x7168       8.81 vs  9.19 us    +4%

The same reformulation measures the same on both toolchains, so the cost is
the lost LDS partitioning rather than any codegen difference. ``partitioned``
is the default and is the configuration all other numbers here refer to; reach
for ``plain`` only when the target rejects the partitioned form.

See README.md for the locked benchmark protocol.
"""

import argparse
import gc
from pathlib import Path
import sys

import torch
import triton
from triton._C.libtriton.gluon_ir import make_cga_layout
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from third_party.amd.python.examples.gluon.mxfp_gemm_cdna5 import (  # noqa: E402
    MXScaleTensor,
    init_data,
    torch_gemm_mxfp,
)
from third_party.amd.python.examples.gluon.gfx1250_gemm.kernels_deferred import (  # noqa: E402
    make_trig_tensor,
    pack_scale,
    run_benchmark,
)


# (CTA_TILE_M, CTA_TILE_N, BLOCK_K) per selectable tile shape.
MXFP8_TILE_SHAPES = {
    "64x64x512": (64, 64, 512),
    "128x128x256": (128, 128, 256),
    "128x128x512": (128, 128, 512),
    "256x256x256": (256, 256, 256),
}
MXFP8_CLUSTER_SHAPES = {
    "2x2": (2, 2),
    "4x4": (4, 4),
}
# Operand LDS form. "partitioned" is faster everywhere and is the default;
# "plain" exists only to run on toolchains whose CGA factoring rejects the
# partitioned form. See "Operand LDS form" in the module docstring.
MXFP8_SHARED_LAYOUTS = ("partitioned", "plain")
MXFP8_BUFFER_COUNTS = (2, 3)
DEFAULT_TILE = "256x256x256"
DEFAULT_CLUSTER = "4x4"
DEFAULT_SHARED_LAYOUT = "partitioned"
DEFAULT_BUFFERS = 2

NUM_WARPS = 8
WAVES_PER_EU = 2
SCALE_BLOCK = 32
PRESHUFFLE_FACTOR = 128
# One tile issues three TDM descriptors: A data, B data, and the fused scale
# pair.
MXFP8_TDM_PER_TILE = gl.constexpr(3)
AGPR_ATTRS = (("amdgpu-agpr-alloc", "0,0"),)


def mxfp8_tiled_config(tile, cluster):
    """Resolve one tile/cluster selection into launch and layout parameters.

    ``preshuffle_scales`` and ``serial_n2_output`` are derived rather than
    selectable; see the module docstring for the constraints that fix them.
    """
    cta_tile_m, cta_tile_n, block_k = MXFP8_TILE_SHAPES[tile]
    cluster_m, cluster_n = MXFP8_CLUSTER_SHAPES[cluster]
    block_m = cta_tile_m * cluster_m
    block_n = cta_tile_n * cluster_n
    # A preshuffled scale tile has only BLOCK/128 rows to distribute, so it
    # needs at least one row per CTA along each axis.
    preshuffle_scales = (
        block_m // PRESHUFFLE_FACTOR >= cluster_m
        and block_n // PRESHUFFLE_FACTOR >= cluster_n)
    scale_step = block_k // SCALE_BLOCK
    if preshuffle_scales:
        scale_step *= PRESHUFFLE_FACTOR
    return {
        "cta_tile_m": cta_tile_m,
        "cta_tile_n": cta_tile_n,
        "block_k": block_k,
        "cluster_m": cluster_m,
        "cluster_n": cluster_n,
        "block_m": block_m,
        "block_n": block_n,
        "preshuffle_scales": preshuffle_scales,
        "scale_step": scale_step,
        "serial_n2_output": cta_tile_m == 256 and cta_tile_n == 256,
    }


# ---------------------------------------------------------------------------
# Grid mapping and output epilogues, unchanged from the baseline.

@gluon.jit
def mxfp8_tiled_get_pids(
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
def mxfp8_tiled_cluster_wait():
    gl.amd.gfx1250.cluster.arrive()
    gl.amd.gfx1250.cluster.wait()


@gluon.jit
def mxfp8_tiled_store_full_tile(
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


# The 256x256 tile keeps the baseline's half-width output stage. Two serial
# stores reduce accumulator partition width and let the output allocation reuse
# the dead operand arena.
@gluon.jit
def mxfp8_tiled_store_serial_n2(
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


# ---------------------------------------------------------------------------
# MXFP8 device functions, generalized over BLOCK_K and the scale storage form.

@gluon.jit
def mxfp8_tiled_issue_data(
        a_desc, b_desc, a_buf, b_buf, slot, BLOCK_K: gl.constexpr):
    tdm.async_load(
        a_desc, dest=a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(
        b_desc, dest=b_buf.index(slot), warp_used_hint=0b00001111)
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K])
    return a_desc, b_desc


@gluon.jit
def mxfp8_tiled_issue_scale(
        as_desc, bs_desc, as_buf, bs_buf, slot,
        SCALE_STEP: gl.constexpr):
    tdm.async_load_fused([
        (as_desc, as_buf.index(slot), 0b00000011),
        (bs_desc, bs_buf.index(slot), 0b00001100),
    ])
    as_desc = tdm.update_tensor_descriptor(
        as_desc, add_offsets=[0, SCALE_STEP])
    bs_desc = tdm.update_tensor_descriptor(
        bs_desc, add_offsets=[0, SCALE_STEP])
    return as_desc, bs_desc


@gluon.jit
def mxfp8_tiled_load_scale(
        scale_buf, slot, start_k: gl.constexpr, LAYOUT: gl.constexpr,
        BLOCK_NONK: gl.constexpr, BK_SCALE: gl.constexpr,
        PRESHUFFLE: gl.constexpr):
    """Load the four block-32 scale lanes one scaled-WMMA fragment consumes.

    Both forms are viewed as ``[non-K, K-block]`` before slicing. Keep the
    single return: a ``return`` inside a statically folded ``if`` does not stop
    the frontend from also tracing what follows it.
    """
    if PRESHUFFLE:
        # Undo the factor-128 host permutation to recover [non-K, K-block].
        view = scale_buf.index(slot).reshape(
            (BLOCK_NONK // 128, BK_SCALE // 4, 32, 4, 4)
        ).permute((0, 3, 2, 1, 4)).reshape((BLOCK_NONK, BK_SCALE))
    else:
        view = scale_buf.index(slot)
    return view.slice(0, BLOCK_NONK, 0).slice(
        start_k, 4, 1).load(layout=LAYOUT)


@gluon.jit
def mxfp8_tiled_consume(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, DOT_A: gl.constexpr,
        DOT_B: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
        SCALE_B_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
        PRESHUFFLE_SCALES: gl.constexpr):
    """Consume one slot as BLOCK_K / 128 warp-pipelined K128 WMMA groups."""
    bk_scale: gl.constexpr = BLOCK_K // 32
    for k_step in gl.static_range(BLOCK_K // 128):
        with gl.amd.warp_pipeline_stage("load", priority=0):
            a = a_buf.index(slot).slice(0, BLOCK_M, 0).slice(
                k_step * 128, 128, 1).load(layout=DOT_A)
            a_scale = mxfp8_tiled_load_scale(
                as_buf, slot, k_step * 4, SCALE_A_LAYOUT, BLOCK_M,
                bk_scale, PRESHUFFLE_SCALES)
            b = b_buf.index(slot).slice(0, BLOCK_N, 0).slice(
                k_step * 128, 128, 1).permute([1, 0]).load(layout=DOT_B)
            b_scale = mxfp8_tiled_load_scale(
                bs_buf, slot, k_step * 4, SCALE_B_LAYOUT, BLOCK_N,
                bk_scale, PRESHUFFLE_SCALES)
        with gl.amd.warp_pipeline_stage("compute", priority=1):
            acc = gl.amd.gfx1250.wmma_scaled(
                a, a_scale, "e4m3", b, b_scale, "e4m3", acc)
    return acc


@gluon.jit
def mxfp8_tiled_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, slot, refill_slot, a_desc, b_desc,
        as_desc, bs_desc, acc, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        BLOCK_K: gl.constexpr, SCALE_STEP: gl.constexpr,
        PRESHUFFLE_SCALES: gl.constexpr, RING_WAIT: gl.constexpr):
    with gl.amd.warp_pipeline_stage("refill", priority=0):
        # ``refill_slot`` was drained by the previous tile, including the
        # trailing wave group, so this write cannot overlap a live LDS read.
        gl.amd.gfx1250.cluster.arrive()
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc = mxfp8_tiled_issue_data(
            a_desc, b_desc, a_buf, b_buf, refill_slot, BLOCK_K)
        as_desc, bs_desc = mxfp8_tiled_issue_scale(
            as_desc, bs_desc, as_buf, bs_buf, refill_slot, SCALE_STEP)
    # After the issue, so this tile and the refill stay outstanding together.
    tdm.async_wait(RING_WAIT)
    acc = mxfp8_tiled_consume(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, DOT_A, DOT_B,
        SCALE_A_LAYOUT, SCALE_B_LAYOUT, BLOCK_M, BLOCK_N, BLOCK_K,
        PRESHUFFLE_SCALES)
    return a_desc, b_desc, as_desc, bs_desc, acc


@gluon.jit
def mxfp8_tiled_consume_and_refill_three(
        a_buf, b_buf, as_buf, bs_buf, slot, refill_slot, a_desc, b_desc,
        as_desc, bs_desc, acc, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
        SCALE_A_LAYOUT: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        BLOCK_K: gl.constexpr, SCALE_STEP: gl.constexpr,
        PRESHUFFLE_SCALES: gl.constexpr):
    # Two prefetched tiles are outstanding. Retire the older tile while
    # leaving the next tile in flight, then fill the third, disjoint slot.
    tdm.async_wait(3)
    with gl.amd.warp_pipeline_stage("refill_three", priority=0):
        gl.amd.gfx1250.cluster.arrive()
        gl.amd.gfx1250.cluster.wait()
        a_desc, b_desc = mxfp8_tiled_issue_data(
            a_desc, b_desc, a_buf, b_buf, refill_slot, BLOCK_K)
        as_desc, bs_desc = mxfp8_tiled_issue_scale(
            as_desc, bs_desc, as_buf, bs_buf, refill_slot, SCALE_STEP)
    acc = mxfp8_tiled_consume(
        a_buf, b_buf, as_buf, bs_buf, slot, acc, DOT_A, DOT_B,
        SCALE_A_LAYOUT, SCALE_B_LAYOUT, BLOCK_M, BLOCK_N, BLOCK_K,
        PRESHUFFLE_SCALES)
    return a_desc, b_desc, as_desc, bs_desc, acc


@gluon.jit
def mxfp8_tiled_cluster_kernel_gfx1250(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_scale, GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
        SHARED_LAYOUT_B: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
        SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
        OUTPUT_CGA_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, CTA_M: gl.constexpr,
        CTA_N: gl.constexpr, CTA_TILE_M: gl.constexpr,
        CTA_TILE_N: gl.constexpr, SCALE_STEP: gl.constexpr,
        PRESHUFFLE_SCALES: gl.constexpr, SERIAL_N2_OUTPUT: gl.constexpr,
        NUM_SLOTS: gl.constexpr):
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    gl.static_assert(BLOCK_M // CTA_M == CTA_TILE_M)
    gl.static_assert(BLOCK_N // CTA_N == CTA_TILE_N)
    gl.static_assert(BLOCK_K % 128 == 0)
    # Eight XCDs, output tiles grouped four deep along M to reuse B.
    pid_m, pid_n = mxfp8_tiled_get_pids(
        M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a, [BLOCK_M, 4])
    scale_b_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b, [BLOCK_N, 4])
    gl.static_assert(NUM_SLOTS == 2 or NUM_SLOTS == 3)
    slots: gl.constexpr = NUM_SLOTS
    ring_wait: gl.constexpr = MXFP8_TDM_PER_TILE * (slots - 1)
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [slots, BLOCK_M, BLOCK_K], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [slots, BLOCK_N, BLOCK_K], SHARED_LAYOUT_B)
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K),
        strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, BLOCK_K),
        layout=SHARED_LAYOUT_B)
    if PRESHUFFLE_SCALES:
        # Scales arrive permuted in groups of 128 along the non-K axis, so one
        # logical row of the scale tile covers 128 rows of the operand.
        as_buf = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty,
            [slots, BLOCK_M // 128, SCALE_STEP], SHARED_SCALE_A)
        bs_buf = gl.allocate_shared_memory(
            b_scale_ptr.type.element_ty,
            [slots, BLOCK_N // 128, SCALE_STEP], SHARED_SCALE_B)
        as_desc = tdm.make_tensor_descriptor(
            base=a_scale_ptr + (pid_m * BLOCK_M) // 128 * stride_scale,
            shape=(M // 128, K // 32 * 128),
            strides=(stride_scale, 1),
            block_shape=(BLOCK_M // 128, SCALE_STEP), layout=SHARED_SCALE_A)
        bs_desc = tdm.make_tensor_descriptor(
            base=b_scale_ptr + (pid_n * BLOCK_N) // 128 * stride_scale,
            shape=(N // 128, K // 32 * 128),
            strides=(stride_scale, 1),
            block_shape=(BLOCK_N // 128, SCALE_STEP), layout=SHARED_SCALE_B)
    else:
        as_buf = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty,
            [slots, BLOCK_M, SCALE_STEP], SHARED_SCALE_A)
        bs_buf = gl.allocate_shared_memory(
            b_scale_ptr.type.element_ty,
            [slots, BLOCK_N, SCALE_STEP], SHARED_SCALE_B)
        as_desc = tdm.make_tensor_descriptor(
            base=a_scale_ptr + pid_m * BLOCK_M * stride_scale,
            shape=(M, K // 32), strides=(stride_scale, 1),
            block_shape=(BLOCK_M, SCALE_STEP), layout=SHARED_SCALE_A)
        bs_desc = tdm.make_tensor_descriptor(
            base=b_scale_ptr + pid_n * BLOCK_N * stride_scale,
            shape=(N, K // 32), strides=(stride_scale, 1),
            block_shape=(BLOCK_N, SCALE_STEP), layout=SHARED_SCALE_B)
    # Double buffering starts with one tile; triple buffering starts with two.
    a_desc, b_desc = mxfp8_tiled_issue_data(
        a_desc, b_desc, a_buf, b_buf, 0, BLOCK_K)
    as_desc, bs_desc = mxfp8_tiled_issue_scale(
        as_desc, bs_desc, as_buf, bs_buf, 0, SCALE_STEP)
    if NUM_SLOTS == 3:
        a_desc, b_desc = mxfp8_tiled_issue_data(
            a_desc, b_desc, a_buf, b_buf, 1, BLOCK_K)
        as_desc, bs_desc = mxfp8_tiled_issue_scale(
            as_desc, bs_desc, as_buf, bs_buf, 1, SCALE_STEP)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, BLOCK_K)
    gl.assume(iter_max >= NUM_SLOTS)
    if NUM_SLOTS == 2:
        # Tile t refills the slot drained by tile t-1 with tile t+1.
        for tile_idx in range(0, iter_max - 1):
            slot = tile_idx % 2
            a_desc, b_desc, as_desc, bs_desc, acc = (
                mxfp8_tiled_consume_and_refill(
                    a_buf, b_buf, as_buf, bs_buf, slot, (tile_idx + 1) % 2,
                    a_desc, b_desc, as_desc, bs_desc, acc, dot_a, dot_b,
                    scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N, BLOCK_K,
                    SCALE_STEP, PRESHUFFLE_SCALES, ring_wait))
        tdm.async_wait(0)
        mxfp8_tiled_cluster_wait()
        last_slot = (iter_max - 1) % 2
        acc = mxfp8_tiled_consume(
            a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b,
            scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N, BLOCK_K,
            PRESHUFFLE_SCALES)
    else:
        for tile_idx in range(0, iter_max - 2):
            slot = tile_idx % 3
            a_desc, b_desc, as_desc, bs_desc, acc = (
                mxfp8_tiled_consume_and_refill_three(
                    a_buf, b_buf, as_buf, bs_buf, slot, (tile_idx + 2) % 3,
                    a_desc, b_desc, as_desc, bs_desc, acc, dot_a, dot_b,
                    scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N, BLOCK_K,
                    SCALE_STEP, PRESHUFFLE_SCALES))
        # Two prefetched tiles remain after the final refill.
        tdm.async_wait(3)
        mxfp8_tiled_cluster_wait()
        penultimate_slot = (iter_max - 2) % 3
        acc = mxfp8_tiled_consume(
            a_buf, b_buf, as_buf, bs_buf, penultimate_slot, acc, dot_a, dot_b,
            scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N, BLOCK_K,
            PRESHUFFLE_SCALES)
        tdm.async_wait(0)
        mxfp8_tiled_cluster_wait()
        last_slot = (iter_max - 1) % 3
        acc = mxfp8_tiled_consume(
            a_buf, b_buf, as_buf, bs_buf, last_slot, acc, dot_a, dot_b,
            scale_a_layout, scale_b_layout, BLOCK_M, BLOCK_N, BLOCK_K,
            PRESHUFFLE_SCALES)
    # The output stage aliases cluster-distributed operand LDS. All CTAs must
    # finish the final operand reads before any CTA repurposes that arena.
    mxfp8_tiled_cluster_wait()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    if SERIAL_N2_OUTPUT:
        mxfp8_tiled_store_serial_n2(
            c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
            OUTPUT_CGA_LAYOUT, CTA_M, CTA_N)
    else:
        mxfp8_tiled_store_full_tile(
            c_ptr, pid_m, pid_n, stride_cm, stride_cn, M, N, acc,
            WMMA_LAYOUT, BLOCK_M, BLOCK_N, CTA_N)


def build_mxfp8_tiled_layouts(config, shared_layout=DEFAULT_SHARED_LAYOUT):
    """Build cluster data, scale, and WMMA layouts for one configuration.

    Data operands use the K128 scaled-WMMA instruction shape. Scale layouts
    reuse the corresponding A/B CGA bases, so data and its block-32 scales are
    owned by the same CTA partition. A 256-wide CTA tile slices N to 128
    columns to match the serial epilogue and to keep the accumulator off the
    spill threshold; narrower tiles partition at their own width.

    ``shared_layout`` picks the operand LDS form; see MXFP8_SHARED_LAYOUTS.
    """
    cta_m = config["cta_tile_m"]
    cta_n = config["cta_tile_n"]
    block_k = config["block_k"]
    cluster_m = config["cluster_m"]
    cluster_n = config["cluster_n"]
    block_m = config["block_m"]
    block_n = config["block_n"]
    slice_m = cta_m
    slice_n = 128 if cta_n == 256 else cta_n
    cga_c = make_cga_layout(
        [cluster_m, cluster_n], [cluster_m, cluster_n], [0, 1])
    output_cga = tuple(tuple(basis) + (0, 0) for basis in cga_c)
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[block_k, 16]], [block_m, block_k], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[block_k, 16]], [block_n, block_k], [1, 0])
    _, _, local_wmma = gl.amd.gfx1250.make_partitioned_dot_layouts(
        block_m, block_n, local_a, local_b, NUM_WARPS, [16, 16, 128],
        a_transposed=False, b_transposed=True,
        slice_m=slice_m, slice_n=slice_n)
    wmma = gl.amd.AMDWMMALayout(
        3, local_wmma.transposed, local_wmma.warp_bases,
        local_wmma.reg_bases, local_wmma.instr_shape, cga_c)
    dot_a = gl.DotOperandLayout(0, wmma, 16)
    dot_b = gl.DotOperandLayout(1, wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple(tuple([basis[1], basis[0]])
                  for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[block_k, 16]], [block_m, block_k], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[block_k, 16]], [block_n, block_k], [1, 0], cga_b)
    if shared_layout == "plain":
        shared_a, shared_b = padded_a, padded_b
    else:
        shared_a, shared_b, _ = gl.amd.gfx1250.make_partitioned_dot_layouts(
            cta_m, cta_n, padded_a, padded_b, NUM_WARPS, [16, 16, 128],
            a_transposed=False, b_transposed=True,
            slice_m=slice_m, slice_n=slice_n)
    scale_step = config["scale_step"]
    if config["preshuffle_scales"]:
        shared_as = gl.PaddedSharedLayout.with_identity_for(
            [[256, 8]], [block_m // 128, scale_step], [1, 0], cga_a)
        shared_bs = gl.PaddedSharedLayout.with_identity_for(
            [[256, 8]], [block_n // 128, scale_step], [1, 0], cga_b)
    else:
        shared_as = gl.PaddedSharedLayout.with_identity_for(
            [[scale_step, 8]], [block_m, scale_step], [1, 0], cga_a)
        shared_bs = gl.PaddedSharedLayout.with_identity_for(
            [[scale_step, 8]], [block_n, scale_step], [1, 0], cga_b)
    return shared_a, shared_b, shared_as, shared_bs, wmma, output_cga


def make_mxfp8_tiled_case(args):
    """Build the E4M3/E8M0 block-32 benchmark case for one tile/cluster pick."""
    config = mxfp8_tiled_config(args.tile, args.cluster)
    block_m = config["block_m"]
    block_n = config["block_n"]
    block_k = config["block_k"]
    if args.M % block_m or args.N % block_n or args.K % block_k:
        raise ValueError(
            f"tile {args.tile} on a {args.cluster} cluster requires M "
            f"divisible by {block_m}, N by {block_n}, and K by {block_k}")
    num_slots = getattr(args, "buffers", DEFAULT_BUFFERS)
    if num_slots == 3 and args.tile not in ("64x64x512", "128x128x256"):
        raise ValueError(
            "full-BLOCK_K triple buffering exceeds the 320-KiB LDS limit "
            f"for {args.tile}; use --buffers 2, --tile 64x64x512, or "
            "--tile 128x128x256")
    if args.K // block_k < num_slots:
        raise ValueError(
            f"{num_slots}-buffer refill needs at least {num_slots} K tiles; "
            f"K={args.K} "
            f"gives {args.K // block_k} at BLOCK_K={block_k}")
    if (args.K // SCALE_BLOCK) % 4:
        raise ValueError(
            "scaled WMMA consumes four block-32 scales per fragment, so "
            "K/32 must be divisible by 4")
    torch.manual_seed(args.seed)
    mode = args.input_mode or "trig"
    if mode == "trig":
        a = make_trig_tensor((args.M, args.K), torch.float8_e4m3fn, False)
        b = make_trig_tensor((args.K, args.N), torch.float8_e4m3fn, True)
    else:
        a = init_data("float8_e4m3", args.M, args.K)
        b = init_data("float8_e4m3", args.K, args.N)
    scale_k = args.K // SCALE_BLOCK
    a_scale_obj = MXScaleTensor(
        size=(args.M, scale_k)).random(low=1.0, high=32.0)
    b_scale_obj = MXScaleTensor(
        size=(args.N, scale_k)).random(low=1.0, high=32.0)
    c_ref = None
    if args.check:
        c_ref = torch_gemm_mxfp(
            a, b, a_scale_obj, b_scale_obj, SCALE_BLOCK,
            args.M, args.N, args.K).to(torch.bfloat16)
    if config["preshuffle_scales"]:
        a_scale = pack_scale(a_scale_obj.data, 4)
        b_scale = pack_scale(b_scale_obj.data, 4)
    else:
        a_scale = a_scale_obj.data.contiguous()
        b_scale = b_scale_obj.data.contiguous()
    a_d = a.contiguous().cuda()
    b_d = b.T.contiguous().cuda()
    as_d = a_scale.cuda()
    bs_d = b_scale.cuda()
    c_d = torch.empty(
        (args.M, args.N), dtype=torch.bfloat16, device="cuda")
    layouts = build_mxfp8_tiled_layouts(
        config, getattr(args, "shared_layout", DEFAULT_SHARED_LAYOUT))
    shared_a, shared_b, shared_as, shared_bs, wmma, output_cga = layouts
    grid = (
        triton.cdiv(args.M, block_m) * triton.cdiv(args.N, block_n), 1)
    num_ctas = config["cluster_m"] * config["cluster_n"]

    def launch():
        return mxfp8_tiled_cluster_kernel_gfx1250[grid](
            a_d, b_d, c_d, as_d, bs_d, args.M, args.N, args.K,
            a_d.stride(0), a_d.stride(1), b_d.stride(1), b_d.stride(0),
            c_d.stride(0), c_d.stride(1), as_d.stride(0),
            GRID_MN=grid[0], SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b, SHARED_SCALE_A=shared_as,
            SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma,
            OUTPUT_CGA_LAYOUT=output_cga,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            CTA_M=config["cluster_m"], CTA_N=config["cluster_n"],
            CTA_TILE_M=config["cta_tile_m"],
            CTA_TILE_N=config["cta_tile_n"],
            SCALE_STEP=config["scale_step"],
            PRESHUFFLE_SCALES=config["preshuffle_scales"],
            SERIAL_N2_OUTPUT=config["serial_n2_output"],
            NUM_SLOTS=num_slots,
            num_warps=NUM_WARPS, waves_per_eu=WAVES_PER_EU,
            num_ctas=num_ctas, llvm_fn_attrs=AGPR_ATTRS)

    def check():
        c_d.zero_()
        launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            c_d.cpu(), c_ref, rtol=1e-2, atol=5e-1)
        print("result verified", flush=True)

    return launch, check


def describe(args):
    """Print the resolved configuration behind a tile/cluster selection."""
    config = mxfp8_tiled_config(args.tile, args.cluster)
    tiles_m = triton.cdiv(args.M, config["block_m"])
    tiles_n = triton.cdiv(args.N, config["block_n"])
    num_ctas = config["cluster_m"] * config["cluster_n"]
    print(f"tile             : {args.tile} per CTA")
    print(f"cluster          : {args.cluster} ({num_ctas} CTAs)")
    print(f"cluster tile     : {config['block_m']}x{config['block_n']}"
          f"x{config['block_k']}")
    print(f"k tiles          : {args.K // config['block_k']}")
    print(f"output tiles     : {tiles_m * tiles_n}")
    print(f"ctas resident    : {tiles_m * tiles_n * num_ctas}")
    print(f"scale form       : "
          f"{'preshuffled' if config['preshuffle_scales'] else 'unshuffled'}")
    print(f"output epilogue  : "
          f"{'serial-n2' if config['serial_n2_output'] else 'full-tile'}")
    print(f"operand lds      : "
          f"{getattr(args, 'shared_layout', DEFAULT_SHARED_LAYOUT)}")
    print(f"buffer slots     : {getattr(args, 'buffers', DEFAULT_BUFFERS)}")


def parse_args():
    """Parse the standalone correctness/benchmark command-line interface."""
    parser = argparse.ArgumentParser(
        description=("Run the tile- and cluster-configurable GFX1250 MXFP8 "
                     "GEMM"))
    parser.add_argument(
        "--tile", choices=tuple(MXFP8_TILE_SHAPES), default=DEFAULT_TILE,
        help="per-CTA output tile and K depth")
    parser.add_argument(
        "--cluster", choices=tuple(MXFP8_CLUSTER_SHAPES),
        default=DEFAULT_CLUSTER, help="square CTA-cluster shape")
    parser.add_argument(
        "--shared-layout", choices=MXFP8_SHARED_LAYOUTS,
        default=DEFAULT_SHARED_LAYOUT,
        help="operand LDS form; 'plain' drops LDS partitioning to satisfy "
             "toolchains that reject the partitioned form")
    parser.add_argument(
        "--buffers", type=int, choices=MXFP8_BUFFER_COUNTS,
        default=DEFAULT_BUFFERS, help="number of operand TDM ring slots")
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-K", type=int, default=65536)
    parser.add_argument(
        "--input-mode", choices=("random", "trig"), default="trig",
        help="input value pattern")
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
    print(f"\n=== MXFP8 {args.tile} cga{args.cluster} ===", flush=True)
    describe(args)
    launch, check = make_mxfp8_tiled_case(args)
    if args.check:
        check()
    if args.benchmark:
        run_benchmark(launch, args.M, args.N, args.K, args)
    del launch, check
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
