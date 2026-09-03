"""Public kernel API for the canonical GFX1250 Gluon GEMM examples.

The package exports one performance-retained kernel per datatype family.
Launch configuration, layout construction, correctness checks, and the CLI
remain in :mod:`.kernels`; use :mod:`.bench_gfx1250_gemms` to run them.

Device kernels are exported here for integration tests and focused experiments,
but callers must preserve the launch contracts documented in ``README.md``:
eight warps, the required CTA cluster, CGA-compatible layouts, and BF16 output.
"""

from .kernels import (
    KERNEL_NAMES,
    bf16_kernelc_2pf_bk128_fused_cluster_4x4_gfx1250,
    fp8_mxfp4_cluster_gemm_gfx1250,
    fp8_scaled_cluster_bf16_style_kernel_gfx1250,
    mxfp4_bk512_warp_pipeline_gfx1250,
)

__all__ = [
    "KERNEL_NAMES",
    "bf16_kernelc_2pf_bk128_fused_cluster_4x4_gfx1250",
    "fp8_mxfp4_cluster_gemm_gfx1250",
    "fp8_scaled_cluster_bf16_style_kernel_gfx1250",
    "mxfp4_bk512_warp_pipeline_gfx1250",
]
