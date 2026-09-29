#!/usr/bin/env bash
set -euo pipefail
cd /home/jungpark/mnt/gemm/triton
source /home/jungpark/.rock/bin/activate
unset PYTHONPATH TRITON_KERNEL_OVERRIDE TRITON_OVERRIDE_DIR TRITON_KERNEL_DUMP TRITON_DUMP_DIR TRITON_HIP_USE_EXPERT_SCHEDULING AMD_SERIALIZE_KERNEL LD_PRELOAD
export PYTHONPATH=/home/jungpark/mnt/wpindex/triton/python
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR=/tmp/mxfp8_five_shapes_0929_cache
python3 -u third_party/amd/python/examples/gluon/gfx1250_gemm/kernels_mxfp8_five_shapes_0929.py "$@"
