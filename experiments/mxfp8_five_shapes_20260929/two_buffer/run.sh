#!/usr/bin/env bash
set -euo pipefail
cd /home/jungpark/mnt/gemm/triton
source /home/jungpark/.rock/bin/activate
unset TRITON_KERNEL_OVERRIDE TRITON_OVERRIDE_DIR TRITON_KERNEL_DUMP TRITON_DUMP_DIR TRITON_HIP_USE_EXPERT_SCHEDULING AMD_SERIALIZE_KERNEL LD_PRELOAD
export PYTHONPATH=/home/jungpark/mnt/wpindex/triton/python
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR=/tmp/mxfp8_five_shapes_0929_two_buffer_cache
python3 -u experiments/mxfp8_five_shapes_20260929/two_buffer/compare.py "$@"
