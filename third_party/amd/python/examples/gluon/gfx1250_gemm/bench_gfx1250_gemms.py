"""Run the canonical GFX1250 Gluon GEMM correctness and performance harness.

This wrapper supports both module execution and direct execution:

  python3 -m third_party.amd.python.examples.gluon.gfx1250_gemm.bench_gfx1250_gemms ...
  python3 third_party/amd/python/examples/gluon/gfx1250_gemm/bench_gfx1250_gemms.py ...

All behavior lives in ``kernels.main`` so both invocation forms use identical
kernel objects, input preparation, validation, and timing methodology.
"""

from pathlib import Path
import sys


# Module execution already has a package context. Direct script execution does
# not, so add the repository root and use the same absolute package import.
# Keeping this compatibility logic here prevents path handling from leaking
# into the performance-sensitive kernel module.
if __package__:
    from .kernels import main
else:
    repo_root = Path(__file__).resolve().parents[6]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from third_party.amd.python.examples.gluon.gfx1250_gemm.kernels import main


if __name__ == "__main__":
    main()
