"""Conservative, forward-looking GFX1250 WMMA B-reuse assembly patch.

Mark an instruction only when the next WMMA consumes the same physical B
register range, with only waits or VGPR-bank selection between them. Loads,
register writes, barriers, and basic-block boundaries terminate a sequence.
"""

import re
from pathlib import Path


SET_MSB = re.compile(r"msbs: dst=(\d+) src0=(\d+) src1=(\d+) src2=(\d+)")
VGPR_RANGE = re.compile(r"v(?:\[(\d+):(\d+)\]|(\d+))")
WMMA_OPS = {
    "v_wmma_f32_16x16x32_bf16",
    "v_wmma_scale_f32_16x16x128_f8f6f4",
    "v_wmma_scale_f32_32x16x128_f4",
}


def physical_range(operand, msb):
    match = VGPR_RANGE.fullmatch(operand.strip())
    if not match:
        raise ValueError(f"Cannot parse VGPR operand: {operand}")
    lo, hi = ((int(match[3]), int(match[3])) if match[3] is not None
              else (int(match[1]), int(match[2])))
    return lo + 256 * msb, hi + 256 * msb


def patch_b_reuse(assembly):
    """Return assembly with eligible matrix_b_reuse modifiers added."""
    if "matrix_b_reuse" in assembly:
        raise ValueError("Expected baseline assembly without B reuse")
    lines = assembly.splitlines(keepends=True)
    previous = None
    msb = 0
    marked = 0
    for index, line in enumerate(lines):
        change = SET_MSB.search(line)
        if change:
            msb = int(change[3])
        text = line.strip()
        if not text or text.startswith((".", ";", "#")) or text.endswith(":"):
            if text.endswith(":") and not text.startswith(".Ltmp"):
                previous = None
            continue
        clean = re.sub(r"/\*.*?\*/", "", text)
        opcode = clean.split()[0]
        if opcode in WMMA_OPS:
            operands = clean.split(None, 1)[1].split(",")
            b = physical_range(operands[2], msb)
            if previous is not None and previous[1:] == (opcode, b):
                j = previous[0]
                newline = "\n" if lines[j].endswith("\n") else ""
                lines[j] = lines[j].rstrip("\n") + " matrix_b_reuse" + newline
                marked += 1
            previous = (index, opcode, b)
        elif not re.match(r"s_(?:wait_dscnt|wait_alu|delay_alu|set_vgpr_msb)\b", opcode):
            previous = None
    if not marked:
        raise ValueError("No eligible WMMA B-reuse sequences found")
    patched = "".join(lines)
    if patched.replace(" matrix_b_reuse", "") != assembly:
        raise RuntimeError("Patch changed more than B-reuse modifiers")
    return patched


def compile_b_reuse(module, launch, baseline_assembly, override_dir):
    """Compile a separate module's launch with a freshly generated override.

    The caller must keep the baseline in a distinct imported module so its
    JIT cache remains available for paired comparisons. Returns the compiled
    kernel; the launch closure subsequently uses its cached patched kernel.
    """
    import triton
    from triton.runtime.cache import get_override_manager

    patched = patch_b_reuse(baseline_assembly)
    old_override = triton.knobs.compilation.override
    old_dir = triton.knobs.cache.override_dir
    old_always_compile = triton.knobs.compilation.always_compile
    try:
        triton.knobs.compilation.always_compile = True
        triton.knobs.compilation.override = False
        kernel = launch()
        triton.knobs.cache.override_dir = str(Path(override_dir).resolve())
        manager = get_override_manager(kernel.src.hash())
        destination = Path(manager.cache_dir) / (kernel.name + ".amdgcn")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(patched)
        for value in vars(module).values():
            if isinstance(value, triton.runtime.JITFunction):
                value.device_caches.clear()
        triton.knobs.compilation.override = True
        kernel = launch()
        if kernel.asm["amdgcn"] != patched:
            raise RuntimeError("Compiled assembly did not use the B-reuse override")
        return kernel
    finally:
        triton.knobs.compilation.override = old_override
        triton.knobs.cache.override_dir = old_dir
        triton.knobs.compilation.always_compile = old_always_compile
