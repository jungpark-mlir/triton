"""Check and compare retained GFX1250 kernels with assembly B reuse.

Select the wpindex compiler through PYTHONPATH and run under gpu-lock, e.g.:
  gpu-lock python3 bench_b_reuse.py --kernel mxfp8_b2 --output-dir /tmp/b-reuse

Defaults: 4096x4096x65536, eight alternating pairs, 20 warmups and
50 graph iterations x 20 replays. Both variants always pass correctness
before timing. Use --rounds 0 for correctness only. BF16/MXFP8/MXFP4 use
hipBLASLt-equivalent trig inputs; FP8 x MXFP4 defaults to random seed 42.
Outputs include raw samples, compiler paths, source hash and both assemblies.
No experiment directories, pre-generated overrides, or profiler dumps are needed.
"""

import argparse
import contextlib
import datetime
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import statistics
import sys
import tempfile
import types

if __package__:
    from .wmma_b_reuse import compile_b_reuse
else:
    from wmma_b_reuse import compile_b_reuse


SELECTORS = ("bf16", "mxfp8_b2", "mxfp4", "fp8_mxfp4_five")


def load_source(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=SELECTORS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for dim, default in (("M", 4096), ("N", 4096), ("K", 65536)):
        parser.add_argument("-" + dim, type=int, default=default)
    parser.add_argument("--input-mode", choices=("trig", "random"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--iters-per-graph", type=int, default=50)
    options = parser.parse_args()
    if min(options.M, options.N, options.K, options.replays, options.iters_per_graph) <= 0:
        parser.error("dimensions and timing iteration counts must be positive")
    if min(options.rounds, options.warmup) < 0:
        parser.error("rounds and warmup must be nonnegative")
    if options.input_mode is None:
        options.input_mode = "random" if options.kernel.startswith("fp8_mxfp4") else "trig"

    import triton
    from triton._C import libtriton

    triton.knobs.compilation.always_compile = True
    triton.knobs.compilation.override = False
    source = Path(__file__).with_name("kernels_best_0927.py")
    sys.path.insert(0, str(source.resolve().parents[6]))
    out = (options.output_dir / options.kernel /
           f"{options.M}x{options.N}x{options.K}")
    out.mkdir(parents=True, exist_ok=True)
    # Each invocation owns fresh override storage, avoiding stale later-stage files.
    override_dir = tempfile.mkdtemp(prefix="overrides-", dir=out)
    base = load_source("b_reuse_baseline", source)
    reuse = load_source("b_reuse_candidate", source)
    builder = base.BEST_CASE_BUILDERS[options.kernel]
    state = {}

    def build(args):
        launch, check = builder(args)
        state["check"] = check
        return launch, check

    base.BEST_CASE_BUILDERS[options.kernel] = build
    benchmark = base.run_benchmark

    def measure(launch, M, N, K, args):
        baseline = launch()
        reused_launch = types.FunctionType(
            launch.__code__, {**launch.__globals__, **vars(reuse)},
            closure=launch.__closure__)
        compile_b_reuse(reuse, reused_launch, baseline.asm["amdgcn"], override_dir)
        check = state["check"]
        cells = tuple(types.CellType(reused_launch) if name == "launch" else cell
                      for name, cell in zip(check.__code__.co_freevars, check.__closure__))
        print("CHECK B reuse", flush=True)
        types.FunctionType(check.__code__, check.__globals__, closure=cells)()
        launches = [("baseline", launch), ("reuse", reused_launch)]
        results = {}
        for name, fn in launches:
            kernel = fn()
            enabled = "matrix_b_reuse" in kernel.asm["amdgcn"]
            if enabled != (name == "reuse"):
                raise RuntimeError(f"Unexpected reuse state for {name}")
            results[name] = dict(
                samples_us=[], vgpr=kernel.n_regs,
                scratch_bytes=kernel.metadata.global_scratch_size,
                lds_bytes=kernel.metadata.shared,
                reuse_count=kernel.asm["amdgcn"].count("matrix_b_reuse"))
            (out / (name + ".amdgcn")).write_text(kernel.asm["amdgcn"])
            print(name, results[name], flush=True)
        for iteration in range(options.rounds):
            for name, fn in launches[::1 if iteration % 2 == 0 else -1]:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    benchmark(fn, M, N, K, args)
                match = re.search(r"per-iter\s*:\s*([\d.]+) us", buf.getvalue())
                if match is None:
                    raise RuntimeError(f"Cannot parse benchmark output: {buf.getvalue()}")
                us = float(match[1])
                results[name]["samples_us"].append(us)
                print("PAIR", iteration + 1, name, us, flush=True)
        for value in results.values():
            if value["samples_us"]:
                value["median_us"] = statistics.median(value["samples_us"])
        record = dict(
            selector=options.kernel, shape=[M, N, K], input_mode=args.input_mode,
            seed=args.seed, correctness="both passed", compiler=triton.__file__,
            native=libtriton.__file__, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            rounds=options.rounds, warmup=args.warmup, replays=args.replays,
            iterations_per_graph=args.iters_per_graph, results=results)
        if options.rounds:
            b, r = results["baseline"], results["reuse"]
            record["reuse_wins"] = sum(y < x for x, y in zip(b["samples_us"], r["samples_us"]))
            record["latency_reduction_percent"] = 100 * (1 - r["median_us"] / b["median_us"])
        (out / "results.json").write_text(json.dumps(record, indent=2) + "\n")
        print("RESULTS", out / "results.json", flush=True)

    base.run_benchmark = measure
    sys.argv = [str(source), "--kernel", options.kernel,
                "-M", str(options.M), "-N", str(options.N), "-K", str(options.K),
                "--input-mode", options.input_mode, "--seed", str(options.seed),
                "--check", "--benchmark", "--warmup", str(options.warmup),
                "--replays", str(options.replays),
                "--iters-per-graph", str(options.iters_per_graph)]
    if options.kernel == "mxfp8_b2":
        sys.argv += ["--mxfp8-b2-experiment", "unroll6-parent"]
    base.main()


if __name__ == "__main__":
    main()
