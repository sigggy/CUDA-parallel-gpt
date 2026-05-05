#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from benchmark_matrix import METHOD_RUN_FILTERS, RUN_BY_LABEL


ROOT_DIR = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT_DIR / "build"
DATASET = ROOT_DIR / "training_data" / "datasets" / "names.txt"
CUDA_METHODS = (
    "parallel_cpp",
    "baseline_cuda",
    "batching_only",
    "float_only",
    "tiled_matmul_only",
    "batching_float_tiled",
)
BUILD_METHODS = ("serial_cpp", *CUDA_METHODS)
METHODS = ("serial_python", "unbatched_torch", "batched_torch", *BUILD_METHODS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run one benchmark preset for one method")
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("preset", choices=sorted(RUN_BY_LABEL))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def benchmark_command(method: str, preset: str, batch_size: int, device: str) -> list[str]:
    run = RUN_BY_LABEL[preset]
    command = [
        "--mode",
        "benchmark",
        "--dataset",
        str(DATASET),
        "--label",
        run.label,
        "--num-steps",
        str(run.steps),
        "--n-layer",
        str(run.n_layer),
        "--n-embd",
        str(run.n_embd),
        "--block-size",
        str(run.block_size),
        "--n-head",
        str(run.n_head),
    ]
    if method in {"unbatched_torch", "batched_torch"}:
        command.extend(["--device", device])
    if method == "batched_torch" or method in CUDA_METHODS:
        command.extend(["--batch-size", str(batch_size)])
    if method in BUILD_METHODS:
        return [str(BUILD_DIR / method), *command]
    if method == "batched_torch":
        script_name = "parallel.py"
        script_dir = "parallel_torch"
    elif method == "unbatched_torch":
        script_name = "serial.py"
        script_dir = "serial_torch"
    else:
        script_name = "serial.py"
        script_dir = method
    return [sys.executable, str(ROOT_DIR / "methods" / script_dir / script_name), *command]


def main() -> int:
    args = parse_args()
    allowed_presets = METHOD_RUN_FILTERS.get(args.method)
    if allowed_presets is not None and args.preset not in allowed_presets:
        print(f"{args.method} does not run preset {args.preset}", file=sys.stderr)
        return 2
    completed = subprocess.run(
        benchmark_command(args.method, args.preset, args.batch_size, args.device),
        cwd=ROOT_DIR,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
