#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from benchmark_matrix import BENCHMARK_PRESETS, BENCHMARK_RUNS, BenchmarkRun


ROOT_DIR = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT_DIR / "build"
DATASET = ROOT_DIR / "training_data" / "datasets" / "names.txt"
FIXTURE_DIR = ROOT_DIR / "training_data" / "fixtures" / "small_case"
DEFAULT_OUTPUT = ROOT_DIR / "benchmark_results.json"
DEFAULT_BATCH_SIZE = "512"
METHODS = ("serial_torch", "parallel_torch", "serial_cpp", "parallel_cpp")


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run the benchmark sweep")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> CommandResult:
    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def write_results(output_path: Path, results: dict[str, object]) -> None:
    output_path.write_text(json.dumps(results, indent=2) + "\n")


def parse_key_value_line(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in reversed(text.splitlines()):
        if "=" not in line:
            continue
        for part in line.strip().split():
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key] = value
        if fields:
            break
    return fields


def method_prefix(method: str) -> list[str]:
    if method == "serial_torch":
        return [sys.executable, str(ROOT_DIR / "methods" / "serial_torch" / "serial.py")]
    if method == "parallel_torch":
        return [sys.executable, str(ROOT_DIR / "methods" / "parallel_torch" / "parallel.py")]
    return [str(BUILD_DIR / method)]


def validate_command(method: str) -> list[str]:
    return [*method_prefix(method), "--mode", "validate"]


def benchmark_command(method: str, run: BenchmarkRun) -> list[str]:
    command = [
        *method_prefix(method),
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
    if method in {"parallel_torch", "parallel_cpp"}:
        command.extend(["--batch-size", DEFAULT_BATCH_SIZE])
    return command


def method_skip_reason(method: str, have_torch: bool, have_nvcc: bool) -> str | None:
    if method.endswith("_torch") and not have_torch:
        return "torch not installed"
    if method == "parallel_cpp" and not have_nvcc:
        return "nvcc not found on PATH"
    return None


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve()
    have_torch = importlib.util.find_spec("torch") is not None
    have_nvcc = shutil.which("nvcc") is not None
    skip_reasons = {
        method: method_skip_reason(method, have_torch, have_nvcc) for method in METHODS
    }

    results: dict[str, object] = {
        "dataset": str(DATASET),
        "fixture_dir": str(FIXTURE_DIR),
        "batch_size": int(DEFAULT_BATCH_SIZE),
        "presets": BENCHMARK_PRESETS,
        "build": {},
        "validate": {},
        "benchmarks": [],
    }

    build_command = ["make", "-C", str(ROOT_DIR), "fixtures", "build/serial_cpp"]
    if have_nvcc:
        build_command.append("build/parallel_cpp")
    else:
        results["build"]["parallel_cpp"] = {
            "status": "skipped",
            "reason": "nvcc not found on PATH",
        }

    print(f"build: {' '.join(build_command)}", flush=True)
    build_result = run_command(build_command)
    results["build"]["core"] = {
        "status": "pass" if build_result.returncode == 0 else "fail",
        "command": build_command,
        "returncode": build_result.returncode,
        "stdout": build_result.stdout,
        "stderr": build_result.stderr,
    }
    write_results(output_path, results)
    print(f"build core: {results['build']['core']['status']}", flush=True)
    if build_result.returncode != 0:
        return build_result.returncode

    valid_methods: dict[str, bool] = {}
    for method in METHODS:
        reason = skip_reasons[method]
        if reason is not None:
            valid_methods[method] = False
            results["validate"][method] = {
                "status": "skipped",
                "reason": reason,
            }
            print(f"validate {method}: skipped, {reason}", flush=True)
            write_results(output_path, results)
            continue

        print(f"validate {method}: running", flush=True)
        validate_result = run_command(validate_command(method))
        valid_methods[method] = validate_result.returncode == 0
        results["validate"][method] = {
            "status": "pass" if validate_result.returncode == 0 else "fail",
            "returncode": validate_result.returncode,
            "stdout": validate_result.stdout,
            "stderr": validate_result.stderr,
        }
        print(f"validate {method}: {results['validate'][method]['status']}", flush=True)
        write_results(output_path, results)

    for run in BENCHMARK_RUNS:
        preset_details = BENCHMARK_PRESETS[run.label]
        for method in METHODS:
            if not valid_methods.get(method, False):
                entry = {
                    "method": method,
                    "preset": run.label,
                    "preset_details": preset_details,
                    "status": "skipped",
                    "reason": skip_reasons[method] or "validation failed",
                }
                results["benchmarks"].append(entry)
                print(f"benchmark {method} {run.label}: skipped", flush=True)
                write_results(output_path, results)
                continue

            print(f"benchmark {method} {run.label}: running", flush=True)
            benchmark_result = run_command(benchmark_command(method, run))
            parsed = parse_key_value_line(benchmark_result.stdout)
            entry = {
                "method": method,
                "preset": run.label,
                "preset_details": preset_details,
                "status": "pass" if benchmark_result.returncode == 0 else "fail",
                "returncode": benchmark_result.returncode,
                "stdout": benchmark_result.stdout,
                "stderr": benchmark_result.stderr,
                "parsed": parsed,
            }
            results["benchmarks"].append(entry)

            summary = ""
            if "total_program_seconds" in parsed:
                summary = f" total_program_seconds={parsed['total_program_seconds']}"
            print(f"benchmark {method} {run.label}: {entry['status']}{summary}", flush=True)
            write_results(output_path, results)

    print(f"wrote benchmark results to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
