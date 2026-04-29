#!/usr/bin/env python3

from __future__ import annotations

import json
import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT_DIR / "build"
DATASET = ROOT_DIR / "training_data" / "datasets" / "names.txt"
FIXTURE_DIR = ROOT_DIR / "training_data" / "fixtures" / "small_case"
DEFAULT_OUTPUT = ROOT_DIR / "benchmark_results.json"
METHODS = ("serial_python", "serial_torch", "parallel_torch", "serial_cpp", "parallel_cpp")
PRESETS = (
    "small",
    "medium",
    "large",
    "very-large",
    "extra-large",
    "names-1k",
    "names-5k",
    "names-10k",
    "names-20k",
    "names-30k",
    "model-small-1k",
    "model-medium-1k",
    "model-large-1k",
    "model-very-large-1k",
    "model-extra-large-1k",
)
DEFAULT_BATCH_SIZE = "32"
METHOD_PRESETS = {
    "serial_python": {"small"},
}


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


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


def write_results(output_path: Path, results: dict[str, object]) -> None:
    output_path.write_text(json.dumps(results, indent=2) + "\n")


def run_validate_method(method: str) -> CommandResult:
    if method == "serial_python":
        return run_command(
            [
                sys.executable,
                str(ROOT_DIR / "methods" / "serial_python" / "serial.py"),
                "--mode",
                "validate",
                "--fixture-dir",
                str(FIXTURE_DIR),
            ]
        )
    if method == "serial_torch":
        return run_command(
            [
                sys.executable,
                str(ROOT_DIR / "methods" / "serial_torch" / "serial.py"),
                "--mode",
                "validate",
                "--fixture-dir",
                str(FIXTURE_DIR),
            ]
        )
    if method == "parallel_torch":
        return run_command(
            [
                sys.executable,
                str(ROOT_DIR / "methods" / "parallel_torch" / "parallel.py"),
                "--mode",
                "validate",
                "--fixture-dir",
                str(FIXTURE_DIR),
                "--batch-size",
                DEFAULT_BATCH_SIZE,
            ]
        )
    command = [
        str(BUILD_DIR / method),
        "--mode",
        "validate",
        "--fixture-dir",
        str(FIXTURE_DIR),
    ]
    if method == "parallel_cpp":
        command.extend(["--batch-size", DEFAULT_BATCH_SIZE])
    return run_command(
        command
    )


def run_benchmark_method(method: str, preset: str) -> CommandResult:
    if method == "serial_python":
        return run_command(
            [
                sys.executable,
                str(ROOT_DIR / "methods" / "serial_python" / "serial.py"),
                "--mode",
                "benchmark",
                "--dataset",
                str(DATASET),
                "--preset",
                preset,
            ]
        )
    if method == "serial_torch":
        return run_command(
            [
                sys.executable,
                str(ROOT_DIR / "methods" / "serial_torch" / "serial.py"),
                "--mode",
                "benchmark",
                "--dataset",
                str(DATASET),
                "--preset",
                preset,
            ]
        )
    if method == "parallel_torch":
        return run_command(
            [
                sys.executable,
                str(ROOT_DIR / "methods" / "parallel_torch" / "parallel.py"),
                "--mode",
                "benchmark",
                "--dataset",
                str(DATASET),
                "--preset",
                preset,
                "--batch-size",
                DEFAULT_BATCH_SIZE,
            ]
        )
    command = [
        str(BUILD_DIR / method),
        "--mode",
        "benchmark",
        "--dataset",
        str(DATASET),
        "--preset",
        preset,
    ]
    if method == "parallel_cpp":
        command.extend(["--batch-size", DEFAULT_BATCH_SIZE])
    return run_command(command)


def main() -> int:
    output_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUTPUT

    results: dict[str, object] = {
        "dataset": str(DATASET),
        "fixture_dir": str(FIXTURE_DIR),
        "build": {},
        "validate": {},
        "benchmarks": [],
    }

    build_commands: list[list[str]] = [
        ["make", "-C", str(ROOT_DIR), "fixtures", "build/serial_cpp"],
    ]
    if shutil.which("nvcc") is not None:
        build_commands[0].append("build/parallel_cpp")
    else:
        results["build"] = {
            "parallel_cpp": {
                "status": "skipped",
                "reason": "nvcc not found on PATH",
            }
        }
        print("build parallel_cpp: skipped, nvcc not found on PATH", flush=True)
        write_results(output_path, results)

    for command in build_commands:
        print(f"build: {' '.join(command)}", flush=True)
        build_result = run_command(command)
        results["build"]["core"] = {
            "command": command,
            "returncode": build_result.returncode,
            "stdout": build_result.stdout,
            "stderr": build_result.stderr,
        }
        print(f"build core: {'pass' if build_result.returncode == 0 else 'fail'}", flush=True)
        write_results(output_path, results)
        if build_result.returncode != 0:
            return build_result.returncode

    valid_methods: dict[str, bool] = {}
    for method in METHODS:
        if method.endswith("_torch") and importlib.util.find_spec("torch") is None:
            valid_methods[method] = False
            results["validate"][method] = {
                "status": "skipped",
                "reason": "torch not installed",
            }
            print(f"validate {method}: skipped, torch not installed", flush=True)
            write_results(output_path, results)
            continue
        if method == "parallel_cpp" and shutil.which("nvcc") is None:
            valid_methods[method] = False
            results["validate"][method] = {
                "status": "skipped",
                "reason": "nvcc not found on PATH",
            }
            print(f"validate {method}: skipped, nvcc not found on PATH", flush=True)
            write_results(output_path, results)
            continue

        print(f"validate {method}: running", flush=True)
        validate_result = run_validate_method(method)
        valid_methods[method] = validate_result.returncode == 0
        results["validate"][method] = {
            "status": "pass" if validate_result.returncode == 0 else "fail",
            "returncode": validate_result.returncode,
            "stdout": validate_result.stdout,
            "stderr": validate_result.stderr,
        }
        print(f"validate {method}: {results['validate'][method]['status']}", flush=True)
        write_results(output_path, results)

    for preset in PRESETS:
        for method in METHODS:
            allowed_presets = METHOD_PRESETS.get(method)
            if allowed_presets is not None and preset not in allowed_presets:
                results["benchmarks"].append(
                    {
                        "method": method,
                        "preset": preset,
                        "status": "skipped",
                        "reason": "preset disabled for method",
                    }
                )
                print(f"benchmark {method} {preset}: skipped, preset disabled for method", flush=True)
                write_results(output_path, results)
                continue

            if not valid_methods.get(method, False):
                results["benchmarks"].append(
                    {
                        "method": method,
                        "preset": preset,
                        "status": "skipped",
                    }
                )
                print(f"benchmark {method} {preset}: skipped, method invalid", flush=True)
                write_results(output_path, results)
                continue

            print(f"benchmark {method} {preset}: running", flush=True)
            benchmark_result = run_benchmark_method(method, preset)
            parsed = parse_key_value_line(benchmark_result.stdout)
            benchmark_entry = {
                "method": method,
                "preset": preset,
                "status": "pass" if benchmark_result.returncode == 0 else "fail",
                "returncode": benchmark_result.returncode,
                "stdout": benchmark_result.stdout,
                "stderr": benchmark_result.stderr,
                "parsed": parsed,
            }
            results["benchmarks"].append(benchmark_entry)
            summary = ""
            if "total_program_seconds" in parsed:
                summary = f" total_program_seconds={parsed['total_program_seconds']}"
            print(f"benchmark {method} {preset}: {benchmark_entry['status']}{summary}", flush=True)
            write_results(output_path, results)

    write_results(output_path, results)
    print(f"wrote benchmark results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
