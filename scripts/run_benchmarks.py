#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from benchmark_matrix import BENCHMARK_RUNS, METHOD_RUN_FILTERS, BenchmarkRun, run_details


ROOT_DIR = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT_DIR / "build"
DATASET = ROOT_DIR / "training_data" / "datasets" / "names.txt"
FIXTURE_DIR = ROOT_DIR / "training_data" / "fixtures" / "small_case"
DEFAULT_OUTPUT = ROOT_DIR / "benchmark_results.json"
METHODS = ("serial_python", "serial_torch", "parallel_torch", "serial_cpp", "parallel_cpp")
DEFAULT_BATCH_SIZE = "512"
RUN_DETAILS = run_details()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run all validation and benchmark methods")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path, written incrementally while benchmarks run",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=None,
        help="HTML report path, written after the full benchmark sweep finishes",
    )
    return parser.parse_args()


def html_escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def parse_float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def format_decimal(value: object, digits: int) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "&mdash;"
    return f"{parsed:.{digits}f}"


def format_text(value: object) -> str:
    if value is None:
        return "&mdash;"
    return html_escape(value)


def format_speedup(baseline_seconds: float | None, current_seconds: object) -> str:
    current = parse_float(current_seconds)
    if baseline_seconds is None or current is None or current == 0.0:
        return "&mdash;"
    return f"{baseline_seconds / current:.2f}&times;"


def benchmark_lookup(results: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    entries = results.get("benchmarks", [])
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    if not isinstance(entries, list):
        return lookup
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        method = entry.get("method")
        preset = entry.get("preset")
        if isinstance(method, str) and isinstance(preset, str):
            lookup[(preset, method)] = entry
    return lookup


def baseline_seconds(
    lookup: dict[tuple[str, str], dict[str, object]], preset: str, method: str
) -> float | None:
    entry = lookup.get((preset, method))
    if entry is None or entry.get("status") != "pass":
        return None
    parsed = entry.get("parsed")
    if not isinstance(parsed, dict):
        return None
    return parse_float(parsed.get("total_program_seconds"))


def render_preset_info() -> str:
    rows = []
    for run in BENCHMARK_RUNS:
        rows.append(
            "<tr>"
            f"<td>{html_escape(run.label)}</td>"
            f"<td>{run.n_layer}</td>"
            f"<td>{run.n_embd}</td>"
            f"<td>{run.block_size}</td>"
            f"<td>{run.n_head}</td>"
            f"<td>{run.steps}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_benchmark_rows(results: dict[str, object]) -> str:
    lookup = benchmark_lookup(results)
    rows = []
    for run in BENCHMARK_RUNS:
        preset = run.label
        rows.append(
            "<tr class=\"preset-row\">"
            f"<th colspan=\"9\">preset: {html_escape(preset)}</th>"
            "</tr>"
        )
        serial_cpp_seconds = baseline_seconds(lookup, preset, "serial_cpp")
        serial_python_seconds = baseline_seconds(lookup, preset, "serial_python")
        for method in METHODS:
            entry = lookup.get((preset, method))
            if entry is None:
                rows.append(
                    "<tr>"
                    f"<td>{html_escape(method)}</td>"
                    f"<td>{html_escape(preset)}</td>"
                    "<td>missing</td>"
                    "<td>&mdash;</td>"
                    "<td>&mdash;</td>"
                    "<td>&mdash;</td>"
                    "<td>&mdash;</td>"
                    "<td>&mdash;</td>"
                    "<td>&mdash;</td>"
                    "</tr>"
                )
                continue

            parsed = entry.get("parsed")
            if not isinstance(parsed, dict):
                parsed = {}
            status = entry.get("status", "unknown")
            total_seconds = parsed.get("total_program_seconds")
            rows.append(
                "<tr>"
                f"<td>{html_escape(method)}</td>"
                f"<td>{html_escape(preset)}</td>"
                f"<td>{html_escape(status)}</td>"
                f"<td>{format_text(parsed.get('steps'))}</td>"
                f"<td>{format_decimal(parsed.get('loss'), 4)}</td>"
                f"<td>{format_decimal(parsed.get('forward_pass_seconds_cumulative'), 4)}</td>"
                f"<td>{format_decimal(total_seconds, 4)}</td>"
                f"<td>{format_speedup(serial_cpp_seconds, total_seconds)}</td>"
                f"<td>{format_speedup(serial_python_seconds, total_seconds)}</td>"
                "</tr>"
            )
    return "\n".join(rows)


def write_html_report(output_path: Path, results: dict[str, object]) -> None:
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Benchmark Results</title>
  <style>
    body {{
      color: #1f2933;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.4;
      margin: 24px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    h2 {{
      margin-top: 28px;
    }}
    table {{
      border-collapse: collapse;
      margin-bottom: 24px;
      width: 100%;
    }}
    th, td {{
      border: 1px solid #d6dde6;
      padding: 7px 9px;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      background: #eef2f7;
      font-weight: 600;
    }}
    .preset-row th {{
      background: #dbeafe;
      color: #102a43;
      font-size: 1rem;
    }}
    .meta {{
      color: #52616b;
      margin: 0 0 18px;
    }}
  </style>
</head>
<body>
  <h1>Benchmark Results</h1>
  <p class="meta">Dataset: {html_escape(results.get("dataset", ""))}<br>Default batch size: {DEFAULT_BATCH_SIZE}</p>

  <h2>Preset Info</h2>
  <table>
    <thead>
      <tr>
        <th>Preset</th>
        <th>n_layer</th>
        <th>n_embd</th>
        <th>block_size</th>
        <th>n_head</th>
        <th>steps</th>
      </tr>
    </thead>
    <tbody>
{render_preset_info()}
    </tbody>
  </table>

  <h2>Results</h2>
  <table>
    <thead>
      <tr>
        <th>Method</th>
        <th>Preset</th>
        <th>Status</th>
        <th>Steps</th>
        <th>Loss</th>
        <th>Fwd pass (s)</th>
        <th>Total (s)</th>
        <th>vs serial_cpp</th>
        <th>vs serial_python</th>
      </tr>
    </thead>
    <tbody>
{render_benchmark_rows(results)}
    </tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(document)


def run_validate_method(method: str) -> CommandResult:
    if method == "serial_python":
        return run_command(
            [
                sys.executable,
                str(ROOT_DIR / "methods" / "serial_python" / "serial.py"),
                "--mode",
                "validate",
            ]
        )
    if method == "serial_torch":
        return run_command(
            [
                sys.executable,
                str(ROOT_DIR / "methods" / "serial_torch" / "serial.py"),
                "--mode",
                "validate",
            ]
        )
    if method == "parallel_torch":
        return run_command(
            [
                sys.executable,
                str(ROOT_DIR / "methods" / "parallel_torch" / "parallel.py"),
                "--mode",
                "validate",
            ]
        )
    command = [
        str(BUILD_DIR / method),
        "--mode",
        "validate",
    ]
    return run_command(command)


def benchmark_command(method: str, run: BenchmarkRun) -> list[str]:
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
    if method == "parallel_torch" or method == "parallel_cpp":
        command.extend(["--batch-size", DEFAULT_BATCH_SIZE])
    return command


def run_benchmark_method(method: str, run: BenchmarkRun) -> CommandResult:
    command = benchmark_command(method, run)
    if method == "serial_python":
        return run_command(
            [
                sys.executable,
                str(ROOT_DIR / "methods" / "serial_python" / "serial.py"),
                *command,
            ]
        )
    if method == "serial_torch":
        return run_command(
            [
                sys.executable,
                str(ROOT_DIR / "methods" / "serial_torch" / "serial.py"),
                *command,
            ]
        )
    if method == "parallel_torch":
        return run_command(
            [
                sys.executable,
                str(ROOT_DIR / "methods" / "parallel_torch" / "parallel.py"),
                *command,
            ]
        )
    binary_command = [
        str(BUILD_DIR / method),
        *command,
    ]
    return run_command(binary_command)


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve()
    html_output_path = (
        args.html_output.resolve() if args.html_output is not None else output_path.with_suffix(".html")
    )

    results: dict[str, object] = {
        "dataset": str(DATASET),
        "fixture_dir": str(FIXTURE_DIR),
        "presets": RUN_DETAILS,
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

    for run in BENCHMARK_RUNS:
        preset = run.label
        for method in METHODS:
            allowed_presets = METHOD_RUN_FILTERS.get(method)
            if allowed_presets is not None and preset not in allowed_presets:
                results["benchmarks"].append(
                    {
                        "method": method,
                        "preset": preset,
                        "preset_details": RUN_DETAILS[preset],
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
                        "preset_details": RUN_DETAILS[preset],
                        "status": "skipped",
                    }
                )
                print(f"benchmark {method} {preset}: skipped, method invalid", flush=True)
                write_results(output_path, results)
                continue

            print(f"benchmark {method} {preset}: running", flush=True)
            benchmark_result = run_benchmark_method(method, run)
            parsed = parse_key_value_line(benchmark_result.stdout)
            benchmark_entry = {
                "method": method,
                "preset": preset,
                "preset_details": RUN_DETAILS[preset],
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
    write_html_report(html_output_path, results)
    print(f"wrote benchmark results to {output_path}")
    print(f"wrote benchmark report to {html_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
