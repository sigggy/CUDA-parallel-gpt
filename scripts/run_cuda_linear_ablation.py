#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from benchmark_matrix import BENCHMARK_RUNS, BenchmarkRun, run_details


ROOT_DIR = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT_DIR / "build"
DATASET = ROOT_DIR / "training_data" / "datasets" / "names.txt"
FIXTURE_DIR = ROOT_DIR / "training_data" / "fixtures" / "small_case"
DEFAULT_OUTPUT = ROOT_DIR / "cuda_linear_ablation_results.json"
METHODS = ("serial_cpp", "parallel_cpp_untiled", "parallel_cpp")
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
    parser = argparse.ArgumentParser(description="run the CUDA linear-kernel ablation sweep")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path, written incrementally while the ablation sweep runs",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=None,
        help="HTML report path, written after the full ablation sweep finishes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(DEFAULT_BATCH_SIZE),
        help="batch size for both CUDA methods",
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
        serial_seconds = baseline_seconds(lookup, preset, "serial_cpp")
        untiled_seconds = baseline_seconds(lookup, preset, "parallel_cpp_untiled")
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
                f"<td>{format_speedup(serial_seconds, total_seconds)}</td>"
                f"<td>{format_speedup(untiled_seconds, total_seconds)}</td>"
                "</tr>"
            )
    return "\n".join(rows)


def write_html_report(output_path: Path, results: dict[str, object], batch_size: int) -> None:
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CUDA Linear Ablation</title>
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
  <h1>CUDA Linear Ablation</h1>
  <p class="meta">Dataset: {html_escape(results.get("dataset", ""))}<br>CUDA batch size: {batch_size}</p>

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
        <th>vs untiled</th>
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
    return run_command([str(BUILD_DIR / method), "--mode", "validate"])


def benchmark_command(method: str, run: BenchmarkRun, batch_size: int) -> list[str]:
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
    if method != "serial_cpp":
        command.extend(["--batch-size", str(batch_size)])
    return [str(BUILD_DIR / method), *command]


def run_benchmark_method(method: str, run: BenchmarkRun, batch_size: int) -> CommandResult:
    return run_command(benchmark_command(method, run, batch_size))


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

    build_command = [
        "make",
        "-C",
        str(ROOT_DIR),
        "fixtures",
        "build/serial_cpp",
        "build/parallel_cpp_untiled",
        "build/parallel_cpp",
    ]
    if shutil.which("nvcc") is None:
        results["build"] = {
            "parallel_cpp_untiled": {
                "status": "skipped",
                "reason": "nvcc not found on PATH",
            },
            "parallel_cpp": {
                "status": "skipped",
                "reason": "nvcc not found on PATH",
            },
        }
        print("build CUDA methods: skipped, nvcc not found on PATH", flush=True)
        write_results(output_path, results)
        write_html_report(html_output_path, results, args.batch_size)
        print(f"wrote benchmark results to {output_path}")
        print(f"wrote benchmark report to {html_output_path}")
        return 0

    print(f"build: {' '.join(build_command)}", flush=True)
    build_result = run_command(build_command)
    results["build"]["core"] = {
        "command": build_command,
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
            benchmark_result = run_benchmark_method(method, run, args.batch_size)
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
    write_html_report(html_output_path, results, args.batch_size)
    print(f"wrote benchmark results to {output_path}")
    print(f"wrote benchmark report to {html_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
