#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path


# In-file config only. This script intentionally has no CLI.
INPUT_JSON = "cuda_linear_ablation_results.json"
OUTPUT_FILE = "cuda_ablation_names.svg"

PRESET_PREFIX = "names-"
Y_METRIC_KEY = "total_program_seconds"

TITLE = "CUDA Ablation Across Increasing Name Counts"
SUBTITLE = ""
X_LABEL = "Number of Names"
Y_LABEL = "Runtime (seconds)"
LEGEND_TITLE = "Methods"

METHOD_ORDER = [
    "serial_cpp",
    "baseline_cuda",
    "batching_only",
    "float_only",
    "tiled_matmul_only",
    "batching_float_tiled",
]

METHOD_LABELS = {
    "serial_cpp": "serial_cpp",
    "baseline_cuda": "baseline_cuda",
    "batching_only": "batching_only",
    "float_only": "float_only",
    "tiled_matmul_only": "tiled_matmul_only",
    "batching_float_tiled": "batching_float_tiled",
}

METHOD_COLORS = {
    "serial_cpp": "#111827",
    "baseline_cuda": "#b91c1c",
    "batching_only": "#d97706",
    "float_only": "#2563eb",
    "tiled_matmul_only": "#059669",
    "batching_float_tiled": "#7c3aed",
}

FAIL_IF_MISSING_POINTS = True
VERIFY_FIXED_MODEL = True
FORCE_Y_ZERO = False

SVG_WIDTH = 1280
SVG_HEIGHT = 820
FONT_FAMILY = "Helvetica, Arial, sans-serif"


def parse_float(value: object) -> float:
    return float(str(value))


def format_name_count(value: int) -> str:
    if value % 1000 == 0:
        return f"{value // 1000}k"
    return str(value)


def format_y_tick(value: float) -> str:
    if value >= 100.0:
        return f"{value:.0f}"
    if value >= 10.0:
        return f"{value:.1f}"
    if value >= 1.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def escape_xml(text: object) -> str:
    value = str(text)
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def nice_number(value: float, should_round: bool) -> float:
    exponent = math.floor(math.log10(value))
    fraction = value / (10 ** exponent)
    if should_round:
        if fraction < 1.5:
            nice_fraction = 1.0
        elif fraction < 3.0:
            nice_fraction = 2.0
        elif fraction < 7.0:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0
    else:
        if fraction <= 1.0:
            nice_fraction = 1.0
        elif fraction <= 2.0:
            nice_fraction = 2.0
        elif fraction <= 5.0:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0
    return nice_fraction * (10 ** exponent)


def make_y_ticks(min_value: float, max_value: float, tick_count: int = 6) -> list[float]:
    if min_value == max_value:
        return [min_value]

    span = nice_number(max_value - min_value, False)
    step = nice_number(span / max(tick_count - 1, 1), True)
    nice_min = math.floor(min_value / step) * step
    nice_max = math.ceil(max_value / step) * step

    ticks: list[float] = []
    current = nice_min
    while current <= nice_max + 0.5 * step:
        ticks.append(current)
        current += step
    return ticks


def load_results(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"input json not found: {path}")
    return json.loads(path.read_text())


def build_series(results: dict[str, object]) -> tuple[dict[str, list[tuple[int, float, str]]], dict[str, int], str]:
    benchmarks = results.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise RuntimeError("results json is missing a top-level 'benchmarks' list")

    series_map = {method: {} for method in METHOD_ORDER}
    model_shapes: set[tuple[int, int, int, int]] = set()

    for entry in benchmarks:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "pass":
            continue

        method = entry.get("method")
        preset = entry.get("preset")
        if method not in METHOD_ORDER or not isinstance(preset, str):
            continue
        if not preset.startswith(PRESET_PREFIX):
            continue

        preset_details = entry.get("preset_details")
        parsed = entry.get("parsed")
        if not isinstance(preset_details, dict) or not isinstance(parsed, dict):
            continue

        steps = preset_details.get("steps")
        if steps is None:
            steps = parsed.get("steps")
        if steps is None:
            raise RuntimeError(f"missing steps for method={method} preset={preset}")
        x_value = int(steps)

        metric_value = parsed.get(Y_METRIC_KEY)
        if metric_value is None:
            raise RuntimeError(f"missing parsed.{Y_METRIC_KEY} for method={method} preset={preset}")
        y_value = parse_float(metric_value)

        if x_value in series_map[method]:
            raise RuntimeError(f"duplicate point for method={method} at steps={x_value}")
        series_map[method][x_value] = (y_value, preset)

        model_shapes.add(
            (
                int(preset_details["n_layer"]),
                int(preset_details["n_embd"]),
                int(preset_details["block_size"]),
                int(preset_details["n_head"]),
            )
        )

    if VERIFY_FIXED_MODEL and len(model_shapes) != 1:
        raise RuntimeError(
            "selected presets do not keep model size fixed; found model shapes: "
            + ", ".join(str(shape) for shape in sorted(model_shapes))
        )

    x_values = sorted({x_value for points in series_map.values() for x_value in points})
    if not x_values:
        raise RuntimeError(f"no passing presets found for prefix '{PRESET_PREFIX}' in {INPUT_JSON}")

    if FAIL_IF_MISSING_POINTS:
        missing: list[str] = []
        for method in METHOD_ORDER:
            for x_value in x_values:
                if x_value not in series_map[method]:
                    missing.append(f"{method}@{x_value}")
        if missing:
            preview = ", ".join(missing[:12])
            suffix = "" if len(missing) <= 12 else f", ... ({len(missing)} total)"
            raise RuntimeError(f"missing method/preset points in json: {preview}{suffix}")

    series = {
        method: [
            (x_value, series_map[method][x_value][0], series_map[method][x_value][1])
            for x_value in sorted(series_map[method])
        ]
        for method in METHOD_ORDER
        if series_map[method]
    }

    subtitle = SUBTITLE
    if not subtitle and model_shapes:
        n_layer, n_embd, block_size, n_head = next(iter(model_shapes))
        subtitle = (
            f"Fixed model: n_layer={n_layer}, n_embd={n_embd}, "
            f"block_size={block_size}, n_head={n_head}"
        )

    return series, {x_value: x_value for x_value in x_values}, subtitle


def render_svg(
    series: dict[str, list[tuple[int, float, str]]],
    x_values: list[int],
    subtitle: str,
) -> str:
    left = 110
    right = 280
    top = 110
    bottom = 110

    plot_width = SVG_WIDTH - left - right
    plot_height = SVG_HEIGHT - top - bottom

    x_min = min(x_values)
    x_max = max(x_values)
    y_values = [point[1] for points in series.values() for point in points]
    y_min = min(y_values)
    y_max = max(y_values)

    if FORCE_Y_ZERO:
        y_min = 0.0

    if y_min == y_max:
        y_max = y_min + 1.0

    y_ticks = make_y_ticks(y_min, y_max)
    if y_ticks:
        y_min = min(y_ticks)
        y_max = max(y_ticks)

    def x_to_px(x_value: int) -> float:
        if x_max == x_min:
            return left + plot_width / 2.0
        return left + (x_value - x_min) / (x_max - x_min) * plot_width

    def y_to_px(y_value: float) -> float:
        return top + plot_height - (y_value - y_min) / (y_max - y_min) * plot_height

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" '
        f'viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">'
    )
    parts.append(
        f'<rect x="0" y="0" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="white" />'
    )
    parts.append(
        f'<text x="{left}" y="44" font-size="28" font-weight="700" '
        f'font-family="{escape_xml(FONT_FAMILY)}" fill="#111827">{escape_xml(TITLE)}</text>'
    )
    if subtitle:
        parts.append(
            f'<text x="{left}" y="72" font-size="16" '
            f'font-family="{escape_xml(FONT_FAMILY)}" fill="#4b5563">{escape_xml(subtitle)}</text>'
        )

    for tick in y_ticks:
        y_px = y_to_px(tick)
        parts.append(
            f'<line x1="{left}" y1="{y_px:.2f}" x2="{left + plot_width}" y2="{y_px:.2f}" '
            f'stroke="#e5e7eb" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y_px + 5:.2f}" text-anchor="end" font-size="13" '
            f'font-family="{escape_xml(FONT_FAMILY)}" fill="#374151">{escape_xml(format_y_tick(tick))}</text>'
        )

    for x_value in x_values:
        x_px = x_to_px(x_value)
        parts.append(
            f'<line x1="{x_px:.2f}" y1="{top}" x2="{x_px:.2f}" y2="{top + plot_height}" '
            f'stroke="#f3f4f6" stroke-width="1" />'
        )
        label = format_name_count(x_value)
        parts.append(
            f'<text x="{x_px:.2f}" y="{top + plot_height + 22}" text-anchor="end" '
            f'transform="rotate(-35 {x_px:.2f} {top + plot_height + 22})" font-size="12" '
            f'font-family="{escape_xml(FONT_FAMILY)}" fill="#374151">{escape_xml(label)}</text>'
        )

    parts.append(
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" '
        f'stroke="#111827" stroke-width="1.5" />'
    )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        f'stroke="#111827" stroke-width="1.5" />'
    )

    for method in METHOD_ORDER:
        points = series.get(method)
        if not points:
            continue
        color = METHOD_COLORS.get(method, "#111827")
        point_text = " ".join(f"{x_to_px(x_value):.2f},{y_to_px(y_value):.2f}" for x_value, y_value, _ in points)
        parts.append(
            f'<polyline fill="none" stroke="{escape_xml(color)}" stroke-width="3" '
            f'points="{point_text}" />'
        )
        for x_value, y_value, preset in points:
            x_px = x_to_px(x_value)
            y_px = y_to_px(y_value)
            parts.append(
                f'<circle cx="{x_px:.2f}" cy="{y_px:.2f}" r="4.5" fill="{escape_xml(color)}">'
                f"<title>{escape_xml(f'{METHOD_LABELS.get(method, method)} | {preset} | {Y_METRIC_KEY}={y_value:.6f}')}"
                f"</title></circle>"
            )

    parts.append(
        f'<text x="{left + plot_width / 2:.2f}" y="{SVG_HEIGHT - 24}" text-anchor="middle" '
        f'font-size="16" font-family="{escape_xml(FONT_FAMILY)}" fill="#111827">{escape_xml(X_LABEL)}</text>'
    )
    parts.append(
        f'<text x="28" y="{top + plot_height / 2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 28 {top + plot_height / 2:.2f})" font-size="16" '
        f'font-family="{escape_xml(FONT_FAMILY)}" fill="#111827">{escape_xml(Y_LABEL)}</text>'
    )

    legend_x = left + plot_width + 30
    legend_y = top + 10
    parts.append(
        f'<text x="{legend_x}" y="{legend_y}" font-size="16" font-weight="700" '
        f'font-family="{escape_xml(FONT_FAMILY)}" fill="#111827">{escape_xml(LEGEND_TITLE)}</text>'
    )
    for index, method in enumerate(METHOD_ORDER):
        if method not in series:
            continue
        y_row = legend_y + 26 + index * 28
        color = METHOD_COLORS.get(method, "#111827")
        label = METHOD_LABELS.get(method, method)
        parts.append(
            f'<line x1="{legend_x}" y1="{y_row}" x2="{legend_x + 22}" y2="{y_row}" '
            f'stroke="{escape_xml(color)}" stroke-width="3" />'
        )
        parts.append(
            f'<circle cx="{legend_x + 11}" cy="{y_row}" r="4" fill="{escape_xml(color)}" />'
        )
        parts.append(
            f'<text x="{legend_x + 32}" y="{y_row + 5}" font-size="14" '
            f'font-family="{escape_xml(FONT_FAMILY)}" fill="#111827">{escape_xml(label)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    input_path = Path(INPUT_JSON)
    output_path = Path(OUTPUT_FILE)

    results = load_results(input_path)
    series, x_value_lookup, subtitle = build_series(results)
    svg = render_svg(series, sorted(x_value_lookup), subtitle)

    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
