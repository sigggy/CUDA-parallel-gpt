#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path


# In-file config only. This script intentionally has no CLI.
INPUT_JSON = "data/benchmark_results_new2.json"
OUTPUT_FILE = "report/figures/general_benchmark_names.png"

PRESET_PREFIX = "names-"
Y_METRIC_KEY = "total_program_seconds"
MODEL_SHAPE_FILTER = None
# Optional shape filter for mixed-shape benchmark files.
# Order: (n_layer, n_embd, block_size, n_head).
# Leave this as None when the input already contains a single fixed model shape.

TITLE = "General Benchmark Methods Across Increasing Name Counts"
SUBTITLE = ""
X_LABEL = "Number of Names"
Y_LABEL = "Runtime (seconds)"
LEGEND_TITLE = "Methods"

METHOD_ORDER = [
    "serial_cpp",
    "unbatched_torch",
    "batched_torch",
    "parallel_cpp",
]

METHOD_LABELS = {
    "serial_cpp": "SC",
    "unbatched_torch": "UT",
    "batched_torch": "BT",
    "parallel_cpp": "PC",
}

METHOD_ALIASES = {
    "serial_cpp": ("serial_cpp",),
    "unbatched_torch": ("unbatched_torch", "serial_torch"),
    "batched_torch": ("batched_torch", "parallel_torch"),
    "parallel_cpp": ("parallel_cpp",),
}

METHOD_COLORS = {
    "serial_cpp": "#111827",
    "unbatched_torch": "#2563eb",
    "batched_torch": "#059669",
    "parallel_cpp": "#b91c1c",
}

METHOD_MARKERS = {
    "serial_cpp": "circle",
    "unbatched_torch": "triangle_up",
    "batched_torch": "hexagon",
    "parallel_cpp": "square",
}

FAIL_IF_MISSING_POINTS = True
VERIFY_FIXED_MODEL = True
FORCE_Y_ZERO = False
USE_LOG_Y = True

SVG_WIDTH = 1280
SVG_HEIGHT = 820
FONT_FAMILY = "Helvetica, Arial, sans-serif"
X_TICK_FONT_SIZE = 16
Y_TICK_FONT_SIZE = 15
AXIS_LABEL_FONT_SIZE = 18
LEGEND_FONT_SIZE = 16


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


def regular_polygon_points(
    cx: float,
    cy: float,
    radius: float,
    sides: int,
    rotation_degrees: float = 0.0,
) -> str:
    points: list[str] = []
    rotation_radians = math.radians(rotation_degrees)
    for index in range(sides):
        angle = rotation_radians + 2.0 * math.pi * index / sides
        x_value = cx + radius * math.cos(angle)
        y_value = cy + radius * math.sin(angle)
        points.append(f"{x_value:.2f},{y_value:.2f}")
    return " ".join(points)


def render_marker(
    marker: str,
    cx: float,
    cy: float,
    size: float,
    color: str,
    title: str | None = None,
) -> str:
    title_part = f"<title>{escape_xml(title)}</title>" if title else ""
    common_attrs = f'fill="{escape_xml(color)}" stroke="white" stroke-width="1.2"'

    if marker == "circle":
        shape = f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{size:.2f}" {common_attrs} />'
    elif marker == "square":
        shape = (
            f'<rect x="{cx - size:.2f}" y="{cy - size:.2f}" width="{2.0 * size:.2f}" '
            f'height="{2.0 * size:.2f}" {common_attrs} />'
        )
    elif marker == "diamond":
        shape = (
            f'<polygon points="{regular_polygon_points(cx, cy, size * 1.15, 4, -90.0)}" '
            f'{common_attrs} />'
        )
    elif marker == "triangle_up":
        shape = (
            f'<polygon points="{regular_polygon_points(cx, cy, size * 1.25, 3, -90.0)}" '
            f'{common_attrs} />'
        )
    elif marker == "triangle_down":
        shape = (
            f'<polygon points="{regular_polygon_points(cx, cy, size * 1.25, 3, 90.0)}" '
            f'{common_attrs} />'
        )
    elif marker == "hexagon":
        shape = (
            f'<polygon points="{regular_polygon_points(cx, cy, size * 1.05, 6, -90.0)}" '
            f'{common_attrs} />'
        )
    else:
        raise RuntimeError(f"unsupported marker shape: {marker}")

    return f"<g>{title_part}{shape}</g>"


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


def make_log_y_ticks(min_value: float, max_value: float) -> list[float]:
    if min_value <= 0.0 or max_value <= 0.0:
        raise RuntimeError("log-scale y-axis requires all y values to be > 0")
    min_exp = math.floor(math.log10(min_value))
    max_exp = math.ceil(math.log10(max_value))
    return [10.0 ** exponent for exponent in range(min_exp, max_exp + 1)]


def load_results(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"input json not found: {path}")
    return json.loads(path.read_text())


def write_png_from_svg(svg: str, output_path: Path) -> None:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError("rsvg-convert is required to write PNG output")

    with tempfile.TemporaryDirectory(prefix="general-benchmark-plot-") as temp_dir:
        temp_svg = Path(temp_dir) / "plot.svg"
        temp_svg.write_text(svg)
        completed = subprocess.run(
            [
                converter,
                str(temp_svg),
                "-o",
                str(output_path),
                "-w",
                str(SVG_WIDTH),
                "-h",
                str(SVG_HEIGHT),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "failed to convert svg to png with rsvg-convert: "
                + (completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}")
            )


def build_method_alias_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical_method, aliases in METHOD_ALIASES.items():
        for alias in aliases:
            lookup[alias] = canonical_method
    return lookup


def build_series(results: dict[str, object]) -> tuple[dict[str, list[tuple[int, float, str]]], list[int], str]:
    benchmarks = results.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise RuntimeError("results json is missing a top-level 'benchmarks' list")

    alias_lookup = build_method_alias_lookup()
    series_map = {method: {} for method in METHOD_ORDER}
    model_shapes: set[tuple[int, int, int, int]] = set()

    for entry in benchmarks:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "pass":
            continue

        raw_method = entry.get("method")
        preset = entry.get("preset")
        if not isinstance(raw_method, str) or not isinstance(preset, str):
            continue
        method = alias_lookup.get(raw_method)
        if method not in METHOD_ORDER:
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
            raise RuntimeError(f"missing steps for method={raw_method} preset={preset}")
        x_value = int(steps)

        metric_value = parsed.get(Y_METRIC_KEY)
        if metric_value is None:
            raise RuntimeError(f"missing parsed.{Y_METRIC_KEY} for method={raw_method} preset={preset}")
        y_value = parse_float(metric_value)

        current_shape = (
            int(preset_details["n_layer"]),
            int(preset_details["n_embd"]),
            int(preset_details["block_size"]),
            int(preset_details["n_head"]),
        )
        model_shapes.add(current_shape)
        if MODEL_SHAPE_FILTER is not None and current_shape != MODEL_SHAPE_FILTER:
            continue

        if x_value in series_map[method]:
            raise RuntimeError(
                f"duplicate point for canonical method={method} at steps={x_value}; "
                f"check for mixed old/new method names in the input json"
            )
        series_map[method][x_value] = (y_value, preset)

    selected_shapes = set(model_shapes)
    if MODEL_SHAPE_FILTER is not None:
        selected_shapes = {shape for shape in model_shapes if shape == MODEL_SHAPE_FILTER}
        if not selected_shapes:
            available_shapes = ", ".join(str(shape) for shape in sorted(model_shapes)) or "none"
            raise RuntimeError(
                "no names-* rows matched MODEL_SHAPE_FILTER="
                f"{MODEL_SHAPE_FILTER} in {INPUT_JSON}; available shapes: {available_shapes}"
            )

    if VERIFY_FIXED_MODEL and len(selected_shapes) != 1:
        raise RuntimeError(
            "selected presets do not keep model size fixed; found model shapes: "
            + ", ".join(str(shape) for shape in sorted(selected_shapes))
            + ". Set MODEL_SHAPE_FILTER to one of those tuples."
        )

    x_values = sorted({x_value for points in series_map.values() for x_value in points})
    if not x_values:
        raise RuntimeError(
            f"no passing presets found for prefix '{PRESET_PREFIX}' in {INPUT_JSON}; "
            "make sure the input file is from a general benchmark run that includes names-* presets"
        )

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
    if not subtitle and selected_shapes:
        n_layer, n_embd, block_size, n_head = next(iter(selected_shapes))
        subtitle = (
            f"Fixed model: n_layer={n_layer}, n_embd={n_embd}, "
            f"block_size={block_size}, n_head={n_head}"
        )

    return series, x_values, subtitle


def render_svg(
    series: dict[str, list[tuple[int, float, str]]],
    x_values: list[int],
    subtitle: str,
) -> str:
    left = 110
    right = 70
    top = 110
    bottom = 170

    plot_width = SVG_WIDTH - left - right
    plot_height = SVG_HEIGHT - top - bottom

    x_min = min(x_values)
    x_max = max(x_values)
    y_values = [point[1] for points in series.values() for point in points]
    y_min = min(y_values)
    y_max = max(y_values)

    if USE_LOG_Y and any(value <= 0.0 for value in y_values):
        raise RuntimeError("cannot plot non-positive values on a log-scale y-axis")

    if FORCE_Y_ZERO and not USE_LOG_Y:
        y_min = 0.0

    if USE_LOG_Y:
        if y_min == y_max:
            y_min /= 10.0
            y_max *= 10.0
        y_ticks = make_log_y_ticks(y_min, y_max)
        if y_ticks:
            y_min = min(y_ticks)
            y_max = max(y_ticks)
        log_y_min = math.log10(y_min)
        log_y_max = math.log10(y_max)
    else:
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
        if USE_LOG_Y:
            return top + plot_height - (math.log10(y_value) - log_y_min) / (log_y_max - log_y_min) * plot_height
        return top + plot_height - (y_value - y_min) / (y_max - y_min) * plot_height

    present_methods = [method for method in METHOD_ORDER if method in series]

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
            f'<text x="{left - 12}" y="{y_px + 5:.2f}" text-anchor="end" font-size="{Y_TICK_FONT_SIZE}" '
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
            f'transform="rotate(-35 {x_px:.2f} {top + plot_height + 22})" font-size="{X_TICK_FONT_SIZE}" '
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
        marker = METHOD_MARKERS.get(method, "circle")
        point_text = " ".join(f"{x_to_px(x_value):.2f},{y_to_px(y_value):.2f}" for x_value, y_value, _ in points)
        parts.append(
            f'<polyline fill="none" stroke="{escape_xml(color)}" stroke-width="3" '
            f'points="{point_text}" />'
        )
        for x_value, y_value, preset in points:
            x_px = x_to_px(x_value)
            y_px = y_to_px(y_value)
            parts.append(
                render_marker(
                    marker=marker,
                    cx=x_px,
                    cy=y_px,
                    size=4.8,
                    color=color,
                    title=f"{METHOD_LABELS.get(method, method)} | {preset} | {Y_METRIC_KEY}={y_value:.6f}",
                )
            )

    x_axis_label_y = top + plot_height + 78
    parts.append(
        f'<text x="{left + plot_width / 2:.2f}" y="{x_axis_label_y:.2f}" text-anchor="middle" '
        f'font-size="{AXIS_LABEL_FONT_SIZE}" font-family="{escape_xml(FONT_FAMILY)}" fill="#111827">{escape_xml(X_LABEL)}</text>'
    )
    parts.append(
        f'<text x="28" y="{top + plot_height / 2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 28 {top + plot_height / 2:.2f})" font-size="{AXIS_LABEL_FONT_SIZE}" '
        f'font-family="{escape_xml(FONT_FAMILY)}" fill="#111827">{escape_xml(Y_LABEL)}</text>'
    )

    legend_y = top + plot_height + 122
    legend_item_width = plot_width / max(len(present_methods), 1)
    for index, method in enumerate(present_methods):
        item_x = left + index * legend_item_width + 8
        color = METHOD_COLORS.get(method, "#111827")
        label = METHOD_LABELS.get(method, method)
        marker = METHOD_MARKERS.get(method, "circle")
        parts.append(
            f'<line x1="{item_x:.2f}" y1="{legend_y:.2f}" x2="{item_x + 22:.2f}" y2="{legend_y:.2f}" '
            f'stroke="{escape_xml(color)}" stroke-width="3" />'
        )
        parts.append(render_marker(marker, item_x + 11, legend_y, 4.2, color))
        parts.append(
            f'<text x="{item_x + 32:.2f}" y="{legend_y + 5:.2f}" font-size="{LEGEND_FONT_SIZE}" '
            f'font-family="{escape_xml(FONT_FAMILY)}" fill="#111827">{escape_xml(label)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    input_path = Path(INPUT_JSON)
    output_path = Path(OUTPUT_FILE)

    results = load_results(input_path)
    series, x_values, subtitle = build_series(results)
    svg = render_svg(series, x_values, subtitle)

    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    write_png_from_svg(svg, output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
