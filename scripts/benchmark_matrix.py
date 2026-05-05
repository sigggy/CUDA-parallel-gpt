#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BenchmarkRun:
    label: str
    n_layer: int
    n_embd: int
    block_size: int
    n_head: int
    steps: int


BENCHMARK_RUNS: tuple[BenchmarkRun, ...] = (
    BenchmarkRun("small", n_layer=1, n_embd=64, block_size=128, n_head=4, steps=200),
    BenchmarkRun("medium", n_layer=2, n_embd=128, block_size=128, n_head=8, steps=1000),
    BenchmarkRun("large", n_layer=4, n_embd=256, block_size=256, n_head=8, steps=2500),
    BenchmarkRun("very-large", n_layer=6, n_embd=384, block_size=512, n_head=12, steps=5000),
    BenchmarkRun("extra-large", n_layer=8, n_embd=512, block_size=512, n_head=16, steps=10000),
    BenchmarkRun("names-1k",    n_layer=1, n_embd=64, block_size=512, n_head=4, steps=1000),
    BenchmarkRun("names-2k",    n_layer=1, n_embd=64, block_size=512, n_head=4, steps=2000),
    BenchmarkRun("names-3k",    n_layer=1, n_embd=64, block_size=512, n_head=4, steps=3000),
    BenchmarkRun("names-4k",    n_layer=1, n_embd=64, block_size=512, n_head=4, steps=4000),
    BenchmarkRun("names-6k",    n_layer=1, n_embd=64, block_size=512, n_head=4, steps=6000),
    BenchmarkRun("names-8k",    n_layer=1, n_embd=64, block_size=512, n_head=4, steps=8000),
    BenchmarkRun("names-10k",   n_layer=1, n_embd=64, block_size=512, n_head=4, steps=10000),
    BenchmarkRun("names-12k",   n_layer=1, n_embd=64, block_size=512, n_head=4, steps=12000),
    BenchmarkRun("names-16k",   n_layer=1, n_embd=64, block_size=512, n_head=4, steps=16000),
    BenchmarkRun("names-20k",   n_layer=1, n_embd=64, block_size=512, n_head=4, steps=20000),
    BenchmarkRun("names-24k",   n_layer=1, n_embd=64, block_size=512, n_head=4, steps=24000),
    BenchmarkRun("names-28k",   n_layer=1, n_embd=64, block_size=512, n_head=4, steps=28000),
    BenchmarkRun("names-30k",   n_layer=1, n_embd=64, block_size=512, n_head=4, steps=30000),
    BenchmarkRun("names-31k",   n_layer=1, n_embd=64, block_size=512, n_head=4, steps=31000),
    BenchmarkRun("names-32k",   n_layer=1, n_embd=64, block_size=512, n_head=4, steps=32000),
    BenchmarkRun("model-small-1k", n_layer=1, n_embd=64, block_size=512, n_head=4, steps=5000),
    BenchmarkRun("model-medium-1k", n_layer=2, n_embd=128, block_size=512, n_head=8, steps=5000),
    BenchmarkRun("model-large-1k", n_layer=4, n_embd=256, block_size=512, n_head=8, steps=5000),
    BenchmarkRun("model-very-large-1k", n_layer=6, n_embd=384, block_size=512, n_head=12, steps=5000),
    BenchmarkRun("model-extra-large-1k", n_layer=8, n_embd=512, block_size=512, n_head=16, steps=5000),
)

RUN_BY_LABEL = {run.label: run for run in BENCHMARK_RUNS}
METHOD_RUN_FILTERS = {
    "serial_python": {"small", "medium"},
}


def run_details() -> dict[str, dict[str, int]]:
    return {
        run.label: {
            "n_layer": run.n_layer,
            "n_embd": run.n_embd,
            "block_size": run.block_size,
            "n_head": run.n_head,
            "steps": run.steps,
        }
        for run in BENCHMARK_RUNS
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="inspect the shared benchmark matrix")
    parser.add_argument(
        "--format",
        choices=("tsv", "json"),
        default="tsv",
        help="output format for the shared benchmark matrix",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.format == "json":
        import json

        print(json.dumps([asdict(run) for run in BENCHMARK_RUNS], indent=2))
        return 0

    for run in BENCHMARK_RUNS:
        print(
            "\t".join(
                (
                    run.label,
                    str(run.n_layer),
                    str(run.n_embd),
                    str(run.block_size),
                    str(run.n_head),
                    str(run.steps),
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
