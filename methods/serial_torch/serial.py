#!/usr/bin/env python3
"""
PyTorch single-name forward implementation.

This uses the same tensor math as parallel_torch, but the benchmark loop feeds
one tokenized name at a time to mirror the serial implementations.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

METHODS_DIR = Path(__file__).resolve().parents[1]
if str(METHODS_DIR) not in sys.path:
    sys.path.insert(0, str(METHODS_DIR))

from parallel_torch.parallel import (  # noqa: E402
    BENCHMARK_PRESETS,
    DEFAULT_DATASET,
    DEFAULT_FIXTURE_DIR,
    DEFAULT_SEED,
    ModelConfig,
    TokenBatch,
    build_vocab,
    choose_device,
    compare_arrays,
    encode_doc,
    flatten_logits,
    initialize_model,
    load_docs,
    load_model_from_f32,
    parse_int_list,
    parse_manifest,
    read_f32_file,
    run_forward_batched,
    torch,
)


def make_single_batch(tokens: list[int], device) -> TokenBatch:
    tensor = torch.tensor([tokens], dtype=torch.long, device=device)
    return TokenBatch(tensor, 1, len(tokens))


def validate_fixture(fixture_dir: Path, seed: int, device_name: str) -> None:
    del seed
    device = choose_device(device_name)
    manifest = parse_manifest(fixture_dir / "manifest.txt")
    config = ModelConfig(
        n_layer=int(manifest["n_layer"]),
        n_embd=int(manifest["n_embd"]),
        block_size=int(manifest["block_size"]),
        n_head=int(manifest["n_head"]),
    )
    vocab_size = int(manifest["vocab_size"])
    tokens = parse_int_list(manifest["token_ids"])
    epsilon = float(manifest["validation_epsilon"])
    model = load_model_from_f32(
        config,
        vocab_size,
        read_f32_file(fixture_dir / manifest["weights_init_file"]),
        device,
    )
    batch = make_single_batch(tokens, device)
    with torch.no_grad():
        result = run_forward_batched(model, batch)

    compare_arrays("logits", flatten_logits(result["logits"]), read_f32_file(fixture_dir / manifest["expected_logits_file"]), epsilon)
    compare_arrays("loss", [float(result["loss"].detach().cpu())], read_f32_file(fixture_dir / manifest["expected_loss_file"]), epsilon)
    print(f"validation=pass device={device}")


def run_benchmark(
    dataset_path: Path,
    seed: int,
    preset_name: str,
    num_steps: int | None,
    device_name: str,
) -> None:
    benchmark_start = time.perf_counter()
    device = choose_device(device_name)
    docs = load_docs(dataset_path)
    if not docs:
        raise ValueError(f"dataset is empty: {dataset_path}")
    uchars, vocab, bos = build_vocab(docs)
    preset = BENCHMARK_PRESETS[preset_name]
    config = preset["config"]()
    requested_steps = num_steps if num_steps is not None else preset["steps"]
    steps = min(requested_steps, len(docs))
    model = initialize_model(config, len(uchars) + 1, seed, device)

    last_loss = 0.0
    weighted_loss_sum = 0.0
    loss_item_count = 0
    forward_pass_seconds_cumulative = 0.0
    last_doc = ""

    with torch.no_grad():
        for step_idx in range(steps):
            last_doc = docs[step_idx]
            tokens = encode_doc(last_doc, vocab, bos)
            batch = make_single_batch(tokens, device)
            forward_start = time.perf_counter()
            result = run_forward_batched(model, batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_pass_seconds_cumulative += time.perf_counter() - forward_start

            last_loss = float(result["loss"].detach().cpu())
            loss_items = result["seq_len"]
            weighted_loss_sum += last_loss * loss_items
            loss_item_count += loss_items

    mean_loss = weighted_loss_sum / loss_item_count if loss_item_count else 0.0
    total_program_seconds = time.perf_counter() - benchmark_start
    print(
        "mode=benchmark "
        "method=serial_torch "
        f"device={device} "
        f"preset={preset_name} "
        f"requested_steps={requested_steps} "
        f"steps={steps} "
        f"last_doc={last_doc} "
        f"loss={last_loss:.8f} "
        f"mean_loss={mean_loss:.8f} "
        f"forward_pass_seconds_cumulative={forward_pass_seconds_cumulative:.8f} "
        f"total_program_seconds={total_program_seconds:.8f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="serial PyTorch benchmark scaffold")
    parser.add_argument("--mode", choices=["validate", "benchmark"], required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--preset", choices=sorted(BENCHMARK_PRESETS), default="small")
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "validate":
            validate_fixture(args.fixture_dir, args.seed, args.device)
            return 0
        run_benchmark(args.dataset, args.seed, args.preset, args.num_steps, args.device)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
