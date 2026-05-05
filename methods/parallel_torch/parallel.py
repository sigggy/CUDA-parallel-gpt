#!/usr/bin/env python3
"""
PyTorch batched forward implementation.

This mirrors the serial Python/C++ model math, but evaluates full fixed-length
batches at once with torch tensor operations.
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "training_data" / "datasets" / "names.txt"
DEFAULT_FIXTURE_DIR = REPO_ROOT / "training_data" / "fixtures" / "small_case"
DEFAULT_SEED = 42
VALIDATION_EPSILON = 1e-4


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int = 1
    n_embd: int = 16
    block_size: int = 16
    n_head: int = 4

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def mlp_dim(self) -> int:
        return 4 * self.n_embd


@dataclass
class TokenBatch:
    tokens: object
    batch_size: int
    batch_seq_length: int


@dataclass
class TorchModel:
    config: ModelConfig
    vocab_size: int
    wte: object
    wpe: object
    lm_head: object
    layers: list[dict[str, object]]


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch methods require the torch package. Install torch, then rerun this command."
        )


def choose_device(device_name: str):
    require_torch()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false")
    return torch.device(device_name)


def require_positive_int(value: int | None, name: str) -> int:
    if value is None or value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def config_from_args(args: argparse.Namespace) -> tuple[ModelConfig, int, str]:
    n_layer = require_positive_int(args.n_layer, "--n-layer")
    n_embd = require_positive_int(args.n_embd, "--n-embd")
    block_size = require_positive_int(args.block_size, "--block-size")
    n_head = require_positive_int(args.n_head, "--n-head")
    num_steps = require_positive_int(args.num_steps, "--num-steps")
    if n_embd % n_head != 0:
        raise ValueError("--n-embd must be divisible by --n-head")
    return ModelConfig(n_layer=n_layer, n_embd=n_embd, block_size=block_size, n_head=n_head), num_steps, args.label


def load_docs(dataset_path: Path) -> list[str]:
    with open(dataset_path) as handle:
        return [line.strip() for line in handle if line.strip()]


def build_vocab(docs: list[str]) -> tuple[list[str], dict[str, int], int]:
    uchars = sorted(set("".join(docs)))
    bos = len(uchars)
    vocab = {ch: idx for idx, ch in enumerate(uchars)}
    return uchars, vocab, bos


def encode_doc(doc: str, vocab: dict[str, int], bos: int) -> list[int]:
    return [bos] + [vocab[ch] for ch in doc] + [bos]


def read_f32_file(path: Path) -> list[float]:
    data = path.read_bytes()
    return list(struct.unpack("<" + ("f" * (len(data) // 4)), data))


def parse_manifest(manifest_path: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for line in manifest_path.read_text().splitlines():
        if not line:
            continue
        key, value = line.split("=", 1)
        manifest[key] = value
    return manifest


def parse_int_list(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def compare_arrays(label: str, actual: list[float], expected: list[float], epsilon: float) -> None:
    if len(actual) != len(expected):
        raise ValueError(f"{label} size mismatch")
    max_abs_error = 0.0
    max_idx = 0
    for idx, (left, right) in enumerate(zip(actual, expected)):
        abs_error = abs(left - right)
        if abs_error > max_abs_error:
            max_abs_error = abs_error
            max_idx = idx
    print(f"{label} max_abs_error={max_abs_error:.8g} at_index={max_idx}")
    if max_abs_error > epsilon:
        raise ValueError(f"{label} exceeded validation epsilon")


def random_tensor(rng: random.Random, shape: tuple[int, ...], device) -> object:
    count = math.prod(shape)
    values = [rng.gauss(0.0, 0.08) for _ in range(count)]
    return torch.tensor(values, dtype=torch.float32, device=device).reshape(shape)


def initialize_model(config: ModelConfig, vocab_size: int, seed: int, device) -> TorchModel:
    rng = random.Random(seed)
    wte = random_tensor(rng, (vocab_size, config.n_embd), device)
    wpe = random_tensor(rng, (config.block_size, config.n_embd), device)
    lm_head = random_tensor(rng, (vocab_size, config.n_embd), device)
    layers: list[dict[str, object]] = []
    for _ in range(config.n_layer):
        layers.append(
            {
                "attn_wq": random_tensor(rng, (config.n_embd, config.n_embd), device),
                "attn_wk": random_tensor(rng, (config.n_embd, config.n_embd), device),
                "attn_wv": random_tensor(rng, (config.n_embd, config.n_embd), device),
                "attn_wo": random_tensor(rng, (config.n_embd, config.n_embd), device),
                "mlp_fc1": random_tensor(rng, (config.mlp_dim, config.n_embd), device),
                "mlp_fc2": random_tensor(rng, (config.n_embd, config.mlp_dim), device),
            }
        )
    return TorchModel(
        config=config,
        vocab_size=vocab_size,
        wte=wte,
        wpe=wpe,
        lm_head=lm_head,
        layers=layers,
    )


def load_model_from_f32(config: ModelConfig, vocab_size: int, values: list[float], device) -> TorchModel:
    source = torch.tensor(values, dtype=torch.float32, device=device)
    cursor = 0

    def take(shape: tuple[int, ...]):
        nonlocal cursor
        count = math.prod(shape)
        if cursor + count > source.numel():
            raise ValueError("weights file is smaller than expected")
        tensor = source[cursor : cursor + count].reshape(shape).clone()
        cursor += count
        return tensor

    wte = take((vocab_size, config.n_embd))
    wpe = take((config.block_size, config.n_embd))
    lm_head = take((vocab_size, config.n_embd))
    layers: list[dict[str, object]] = []
    for _ in range(config.n_layer):
        layers.append(
            {
                "attn_wq": take((config.n_embd, config.n_embd)),
                "attn_wk": take((config.n_embd, config.n_embd)),
                "attn_wv": take((config.n_embd, config.n_embd)),
                "attn_wo": take((config.n_embd, config.n_embd)),
                "mlp_fc1": take((config.mlp_dim, config.n_embd)),
                "mlp_fc2": take((config.n_embd, config.mlp_dim)),
            }
        )
    if cursor != source.numel():
        raise ValueError("weights file is larger than expected")
    return TorchModel(config, vocab_size, wte, wpe, lm_head, layers)


def rmsnorm(x):
    mean_square = (x * x).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(mean_square + 1e-5)


def linear(x, weights):
    return torch.matmul(x, weights.transpose(-1, -2))


def run_forward_batched(model: TorchModel, batch: TokenBatch) -> dict[str, object]:
    config = model.config
    usable_seq_len = min(config.block_size, batch.batch_seq_length - 1)
    input_tokens = batch.tokens[:, :usable_seq_len]
    target_tokens = batch.tokens[:, 1 : usable_seq_len + 1]
    positions = torch.arange(usable_seq_len, device=batch.tokens.device)

    x = model.wte[input_tokens] + model.wpe[positions].unsqueeze(0)
    x = rmsnorm(x)

    for layer in model.layers:
        x_residual = x
        x_norm1 = rmsnorm(x)
        q = linear(x_norm1, layer["attn_wq"])
        k = linear(x_norm1, layer["attn_wk"])
        v = linear(x_norm1, layer["attn_wv"])

        q = q.reshape(batch.batch_size, usable_seq_len, config.n_head, config.head_dim)
        k = k.reshape(batch.batch_size, usable_seq_len, config.n_head, config.head_dim)
        v = v.reshape(batch.batch_size, usable_seq_len, config.n_head, config.head_dim)

        attn_logits = torch.einsum("bqhd,bkhd->bhqk", q, k)
        attn_logits = attn_logits / math.sqrt(float(config.head_dim))
        causal_mask = torch.triu(
            torch.ones((usable_seq_len, usable_seq_len), dtype=torch.bool, device=batch.tokens.device),
            diagonal=1,
        )
        attn_logits = attn_logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_out = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        attn_out = attn_out.reshape(batch.batch_size, usable_seq_len, config.n_embd)

        attn_proj = linear(attn_out, layer["attn_wo"])
        x_mid = x_residual + attn_proj
        x_norm2 = rmsnorm(x_mid)
        mlp_hidden = torch.relu(linear(x_norm2, layer["mlp_fc1"]))
        fc2 = linear(mlp_hidden, layer["mlp_fc2"])
        x = x_mid + fc2

    logits = linear(x, model.lm_head)
    loss = F.cross_entropy(
        logits.reshape(batch.batch_size * usable_seq_len, model.vocab_size),
        target_tokens.reshape(batch.batch_size * usable_seq_len),
        reduction="mean",
    )
    return {"seq_len": usable_seq_len, "logits": logits, "loss": loss}


def make_repeated_batch(tokens: list[int], batch_size: int, device) -> TokenBatch:
    rows = [tokens for _ in range(batch_size)]
    tensor = torch.tensor(rows, dtype=torch.long, device=device)
    return TokenBatch(tensor, batch_size, len(tokens))


def build_length_bucketed_batches(
    docs: list[str],
    doc_count: int,
    vocab: dict[str, int],
    bos: int,
    max_batch_size: int,
    device,
) -> list[TokenBatch]:
    batches: list[TokenBatch] = []
    buckets: dict[int, list[list[int]]] = {}
    length_order: list[int] = []

    def emit(seq_length: int) -> None:
        rows = buckets[seq_length]
        tensor = torch.tensor(rows, dtype=torch.long, device=device)
        batches.append(TokenBatch(tensor, len(rows), seq_length))
        buckets[seq_length] = []

    for doc in docs[:doc_count]:
        tokens = encode_doc(doc, vocab, bos)
        seq_length = len(tokens)
        if seq_length not in buckets:
            buckets[seq_length] = []
            length_order.append(seq_length)
        buckets[seq_length].append(tokens)
        if len(buckets[seq_length]) == max_batch_size:
            emit(seq_length)

    for seq_length in length_order:
        if buckets[seq_length]:
            emit(seq_length)

    return batches


def flatten_logits(logits) -> list[float]:
    return logits.detach().cpu().reshape(-1).tolist()


def validate_fixture(fixture_dir: Path, seed: int, batch_size: int, device_name: str) -> None:
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
    batch = make_repeated_batch(tokens, batch_size, device)
    with torch.no_grad():
        result = run_forward_batched(model, batch)

    expected_logits = read_f32_file(fixture_dir / manifest["expected_logits_file"]) * batch_size
    compare_arrays("logits", flatten_logits(result["logits"]), expected_logits, epsilon)
    compare_arrays("loss", [float(result["loss"].detach().cpu())], read_f32_file(fixture_dir / manifest["expected_loss_file"]), epsilon)
    print(f"validation=pass device={device} batch_size={batch_size}")


def run_benchmark(
    dataset_path: Path,
    seed: int,
    config: ModelConfig,
    num_steps: int,
    label: str,
    batch_size: int,
    device_name: str,
) -> None:
    benchmark_start = time.perf_counter()
    device = choose_device(device_name)
    docs = load_docs(dataset_path)
    if not docs:
        raise ValueError(f"dataset is empty: {dataset_path}")
    uchars, vocab, bos = build_vocab(docs)
    requested_steps = num_steps
    steps = min(num_steps, len(docs))
    batches = build_length_bucketed_batches(docs, steps, vocab, bos, batch_size, device)
    model = initialize_model(config, len(uchars) + 1, seed, device)

    last_loss = 0.0
    weighted_loss_sum = 0.0
    loss_item_count = 0
    forward_pass_seconds_cumulative = 0.0
    last_doc = docs[steps - 1] if steps > 0 else ""

    with torch.no_grad():
        for batch in batches:
            forward_start = time.perf_counter()
            result = run_forward_batched(model, batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_pass_seconds_cumulative += time.perf_counter() - forward_start

            last_loss = float(result["loss"].detach().cpu())
            loss_items = batch.batch_size * result["seq_len"]
            weighted_loss_sum += last_loss * loss_items
            loss_item_count += loss_items

    mean_loss = weighted_loss_sum / loss_item_count if loss_item_count else 0.0
    total_program_seconds = time.perf_counter() - benchmark_start
    print(
        "mode=benchmark "
        "method=batched_torch "
        f"device={device} "
        f"preset={label} "
        f"requested_steps={requested_steps} "
        f"steps={steps} "
        f"batch_size={batch_size} "
        f"batches={len(batches)} "
        f"last_doc={last_doc} "
        f"loss={last_loss:.8f} "
        f"mean_loss={mean_loss:.8f} "
        f"forward_pass_seconds_cumulative={forward_pass_seconds_cumulative:.8f} "
        f"total_program_seconds={total_program_seconds:.8f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="batched PyTorch benchmark scaffold")
    parser.add_argument("--mode", choices=["validate", "benchmark"], required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--label", default="custom")
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "validate":
            validate_fixture(DEFAULT_FIXTURE_DIR, args.seed, 1, args.device)
            return 0
        config, num_steps, label = config_from_args(args)
        run_benchmark(args.dataset, args.seed, config, num_steps, label, args.batch_size, args.device)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
