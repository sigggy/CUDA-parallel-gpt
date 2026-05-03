# Parallel GPT Project

## Commands

### Install Python dependencies

```bash
python3 -m pip install -r requirements.txt
```

The PyTorch methods require `torch`.

### Build everything

```bash
make all
```

Note: `parallel_cpp` is now a CUDA-only target. `make all` requires `nvcc` on `PATH`.

### Regenerate the validation fixture bundle

```bash
make fixtures
```

Or directly:

```bash
python3 methods/serial_python/serial.py \
  --mode dump-fixtures \
  --dataset training_data/datasets/names.txt \
  --sample-name anna
```

### Validate each method

Python reference:

```bash
python3 methods/serial_python/serial.py --mode validate
```

Serial PyTorch:

```bash
python3 methods/serial_torch/serial.py --mode validate
```

Batched PyTorch:

```bash
python3 methods/parallel_torch/parallel.py --mode validate
```

Serial C++:

```bash
build/serial_cpp --mode validate
```

Parallel C++:

```bash
build/parallel_cpp --mode validate
```

Note: `parallel_cpp` requires a CUDA build and `nvcc` on `PATH`. The CUDA target is built as C++14 for compatibility with older `nvcc` versions.

### Benchmark a single method

Python reference:

```bash
python3 methods/serial_python/serial.py \
  --mode benchmark \
  --dataset training_data/datasets/names.txt \
  --label medium \
  --num-steps 1000 \
  --n-layer 2 \
  --n-embd 128 \
  --block-size 64 \
  --n-head 8
```

Serial PyTorch:

```bash
python3 methods/serial_torch/serial.py \
  --mode benchmark \
  --dataset training_data/datasets/names.txt \
  --label small \
  --num-steps 200 \
  --n-layer 1 \
  --n-embd 64 \
  --block-size 64 \
  --n-head 4
```

Batched PyTorch:

```bash
python3 methods/parallel_torch/parallel.py \
  --mode benchmark \
  --dataset training_data/datasets/names.txt \
  --label large \
  --num-steps 2500 \
  --n-layer 4 \
  --n-embd 256 \
  --block-size 128 \
  --n-head 8 \
  --batch-size 6
```

Serial C++:

```bash
build/serial_cpp \
  --mode benchmark \
  --dataset training_data/datasets/names.txt \
  --label small \
  --num-steps 200 \
  --n-layer 1 \
  --n-embd 64 \
  --block-size 64 \
  --n-head 4
```

Parallel C++:

```bash
build/parallel_cpp \
  --mode benchmark \
  --dataset training_data/datasets/names.txt \
  --label small \
  --num-steps 200 \
  --n-layer 1 \
  --n-embd 64 \
  --block-size 64 \
  --n-head 4 \
  --batch-size 6
```

Note: `parallel_torch` and `parallel_cpp` preprocess benchmark names into fixed-length batches before launching the tensor/CUDA forward path. Each batch contains only sequences with the same token length; the final batch for a length bucket may be smaller than `--batch-size`.

The full sweep definitions live in `scripts/benchmark_matrix.py`.

If you want the short command for one named run instead of typing the full shape:

```bash
python3 scripts/run_named_benchmark.py parallel_cpp large
```

Shared benchmark labels:

- `small`
- `medium`
- `large`
- `very-large`
- `extra-large`
- `names-1k`
- `names-5k`
- `names-10k`
- `names-20k`
- `names-30k`
- `model-small-1k`
- `model-medium-1k`
- `model-large-1k`
- `model-very-large-1k`
- `model-extra-large-1k`

`small` through `extra-large` increase both model size and number of names, up to 10k names. The `names-*` presets keep the model small-to-medium while increasing the number of names, up to 30k names. The `model-*-1k` presets keep names fixed around 1k while increasing all model-size dimensions. The pure scalar `serial_python` method is only run on `small` and `medium` by the benchmark drivers.

Override the run shape directly if needed:

```bash
build/serial_cpp \
  --mode benchmark \
  --dataset training_data/datasets/names.txt \
  --label custom-10 \
  --num-steps 10 \
  --n-layer 1 \
  --n-embd 64 \
  --block-size 64 \
  --n-head 4
```

For the CUDA method, override the maximum batch size with:

```bash
build/parallel_cpp \
  --mode benchmark \
  --dataset training_data/datasets/names.txt \
  --label custom-10 \
  --num-steps 10 \
  --n-layer 1 \
  --n-embd 64 \
  --block-size 64 \
  --n-head 4 \
  --batch-size 6
```

### Run the full benchmark sweep

This rebuilds, regenerates fixtures, validates all methods, then times each valid method once.

```bash
bash scripts/run_benchmarks.sh
```

### Run the Python benchmark driver

The Python benchmark driver writes structured results to JSON as it goes, writes an HTML report when the full sweep finishes, and skips unavailable optional methods such as CUDA or PyTorch when their dependencies are missing.

```bash
python3 scripts/run_benchmarks.py
```

To choose the output path:

```bash
python3 scripts/run_benchmarks.py benchmark_results.json
```

To choose both output paths:

```bash
python3 scripts/run_benchmarks.py benchmark_results.json --html-output benchmark_results.html
```

To inspect the shared benchmark matrix directly:

```bash
python3 scripts/benchmark_matrix.py --format json
```

## Repo Layout

- `methods/serial_python/kernel.py`: Python reference forward kernel.
- `methods/serial_python/serial.py`: Python runner for fixture generation, validation, and benchmarking.
- `methods/serial_torch/serial.py`: PyTorch runner that processes one tokenized name at a time.
- `methods/parallel_torch/parallel.py`: PyTorch runner that groups equal-length names into batches.
- `methods/serial_cpp/kernel.cpp`: Serial C++ forward kernel.
- `methods/serial_cpp/utils.cpp`: Serial C++ model setup and serialization helpers kept separate from kernel math.
- `methods/serial_cpp/main.cpp`: Serial C++ runner.
- `methods/parallel_cpp/kernel.cu`: CUDA-target translation unit with the forward-only batched compute path.
- `methods/parallel_cpp/utils.cpp`: Parallel C++ model setup and serialization helpers kept separate from CUDA-specific code.
- `methods/parallel_cpp/main.cpp`: Parallel method runner.
- `training_data/datasets/names.txt`: dataset used for benchmarks.
- `training_data/fixtures/small_case/`: deterministic validation data generated from Python.

## How It Works

The project is built around one narrow workload: repeated forward passes on tokenized name data with next-token loss. The point is to compare implementations of the same kernel, not to build a full training framework.

Each method has the same shape:

- a thin runner file for CLI, dataset loading, and validation/benchmark orchestration
- a kernel file containing the actual GPT forward implementation

The Python version is the reference implementation. It is used to generate deterministic ground-truth files for:

- initial weights
- expected logits
- expected scalar loss

Those files live in `training_data/fixtures/small_case/`. Validation works by loading the fixture weights, running the method’s forward pass on the fixed token sequence from the manifest, and comparing the outputs against the Python ground truth within an epsilon.

Benchmarking keeps the GPT-style data flow without the training update. Each executable loads the dataset, builds the vocabulary, initializes weights from the fixed seed, and then processes the first `k` names in dataset order, where `k` is the preset size or `--num-steps`. The serial methods run one tokenized name at a time. The batched PyTorch and CUDA methods tokenize the selected names up front, greedily group them by equal token sequence length into batches of up to `--batch-size`, then loop over the resulting batch array. The outer script uses `/usr/bin/time -p` and reports the raw wall-clock `real` time from one run. This means timing includes process startup and dataset loading for every method equally.

The current methods are:

- `serial_python`: correctness reference and optional timing reference
- `serial_torch`: PyTorch single-name tensor implementation
- `parallel_torch`: PyTorch fixed-length batched tensor implementation
- `serial_cpp`: CPU baseline
- `parallel_cpp`: CUDA-target forward implementation

Right now `parallel_cpp` keeps a copied host-side flow similar to `serial_cpp` up to the point where actual forward computation begins. After that boundary, the parallel method switches to CUDA kernels for embeddings, RMSNorm, linear projections, causal self-attention, ReLU, logits, and loss. It is forward-only, assumes fixed sequence lengths within a batch, does not implement padding, and builds only from the `.cu` translation unit with `nvcc` in C++14 mode.
