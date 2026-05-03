#!/usr/bin/env bash

set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
DATASET="$ROOT_DIR/training_data/datasets/names.txt"
FIXTURE_DIR="$ROOT_DIR/training_data/fixtures/small_case"
PYTHON_BIN="${PYTHON:-python3}"
DEFAULT_BATCH_SIZE="${BATCH_SIZE:-32}"
BENCHMARK_MATRIX="$ROOT_DIR/scripts/benchmark_matrix.py"
baseline_dir="$(mktemp -d)"
trap 'rm -rf "$baseline_dir"' EXIT

serial_python_valid=0
serial_torch_valid=0
parallel_torch_valid=0
serial_valid=0
parallel_valid=0
set_valid() {
    case "$1" in
        serial_python) serial_python_valid="$2" ;;
        serial_torch) serial_torch_valid="$2" ;;
        parallel_torch) parallel_torch_valid="$2" ;;
        serial_cpp) serial_valid="$2" ;;
        parallel_cpp) parallel_valid="$2" ;;
    esac
}

is_valid() {
    case "$1" in
        serial_python) [ "$serial_python_valid" -eq 1 ] ;;
        serial_torch) [ "$serial_torch_valid" -eq 1 ] ;;
        parallel_torch) [ "$parallel_torch_valid" -eq 1 ] ;;
        serial_cpp) [ "$serial_valid" -eq 1 ] ;;
        parallel_cpp) [ "$parallel_valid" -eq 1 ] ;;
        *) return 1 ;;
    esac
}

set_baseline() {
    printf "%s" "$2" > "$baseline_dir/$1"
}

get_baseline() {
    if [ -f "$baseline_dir/$1" ]; then
        cat "$baseline_dir/$1"
    fi
}

run_validate_method() {
    local method="$1"
    case "$method" in
        serial_python)
            "$PYTHON_BIN" "$ROOT_DIR/methods/serial_python/serial.py" --mode validate --fixture-dir "$FIXTURE_DIR"
            ;;
        serial_torch)
            "$PYTHON_BIN" "$ROOT_DIR/methods/serial_torch/serial.py" --mode validate --fixture-dir "$FIXTURE_DIR"
            ;;
        parallel_torch)
            "$PYTHON_BIN" "$ROOT_DIR/methods/parallel_torch/parallel.py" --mode validate --fixture-dir "$FIXTURE_DIR" --batch-size "$DEFAULT_BATCH_SIZE"
            ;;
        serial_cpp)
            "$BUILD_DIR/$method" --mode validate --fixture-dir "$FIXTURE_DIR"
            ;;
        parallel_cpp)
            "$BUILD_DIR/$method" --mode validate --fixture-dir "$FIXTURE_DIR" --batch-size "$DEFAULT_BATCH_SIZE"
            ;;
        *)
            return 1
            ;;
    esac
}

benchmark_once() {
    local method="$1"
    local label="$2"
    local n_layer="$3"
    local n_embd="$4"
    local block_size="$5"
    local n_head="$6"
    local steps="$7"
    local time_file
    time_file="$(mktemp)"
    case "$method" in
        serial_python)
            if ! /usr/bin/time -p "$PYTHON_BIN" "$ROOT_DIR/methods/serial_python/serial.py" \
                --mode benchmark \
                --dataset "$DATASET" \
                --label "$label" \
                --num-steps "$steps" \
                --n-layer "$n_layer" \
                --n-embd "$n_embd" \
                --block-size "$block_size" \
                --n-head "$n_head" \
                > /dev/null 2> "$time_file"; then
                rm -f "$time_file"
                return 1
            fi
            ;;
        serial_torch)
            if ! /usr/bin/time -p "$PYTHON_BIN" "$ROOT_DIR/methods/serial_torch/serial.py" \
                --mode benchmark \
                --dataset "$DATASET" \
                --label "$label" \
                --num-steps "$steps" \
                --n-layer "$n_layer" \
                --n-embd "$n_embd" \
                --block-size "$block_size" \
                --n-head "$n_head" \
                > /dev/null 2> "$time_file"; then
                rm -f "$time_file"
                return 1
            fi
            ;;
        parallel_torch)
            if ! /usr/bin/time -p "$PYTHON_BIN" "$ROOT_DIR/methods/parallel_torch/parallel.py" \
                --mode benchmark \
                --dataset "$DATASET" \
                --label "$label" \
                --num-steps "$steps" \
                --n-layer "$n_layer" \
                --n-embd "$n_embd" \
                --block-size "$block_size" \
                --n-head "$n_head" \
                --batch-size "$DEFAULT_BATCH_SIZE" \
                > /dev/null 2> "$time_file"; then
                rm -f "$time_file"
                return 1
            fi
            ;;
        serial_cpp)
            if ! /usr/bin/time -p "$BUILD_DIR/$method" \
                --mode benchmark \
                --dataset "$DATASET" \
                --label "$label" \
                --num-steps "$steps" \
                --n-layer "$n_layer" \
                --n-embd "$n_embd" \
                --block-size "$block_size" \
                --n-head "$n_head" \
                > /dev/null 2> "$time_file"; then
                rm -f "$time_file"
                return 1
            fi
            ;;
        parallel_cpp)
            if ! /usr/bin/time -p "$BUILD_DIR/$method" \
                --mode benchmark \
                --dataset "$DATASET" \
                --label "$label" \
                --num-steps "$steps" \
                --n-layer "$n_layer" \
                --n-embd "$n_embd" \
                --block-size "$block_size" \
                --n-head "$n_head" \
                --batch-size "$DEFAULT_BATCH_SIZE" \
                > /dev/null 2> "$time_file"; then
                rm -f "$time_file"
                return 1
            fi
            ;;
        *)
            rm -f "$time_file"
            return 1
            ;;
    esac
    awk '/^real / { print $2 }' "$time_file"
    rm -f "$time_file"
    return 0
}

printf "Building binaries and regenerating fixtures...\n"
if ! command -v nvcc >/dev/null 2>&1; then
    printf "parallel_cpp now requires nvcc on PATH. Install the CUDA toolkit before running the full benchmark sweep.\n" >&2
    exit 1
fi
if ! make -C "$ROOT_DIR" fixtures all; then
    printf "Build failed.\n" >&2
    exit 1
fi

for method in serial_python serial_torch parallel_torch serial_cpp parallel_cpp; do
    printf "Validating %s...\n" "$method"
    if run_validate_method "$method"; then
        set_valid "$method" 1
    else
        printf "%s FAILED validation and will be skipped.\n" "$method"
        set_valid "$method" 0
    fi
done

while IFS=$'\t' read -r label n_layer n_embd block_size n_head steps; do
    printf "\nPreset: %s\n" "$label"
    for method in serial_cpp serial_python serial_torch parallel_torch parallel_cpp; do
        if [ "$method" = "serial_python" ] && [ "$label" != "small" ] && [ "$label" != "medium" ]; then
            printf "%s: skipped for preset=%s\n" "$method" "$label"
            continue
        fi
        if ! is_valid "$method"; then
            printf "%s: INVALID\n" "$method"
            continue
        fi
        raw_time="$(benchmark_once "$method" "$label" "$n_layer" "$n_embd" "$block_size" "$n_head" "$steps")" || {
            printf "%s: benchmark failed\n" "$method"
            continue
        }
        printf "%s raw_real=%s\n" "$method" "$raw_time"
        if [ "$method" = "serial_cpp" ]; then
            set_baseline "$label" "$raw_time"
            printf "%s speedup=1.000000\n" "$method"
        else
            baseline_value="$(get_baseline "$label")"
            if [ -n "$baseline_value" ]; then
                speedup="$(awk -v base="$baseline_value" -v current="$raw_time" 'BEGIN { printf "%.6f", base / current }')"
                printf "%s speedup=%s\n" "$method" "$speedup"
            else
                printf "%s speedup=N/A\n" "$method"
            fi
        fi
    done
done < <("$PYTHON_BIN" "$BENCHMARK_MATRIX" --format tsv)
