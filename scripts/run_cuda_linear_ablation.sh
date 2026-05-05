#!/usr/bin/env bash

set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
DATASET="$ROOT_DIR/training_data/datasets/names.txt"
PYTHON_BIN="${PYTHON:-python3}"
DEFAULT_BATCH_SIZE="${BATCH_SIZE:-32}"
BENCHMARK_MATRIX="$ROOT_DIR/scripts/benchmark_matrix.py"

run_validate_method() {
    local method="$1"
    "$BUILD_DIR/$method" --mode validate
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
        parallel_cpp_untiled|parallel_cpp)
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
    printf "CUDA ablation requires nvcc on PATH.\n" >&2
    exit 1
fi
if ! make -C "$ROOT_DIR" fixtures build/serial_cpp build/parallel_cpp_untiled build/parallel_cpp; then
    printf "Build failed.\n" >&2
    exit 1
fi

for method in serial_cpp parallel_cpp_untiled parallel_cpp; do
    printf "Validating %s...\n" "$method"
    if ! run_validate_method "$method"; then
        printf "%s FAILED validation.\n" "$method" >&2
        exit 1
    fi
done

printf "\nUsing batch_size=%s for CUDA methods.\n" "$DEFAULT_BATCH_SIZE"

while IFS=$'\t' read -r label n_layer n_embd block_size n_head steps; do
    printf "\nPreset: %s\n" "$label"

    serial_time="$(benchmark_once serial_cpp "$label" "$n_layer" "$n_embd" "$block_size" "$n_head" "$steps")" || {
        printf "serial_cpp: benchmark failed\n" >&2
        exit 1
    }
    untiled_time="$(benchmark_once parallel_cpp_untiled "$label" "$n_layer" "$n_embd" "$block_size" "$n_head" "$steps")" || {
        printf "parallel_cpp_untiled: benchmark failed\n" >&2
        exit 1
    }
    tiled_time="$(benchmark_once parallel_cpp "$label" "$n_layer" "$n_embd" "$block_size" "$n_head" "$steps")" || {
        printf "parallel_cpp: benchmark failed\n" >&2
        exit 1
    }

    serial_vs_untiled="$(awk -v base="$serial_time" -v current="$untiled_time" 'BEGIN { printf "%.6f", base / current }')"
    serial_vs_tiled="$(awk -v base="$serial_time" -v current="$tiled_time" 'BEGIN { printf "%.6f", base / current }')"
    untiled_vs_tiled="$(awk -v untiled="$untiled_time" -v tiled="$tiled_time" 'BEGIN { printf "%.6f", untiled / tiled }')"

    printf "serial_cpp raw_real=%s speedup=1.000000\n" "$serial_time"
    printf "parallel_cpp_untiled raw_real=%s speedup_vs_serial=%s\n" "$untiled_time" "$serial_vs_untiled"
    printf "parallel_cpp raw_real=%s speedup_vs_serial=%s\n" "$tiled_time" "$serial_vs_tiled"
    printf "tiled_vs_untiled=%s\n" "$untiled_vs_tiled"
done < <("$PYTHON_BIN" "$BENCHMARK_MATRIX" --format tsv)
