# Parallel GPT

This repository contains the code, report, slides, and benchmark infrastructure for our Colorado School of Mines `CSCI-563` project, completed in Spring 2026 by Siggy Sigler, Sam Abderholden, and Dawson Matthews.

## Project Summary

This project studies one narrow question: how much speedup can we get by parallelizing a GPT-style forward pass on GPU while keeping the workload fixed across implementations. The shared workload is next-token-loss evaluation over tokenized name data. The repository contains a reference Python implementation, PyTorch baselines, a serial C++ implementation, and a CUDA C++ implementation, along with deterministic validation fixtures used to verify that the implementations agree on the same forward pass.

The repository includes the full project stack around that comparison: benchmark presets, benchmark runners, plotting scripts, the final report, and presentation material. The main result in the final report is that the custom CUDA implementation outperformed the CPU and PyTorch baselines across the project benchmark presets. This README is meant to orient a reader to the repository and point to the main entrypoints, not to act as a full reproduction guide.

## Important Project Files

- [Final report (PDF)](/Users/jacksigler/Library/CloudStorage/OneDrive-ColoradoSchoolofMines/Mines/Spring_2026/CSCI-563/parrallel-gpt/project_report/report.pdf)
- [Final report source](/Users/jacksigler/Library/CloudStorage/OneDrive-ColoradoSchoolofMines/Mines/Spring_2026/CSCI-563/parrallel-gpt/project_report/report.tex)
- [Project presentation (PDF)](/Users/jacksigler/Library/CloudStorage/OneDrive-ColoradoSchoolofMines/Mines/Spring_2026/CSCI-563/parrallel-gpt/slides/project_presentation.pdf)
- [Presentation source](/Users/jacksigler/Library/CloudStorage/OneDrive-ColoradoSchoolofMines/Mines/Spring_2026/CSCI-563/parrallel-gpt/slides/project_presentation.tex)
- [Benchmark driver](/Users/jacksigler/Library/CloudStorage/OneDrive-ColoradoSchoolofMines/Mines/Spring_2026/CSCI-563/parrallel-gpt/scripts/run_benchmarks.py)
- [Benchmark preset matrix](/Users/jacksigler/Library/CloudStorage/OneDrive-ColoradoSchoolofMines/Mines/Spring_2026/CSCI-563/parrallel-gpt/scripts/benchmark_matrix.py)
- [Validation fixtures](/Users/jacksigler/Library/CloudStorage/OneDrive-ColoradoSchoolofMines/Mines/Spring_2026/CSCI-563/parrallel-gpt/training_data/fixtures/small_case/manifest.txt)

## Commands


Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Build binaries and fixtures:

```bash
make all
make fixtures
```

Validate implementations:

```bash
python3 methods/serial_python/serial.py --mode validate
python3 methods/serial_torch/serial.py --mode validate
python3 methods/parallel_torch/parallel.py --mode validate
build/serial_cpp --mode validate
build/parallel_cpp --mode validate
```

Run the benchmark sweep and inspect the preset matrix:

```bash
python3 scripts/run_benchmarks.py data/benchmark_results.json
python3 scripts/benchmark_matrix.py
```

