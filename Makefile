PYTHON ?= python3
CXX ?= c++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pedantic

# Profiling flags
PROFILE_CXXFLAGS := -std=c++17 -O2 -g -fno-omit-frame-pointer -Wall -Wextra -pedantic

# macOS Instruments
#INSTRUMENTS_CXXFLAGS := -std=c++17 -O2 -g -fno-omit-frame-pointer -Wall -Wextra -pedantic


NVCC ?= nvcc
NVCCFLAGS ?= -std=c++14 -O3 -Xcompiler -Wall,-Wextra,-pedantic

BUILD_DIR := build
DATASET := training_data/datasets/names.txt
FIXTURE_DIR := training_data/fixtures/small_case
SAMPLE_NAME := anna

SERIAL_CPP_SRCS := methods/serial_cpp/main.cpp methods/serial_cpp/kernel.cpp methods/serial_cpp/utils.cpp
PARALLEL_CPP_SRCS := methods/parallel_cpp/main.cpp methods/parallel_cpp/kernel.cu methods/parallel_cpp/utils.cpp
BASELINE_CUDA_SRCS := methods/baseline_cuda/main.cpp methods/baseline_cuda/kernel.cu methods/baseline_cuda/utils.cpp
BATCHING_ONLY_SRCS := methods/batching_only/main.cpp methods/batching_only/kernel.cu methods/batching_only/utils.cpp
FLOAT_ONLY_SRCS := methods/float_only/main.cpp methods/float_only/kernel.cu methods/float_only/utils.cpp
TILED_MATMUL_ONLY_SRCS := methods/tiled_matmul_only/main.cpp methods/tiled_matmul_only/kernel.cu methods/tiled_matmul_only/utils.cpp
BATCHING_FLOAT_TILED_SRCS := methods/batching_float_tiled/main.cpp methods/batching_float_tiled/kernel.cu methods/batching_float_tiled/utils.cpp

.PHONY: all fixtures clean profile

all: $(BUILD_DIR)/serial_cpp $(BUILD_DIR)/parallel_cpp

profile: $(BUILD_DIR)/serial_cpp_profile

fixtures:
	$(PYTHON) methods/serial_python/serial.py --mode dump-fixtures --dataset $(DATASET) --sample-name $(SAMPLE_NAME)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Normal optimized build
$(BUILD_DIR)/serial_cpp: $(SERIAL_CPP_SRCS) methods/serial_cpp/kernel.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SERIAL_CPP_SRCS) -o $@

# Profiling build 
$(BUILD_DIR)/serial_cpp_profile: $(SERIAL_CPP_SRCS) methods/serial_cpp/kernel.hpp | $(BUILD_DIR)
	$(CXX) $(PROFILE_CXXFLAGS) $(SERIAL_CPP_SRCS) -o $@

# CUDA build 
$(BUILD_DIR)/parallel_cpp: $(PARALLEL_CPP_SRCS) methods/parallel_cpp/kernel.hpp | $(BUILD_DIR)
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "parallel_cpp requires nvcc on PATH"; exit 1; }
	$(NVCC) $(NVCCFLAGS) $(PARALLEL_CPP_SRCS) -o $@

$(BUILD_DIR)/parallel_cpp_untiled: $(PARALLEL_CPP_SRCS) methods/parallel_cpp/kernel.hpp | $(BUILD_DIR)
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "parallel_cpp_untiled requires nvcc on PATH"; exit 1; }
	$(NVCC) $(NVCCFLAGS) -DUSE_TILED_LINEAR=0 $(PARALLEL_CPP_SRCS) -o $@

$(BUILD_DIR)/baseline_cuda: $(BASELINE_CUDA_SRCS) methods/baseline_cuda/kernel.hpp | $(BUILD_DIR)
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "baseline_cuda requires nvcc on PATH"; exit 1; }
	$(NVCC) $(NVCCFLAGS) $(BASELINE_CUDA_SRCS) -o $@

$(BUILD_DIR)/batching_only: $(BATCHING_ONLY_SRCS) methods/batching_only/kernel.hpp | $(BUILD_DIR)
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "batching_only requires nvcc on PATH"; exit 1; }
	$(NVCC) $(NVCCFLAGS) $(BATCHING_ONLY_SRCS) -o $@

$(BUILD_DIR)/float_only: $(FLOAT_ONLY_SRCS) methods/float_only/kernel.hpp | $(BUILD_DIR)
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "float_only requires nvcc on PATH"; exit 1; }
	$(NVCC) $(NVCCFLAGS) $(FLOAT_ONLY_SRCS) -o $@

$(BUILD_DIR)/tiled_matmul_only: $(TILED_MATMUL_ONLY_SRCS) methods/tiled_matmul_only/kernel.hpp | $(BUILD_DIR)
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "tiled_matmul_only requires nvcc on PATH"; exit 1; }
	$(NVCC) $(NVCCFLAGS) $(TILED_MATMUL_ONLY_SRCS) -o $@

$(BUILD_DIR)/batching_float_tiled: $(BATCHING_FLOAT_TILED_SRCS) methods/batching_float_tiled/kernel.hpp | $(BUILD_DIR)
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "batching_float_tiled requires nvcc on PATH"; exit 1; }
	$(NVCC) $(NVCCFLAGS) $(BATCHING_FLOAT_TILED_SRCS) -o $@

clean:
	rm -rf $(BUILD_DIR)
