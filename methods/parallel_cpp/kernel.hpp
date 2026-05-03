#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

struct ModelConfig {
  int n_layer = 1;
  int n_embd = 16;
  int block_size = 16;
  int n_head = 4;
  int vocab_size = 0;

  int head_dim() const { return n_embd / n_head; }
  int mlp_dim() const { return 4 * n_embd; }
};

struct LayerWeights {
  std::vector<float> attn_wq;
  std::vector<float> attn_wk;
  std::vector<float> attn_wv;
  std::vector<float> attn_wo;
  std::vector<float> mlp_fc1;
  std::vector<float> mlp_fc2;
};

struct Model {
  ModelConfig config;
  std::vector<float> wte;
  std::vector<float> wpe;
  std::vector<float> lm_head;
  std::vector<LayerWeights> layers;
};

struct KernelResult {
  int seq_len = 0;
  std::vector<float> logits;
  float loss = 0.0f;
};

struct BatchTokens {
  std::vector<int> tokens;
  // Flattened token IDs for the entire batch.
  // Layout: [batch_size, batch_seq_length] stored row-major.
  // Access: tokens[b * batch_seq_length + t]
  // Each value is a token index into the vocabulary.

  int batch_size = 0;
  // Number of sequences in the batch (B).

  int batch_seq_length = 0;
  // Shared sequence length across the entire batch (T).
};

template <typename T> struct DeviceBuffer {
  T *ptr = nullptr;
  std::size_t count = 0;
};

struct DeviceModel {
  ModelConfig config;
  DeviceBuffer<float> wte;
  DeviceBuffer<float> wpe;
  DeviceBuffer<float> lm_head;
  std::vector<DeviceBuffer<float>> attn_wq;
  std::vector<DeviceBuffer<float>> attn_wk;
  std::vector<DeviceBuffer<float>> attn_wv;
  std::vector<DeviceBuffer<float>> attn_wo;
  std::vector<DeviceBuffer<float>> mlp_fc1;
  std::vector<DeviceBuffer<float>> mlp_fc2;
};

struct DeviceWorkspace {
  DeviceBuffer<int> tokens;
  DeviceBuffer<float> embeddings;
  DeviceBuffer<float> hidden;
  // Forward-only transformer workspace and caches for future CUDA kernels.
  DeviceBuffer<float> x;
  DeviceBuffer<float> x_tmp;
  DeviceBuffer<float> x_mid;
  DeviceBuffer<float> x_norm2;
  DeviceBuffer<float> norm;
  DeviceBuffer<float> q;
  DeviceBuffer<float> k_cache;
  DeviceBuffer<float> v_cache; 
  DeviceBuffer<float> attn_out;
  DeviceBuffer<float> mlp_hidden;
  DeviceBuffer<float> logits;
  DeviceBuffer<float> loss;
  DeviceBuffer<float> relu; 
  DeviceBuffer<float> fc2; 
};

struct KernelLaunch {
  int threads = 256;
  int blocks = 1;
};

inline int compute_usable_seq_len(const ModelConfig &config,
                                  const BatchTokens &batch) {
  return std::min(config.block_size, batch.batch_seq_length - 1);
}

inline KernelLaunch make_1d_launch(std::size_t work_items, int threads = 256) {
  KernelLaunch shape;
  shape.threads = threads;
  shape.blocks =
      static_cast<int>((work_items + static_cast<std::size_t>(threads) - 1) /
                       static_cast<std::size_t>(threads));
  if (shape.blocks < 1) {
    shape.blocks = 1;
  }
  return shape;
}

Model make_empty_model(const ModelConfig &config);
Model initialize_model(const ModelConfig &config, std::uint32_t seed);
void load_model_from_f32(Model &host_model, const std::vector<float> &values);
std::vector<float> flatten_model_values(const Model &host_model);
DeviceModel upload_model_to_device(const Model &host_model);
void free_device_model(DeviceModel *device_model);
KernelResult run_forward_batched(const DeviceModel &device_model,
                                 const BatchTokens &batch);
