#pragma once

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

Model make_empty_model(const ModelConfig &config);
Model initialize_model(const ModelConfig &config, std::uint32_t seed);
void load_model_from_f32(Model &model, const std::vector<float> &values);
std::vector<float> flatten_model_values(const Model &model);
KernelResult run_forward(const Model &model, const std::vector<int> &tokens);
