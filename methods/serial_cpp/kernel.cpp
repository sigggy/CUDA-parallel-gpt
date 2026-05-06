#include "kernel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

struct LayerStepCache {
    std::vector<float> k;
    std::vector<float> v;
};

struct StepForwardCache {
    std::vector<LayerStepCache> layers;
};

constexpr float kRmsNormEps = 1e-5f;

std::vector<float> add_vectors(const std::vector<float>& left, const std::vector<float>& right) {
    // Pure elementwise vector add:
    // output[i] = left[i] + right[i]
    std::vector<float> output(left.size(), 0.0f);
    for (std::size_t idx = 0; idx < left.size(); ++idx) {
        output[idx] = left[idx] + right[idx];
    }
    return output;
}

std::vector<float> linear(const std::vector<float>& input, const std::vector<float>& weights, int out_dim) {
    // Dense matrix-vector multiply.
    // Shapes:
    // - input: [in_dim]
    // - weights: [out_dim, in_dim] stored row-major
    // - output: [out_dim]
    //
    // Math:
    // output[out] = sum_in weights[out, in] * input[in]
    const int in_dim = static_cast<int>(input.size());
    std::vector<float> output(out_dim, 0.0f);
    for (int out = 0; out < out_dim; ++out) {
        float sum = 0.0f;
        for (int in = 0; in < in_dim; ++in) {
            sum += weights[out * in_dim + in] * input[in];
        }
        output[out] = sum;
    }
    return output;
}

std::vector<float> softmax(const std::vector<float>& logits) {
    // Convert arbitrary logits into a probability distribution.
    //
    // Math:
    // probs[i] = exp(logits[i]) / sum_j exp(logits[j])
    //
    // We subtract max_logit first for numerical stability so exp() does not overflow.
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size(), 0.0f);
    float exp_sum = 0.0f;
    for (std::size_t idx = 0; idx < logits.size(); ++idx) {
        probs[idx] = std::exp(logits[idx] - max_logit);
        exp_sum += probs[idx];
    }
    for (float& prob : probs) {
        prob /= exp_sum;
    }
    return probs;
}

std::vector<float> rmsnorm(const std::vector<float>& input) {
    // RMSNorm rescales the vector by its root-mean-square magnitude.
    //
    // Math:
    // mean_square = (1 / N) * sum_i input[i]^2
    // scale = 1 / sqrt(mean_square + eps)
    // output[i] = input[i] * scale
    //
    // This keeps the direction of the vector but normalizes its overall scale.
    float mean_square = 0.0f;
    for (float value : input) {
        mean_square += value * value;
    }
    mean_square /= static_cast<float>(input.size());
    const float scale = 1.0f / std::sqrt(mean_square + kRmsNormEps);

    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t idx = 0; idx < input.size(); ++idx) {
        output[idx] = input[idx] * scale;
    }
    return output;
}

KernelResult forward_pass(const Model& model, const std::vector<int>& tokens) {
    // Input: model weights plus one tokenized sequence such as [BOS, a, n, n, a, BOS].
    // Transformation: run embeddings, transformer blocks, and next-token scoring while caching activations.
    // Output: per-position hidden states/logits and the average next-token loss.
    const ModelConfig& config = model.config;
    KernelResult result;
    result.seq_len = std::min(config.block_size, static_cast<int>(tokens.size()) - 1);
    std::vector<StepForwardCache> cache(result.seq_len);
    result.logits.reserve(static_cast<std::size_t>(result.seq_len) * static_cast<std::size_t>(config.vocab_size));

    //* Per token 
    for (int pos = 0; pos < result.seq_len; ++pos) {
        StepForwardCache& step_cache = cache[pos];
        const int input_token = tokens[pos];
        const int target_token = tokens[pos + 1];
        std::vector<float> embed_pre(config.n_embd, 0.0f);

        // Input: one token ID and one position ID.
        // Transformation: look up the token embedding and position embedding, add them elementwise.
        // Output: the pre-normalized representation for this sequence position.
        //* CUDA speedup opporunity //* Per embedding dim
        for (int col = 0; col < config.n_embd; ++col) {
            float token_val = model.wte[input_token * config.n_embd + col];
            float pos_val   = model.wpe[pos * config.n_embd + col];

            embed_pre[col] = token_val + pos_val;
        }

        std::vector<float> x = rmsnorm(embed_pre); //* CUDA opporunity
        step_cache.layers.resize(config.n_layer);

        for (int layer_idx = 0; layer_idx < config.n_layer; ++layer_idx) {
            const LayerWeights& layer = model.layers[layer_idx];
            LayerStepCache& layer_step = step_cache.layers[layer_idx];
            const std::vector<float> x_residual = x;
            const std::vector<float> x_norm1 = rmsnorm(x);
            const std::vector<float> q = linear(x_norm1, layer.attn_wq, config.n_embd);
            layer_step.k = linear(x_norm1, layer.attn_wk, config.n_embd);
            layer_step.v = linear(x_norm1, layer.attn_wv, config.n_embd);

            std::vector<float> x_attn(config.n_embd, 0.0f);
            for (int head = 0; head < config.n_head; ++head) {
                const int head_start = head * config.head_dim();
                std::vector<float> attn_logits(pos + 1, 0.0f);
                for (int t = 0; t <= pos; ++t) {
                    const std::vector<float>& past_k = cache[t].layers[layer_idx].k;
                    float dot = 0.0f;
                    for (int j = 0; j < config.head_dim(); ++j) {
                        dot += q[head_start + j] * past_k[head_start + j];
                    }
                    attn_logits[t] = dot / std::sqrt(static_cast<float>(config.head_dim()));
                }

                const std::vector<float> attn_weights = softmax(attn_logits);
                for (int t = 0; t <= pos; ++t) {
                    const std::vector<float>& past_v = cache[t].layers[layer_idx].v;
                    const float weight = attn_weights[t];
                    for (int j = 0; j < config.head_dim(); ++j) {
                        x_attn[head_start + j] += weight * past_v[head_start + j];
                    }
                }
            }

            const std::vector<float> attn_proj = linear(x_attn, layer.attn_wo, config.n_embd);
            const std::vector<float> x_mid = add_vectors(x_residual, attn_proj);
            const std::vector<float> x_norm2 = rmsnorm(x_mid);
            std::vector<float> relu = linear(x_norm2, layer.mlp_fc1, config.mlp_dim());
            for (float& value : relu) {
                value = std::max(0.0f, value);
            }

            // Input: the post-attention hidden state.
            // Transformation: expand with FC1, apply ReLU, project back down with FC2, then add the residual.
            // Output: the layer output that becomes the next layer's input.
            const std::vector<float> fc2 = linear(relu, layer.mlp_fc2, config.n_embd);
            x = add_vectors(x_mid, fc2);
        }

        // Input: the final hidden state at this position.
        // Transformation: project into vocabulary space and normalize with softmax for the target token.
        // Output: logits for every possible next token and one scalar loss contribution.
        const std::vector<float> logits = linear(x, model.lm_head, config.vocab_size);
        const std::vector<float> probs = softmax(logits);
        result.logits.insert(result.logits.end(), logits.begin(), logits.end());
        result.loss += -std::log(probs[target_token]);
    }

    result.loss /= static_cast<float>(result.seq_len);
    return result;
}

}  // namespace

KernelResult run_forward(const Model& model, const std::vector<int>& tokens) {
    if (tokens.size() < 2) {
        throw std::runtime_error("token sequence must contain at least one input and one target token");
    }

    return forward_pass(model, tokens);
}
