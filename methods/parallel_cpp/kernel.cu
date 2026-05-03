#include "kernel.hpp"

#include <cmath>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA failure while ") + action + ": " + cudaGetErrorString(status));
    }
}

void free_model(DeviceModel* device_model);
void free_workspace(DeviceWorkspace* workspace);

template <typename T>
void free_buffer(DeviceBuffer<T>* device_buffer);

void validate_batch(const ModelConfig& config, const BatchTokens& batch) {
    if (batch.batch_size < 1) {
        throw std::runtime_error("batch must contain at least one sequence");
    }
    if (batch.batch_seq_length < 2) {
        throw std::runtime_error("token sequence must contain at least one input and one target token");
    }
    if (batch.tokens.size() != static_cast<std::size_t>(batch.batch_size) * static_cast<std::size_t>(batch.batch_seq_length)) {
        throw std::runtime_error("batch token count must equal batch_size * batch_seq_length");
    }
    if (config.n_head < 1 || config.n_embd % config.n_head != 0) {
        throw std::runtime_error("n_embd must be divisible by n_head");
    }
    for (int token : batch.tokens) {
        if (token < 0 || token >= config.vocab_size) {
            throw std::runtime_error("batch contains token id outside vocabulary");
        }
    }
}

template <typename T>
void allocate_buffer(DeviceBuffer<T>* device_buffer, std::size_t count, bool zero_initialize = false) {
    device_buffer->ptr = nullptr;
    device_buffer->count = count;
    if (device_buffer->count == 0) {
        return;
    }
    cuda_check(
        cudaMalloc(reinterpret_cast<void**>(&device_buffer->ptr), device_buffer->count * sizeof(T)),
        "allocating device buffer"
    );
    if (zero_initialize) {
        cuda_check(
            cudaMemset(device_buffer->ptr, 0, device_buffer->count * sizeof(T)),
            "zeroing device buffer"
        );
    }
}

template <typename T>
void upload_buffer(DeviceBuffer<T>* device_buffer, const std::vector<T>& host_values) {
    allocate_buffer(device_buffer, host_values.size());
    if (device_buffer->count == 0) {
        return;
    }
    cuda_check(
        cudaMemcpy(device_buffer->ptr, host_values.data(), device_buffer->count * sizeof(T), cudaMemcpyHostToDevice),
        "copying host buffer to device"
    );
}

template <typename T>
void free_buffer(DeviceBuffer<T>* device_buffer) {
    if (device_buffer->ptr != nullptr) {
        cudaFree(device_buffer->ptr);
    }
    device_buffer->ptr = nullptr;
    device_buffer->count = 0;
}

DeviceModel upload_model(const Model& host_model) {
    DeviceModel device_model;
    device_model.config = host_model.config;
    try {
        upload_buffer(&device_model.wte, host_model.wte);
        upload_buffer(&device_model.wpe, host_model.wpe);
        upload_buffer(&device_model.lm_head, host_model.lm_head);

        device_model.attn_wq.reserve(host_model.layers.size());
        device_model.attn_wk.reserve(host_model.layers.size());
        device_model.attn_wv.reserve(host_model.layers.size());
        device_model.attn_wo.reserve(host_model.layers.size());
        device_model.mlp_fc1.reserve(host_model.layers.size());
        device_model.mlp_fc2.reserve(host_model.layers.size());

        for (const LayerWeights& layer : host_model.layers) {
            device_model.attn_wq.emplace_back();
            upload_buffer(&device_model.attn_wq.back(), layer.attn_wq);
            device_model.attn_wk.emplace_back();
            upload_buffer(&device_model.attn_wk.back(), layer.attn_wk);
            device_model.attn_wv.emplace_back();
            upload_buffer(&device_model.attn_wv.back(), layer.attn_wv);
            device_model.attn_wo.emplace_back();
            upload_buffer(&device_model.attn_wo.back(), layer.attn_wo);
            device_model.mlp_fc1.emplace_back();
            upload_buffer(&device_model.mlp_fc1.back(), layer.mlp_fc1);
            device_model.mlp_fc2.emplace_back();
            upload_buffer(&device_model.mlp_fc2.back(), layer.mlp_fc2);
        }
    } catch (...) {
        free_model(&device_model);
        throw;
    }
    return device_model;
}

void free_model(DeviceModel* device_model) {
    free_buffer(&device_model->wte);
    free_buffer(&device_model->wpe);
    free_buffer(&device_model->lm_head);

    for (DeviceBuffer<double>& buffer : device_model->attn_wq) {
        free_buffer(&buffer);
    }
    for (DeviceBuffer<double>& buffer : device_model->attn_wk) {
        free_buffer(&buffer);
    }
    for (DeviceBuffer<double>& buffer : device_model->attn_wv) {
        free_buffer(&buffer);
    }
    for (DeviceBuffer<double>& buffer : device_model->attn_wo) {
        free_buffer(&buffer);
    }
    for (DeviceBuffer<double>& buffer : device_model->mlp_fc1) {
        free_buffer(&buffer);
    }
    for (DeviceBuffer<double>& buffer : device_model->mlp_fc2) {
        free_buffer(&buffer);
    }

    device_model->attn_wq.clear();
    device_model->attn_wk.clear();
    device_model->attn_wv.clear();
    device_model->attn_wo.clear();
    device_model->mlp_fc1.clear();
    device_model->mlp_fc2.clear();
}

DeviceWorkspace allocate_workspace(const ModelConfig& config, const BatchTokens& batch, int usable_seq_len) {
    DeviceWorkspace workspace;
    const std::size_t sequence_count = static_cast<std::size_t>(batch.batch_size);
    const std::size_t time_steps = static_cast<std::size_t>(usable_seq_len);
    const std::size_t hidden_count = sequence_count * time_steps * static_cast<std::size_t>(config.n_embd);
    const std::size_t kv_cache_count = static_cast<std::size_t>(config.n_layer) * hidden_count;
    const std::size_t mlp_hidden_count = sequence_count * time_steps * static_cast<std::size_t>(config.mlp_dim());
    const std::size_t logits_count = sequence_count * time_steps * static_cast<std::size_t>(config.vocab_size);


    try {
        upload_buffer(&workspace.tokens, batch.tokens);
        allocate_buffer(&workspace.embeddings, hidden_count, true);
        allocate_buffer(&workspace.hidden, hidden_count, true);
        allocate_buffer(&workspace.x, hidden_count, true);
        allocate_buffer(&workspace.x_tmp, hidden_count, true);
        allocate_buffer(&workspace.x_mid, hidden_count, true);
        allocate_buffer(&workspace.x_norm2, hidden_count, true);
        allocate_buffer(&workspace.norm, hidden_count, true);
        allocate_buffer(&workspace.q, hidden_count, true);
        allocate_buffer(&workspace.k_cache, kv_cache_count, true);
        allocate_buffer(&workspace.v_cache, kv_cache_count, true);
        allocate_buffer(&workspace.attn_out, hidden_count, true);
        allocate_buffer(&workspace.mlp_hidden, mlp_hidden_count, true);
        allocate_buffer(&workspace.logits, logits_count, true);
        allocate_buffer(&workspace.loss, 1, true);
        allocate_buffer(&workspace.fc2, hidden_count, true);
    } catch (...) {
        free_workspace(&workspace);
        throw;
    }
    return workspace;
}

void free_workspace(DeviceWorkspace* workspace) {
    free_buffer(&workspace->tokens);
    free_buffer(&workspace->embeddings);
    free_buffer(&workspace->hidden);
    free_buffer(&workspace->x);
    free_buffer(&workspace->x_tmp);
    free_buffer(&workspace->x_mid);
    free_buffer(&workspace->x_norm2);
    free_buffer(&workspace->norm);
    free_buffer(&workspace->q);
    free_buffer(&workspace->k_cache);
    free_buffer(&workspace->v_cache);
    free_buffer(&workspace->attn_out);
    free_buffer(&workspace->mlp_hidden);
    free_buffer(&workspace->logits);
    free_buffer(&workspace->loss);
    free_buffer(&workspace->fc2);

}

__global__ void embedding_lookup_kernel(
    const int* tokens,
    const double* wte,
    const double* wpe,
    double* embeddings,
    int batch_size,
    int batch_seq_length,
    int usable_seq_len,
    int n_embd
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * usable_seq_len * n_embd;
    if (idx >= total) {
        return;
    }

    const int col = idx % n_embd;
    const int token_slot = idx / n_embd;
    const int pos = token_slot % usable_seq_len;
    const int batch_idx = token_slot / usable_seq_len;
    const int token_idx = batch_idx * batch_seq_length + pos;
    const int token_id = tokens[token_idx];

    const double token_val = wte[token_id * n_embd + col];
    const double pos_val = wpe[pos * n_embd + col];
    embeddings[idx] = token_val + pos_val;
}

__global__ void add_vec_kernel(
    const double* left,
    const double* right,
    double* output,
    const int n
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= n) return;

    output[tid] = left[tid] + right[tid];
}

__global__ void rmsnorm_kernel(
    const double* input,
    double* output,
    int n_embd,
    int useable_seq_len, 
    int num_batches
) {
    int total_tokens = num_batches * useable_seq_len;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= total_tokens) return;

    int start = idx * n_embd;

    double mean_square = 0.0;

    for (int i = 0; i < n_embd; i++) {
        double value = input[start + i];
        mean_square += value * value;
    }

    mean_square /= (double)n_embd;

    double scale = 1.0 / sqrt(mean_square + 1e-5);

    for (int i = 0; i < n_embd; i++) {
        output[start + i] = input[start + i] * scale;
    }
}


__global__ void linear_kernel(
    const double* input,
    double* output,
    const double* weights,
    int in_dim,
    int out_dim,
    int num_batches,
    int usable_seq_len
) {
    int total_tokens = num_batches * usable_seq_len;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= total_tokens) return;

    int input_start = idx * in_dim;
    int output_start = idx * out_dim;

    for (int out = 0; out < out_dim; ++out) {
        double sum = 0.0;

        for (int in = 0; in < in_dim; ++in) {
            sum += weights[out * in_dim + in] * input[input_start + in];
        }

        output[output_start + out] = sum;
    }
}



__global__ void relu_kernel(
    double* input,
    int n
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= n) return; 

    if (input[idx] >= 0) return; 

    input[idx] = 0.0; 
}

__global__ void self_attn_kernel(
    const double* q,
    const double* k_layer,
    const double* v_layer,
    double* attn_out,
    int batch_size,
    int usable_seq_len,
    int n_head,
    int head_dim,
    int n_embd
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * usable_seq_len * n_embd;
    if (idx >= total) {
        return;
    }

    const int col = idx % n_embd;
    const int token_slot = idx / n_embd;
    const int token_pos = token_slot % usable_seq_len;
    const int batch_idx = token_slot / usable_seq_len;
    const int sequence_start = batch_idx * usable_seq_len * n_embd;
    const int token_start = sequence_start + token_pos * n_embd;
    const int head = col / head_dim;
    const int head_start = head * head_dim;
    const double scale = 1.0 / sqrt(static_cast<double>(head_dim));

    if (head >= n_head) {
        return;
    }

    double max_logit = -INFINITY;
    for (int t = 0; t <= token_pos; ++t) {
        const int past_start = sequence_start + t * n_embd;
        double dot = 0.0;
        for (int j = 0; j < head_dim; ++j) {
            dot += q[token_start + head_start + j] * k_layer[past_start + head_start + j];
        }
        const double score = dot * scale;
        if (score > max_logit) {
            max_logit = score;
        }
    }

    double exp_sum = 0.0;
    for (int t = 0; t <= token_pos; ++t) {
        const int past_start = sequence_start + t * n_embd;
        double dot = 0.0;
        for (int j = 0; j < head_dim; ++j) {
            dot += q[token_start + head_start + j] * k_layer[past_start + head_start + j];
        }
        exp_sum += exp(dot * scale - max_logit);
    }

    double weighted_value = 0.0;
    for (int t = 0; t <= token_pos; ++t) {
        const int past_start = sequence_start + t * n_embd;
        double dot = 0.0;
        for (int j = 0; j < head_dim; ++j) {
            dot += q[token_start + head_start + j] * k_layer[past_start + head_start + j];
        }
        const double weight = exp(dot * scale - max_logit) / exp_sum;
        weighted_value += weight * v_layer[past_start + col];
    }

    attn_out[idx] = weighted_value;
}


__global__ void cross_entropy_loss_kernel(
    const double* logits,
    double* loss,
    const int* tokens,
    int batch_size,
    int batch_seq_length,
    int usable_seq_len,
    int vocab_size
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    double total_loss = 0.0;
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (int pos = 0; pos < usable_seq_len; ++pos) {
            const int logits_start = (batch_idx * usable_seq_len + pos) * vocab_size;
            const int target_token = tokens[batch_idx * batch_seq_length + pos + 1];
            double max_logit = logits[logits_start];
            for (int vocab_idx = 1; vocab_idx < vocab_size; ++vocab_idx) {
                const double value = logits[logits_start + vocab_idx];
                if (value > max_logit) {
                    max_logit = value;
                }
            }

            double exp_sum = 0.0;
            for (int vocab_idx = 0; vocab_idx < vocab_size; ++vocab_idx) {
                exp_sum += exp(logits[logits_start + vocab_idx] - max_logit);
            }
            total_loss += log(exp_sum) + max_logit - logits[logits_start + target_token];
        }
    }
    *loss = total_loss / static_cast<double>(batch_size * usable_seq_len);
}

void launch_embedding(const DeviceModel& device_model, DeviceWorkspace* workspace, const ModelConfig& config, const BatchTokens& batch) {
    const int usable_seq_len = compute_usable_seq_len(config, batch);
    const auto launch = make_1d_launch(
        static_cast<std::size_t>(batch.batch_size) * static_cast<std::size_t>(usable_seq_len) *
        static_cast<std::size_t>(config.n_embd)
    );
    embedding_lookup_kernel<<<launch.blocks, launch.threads>>>(
        workspace->tokens.ptr,
        device_model.wte.ptr,
        device_model.wpe.ptr,
        workspace->embeddings.ptr,
        batch.batch_size,
        batch.batch_seq_length,
        usable_seq_len,
        config.n_embd
    );
    cuda_check(cudaGetLastError(), "launching embedding_lookup_kernel");
}


void launch_rmsnorm(
    const double* input,
    double* output,
    int n_embd,
    int batch_size,
    int usable_seq_len
) {
    const auto launch = make_1d_launch(
        static_cast<std::size_t>(batch_size) *
        static_cast<std::size_t>(usable_seq_len)
    );

    rmsnorm_kernel<<<launch.blocks, launch.threads>>>(
        input,
        output,
        n_embd,
        usable_seq_len,
        batch_size
    );

    cuda_check(cudaGetLastError(), "launching rmsnorm_kernel");
}


void launch_linear(
    const double* input,
    double* output,
    const double* weights, 
    int in_dim, 
    int out_dim, 
    int batch_size,
    int usable_seq_len
) {
    const auto launch = make_1d_launch(
        static_cast<std::size_t>(batch_size) *
        static_cast<std::size_t>(usable_seq_len)
    );

    linear_kernel<<<launch.blocks, launch.threads>>>(
        input,
        output,
        weights,
        in_dim, 
        out_dim,
        batch_size,
        usable_seq_len
    );

    cuda_check(cudaGetLastError(), "launching linear_kernel");
}


void launch_self_attn(
    const double* q,
    const double* k_layer,
    const double* v_layer,
    double* attn_out,
    int batch_size,
    int usable_seq_len,
    int n_head,
    int head_dim,
    int n_embd
) {
    const auto launch = make_1d_launch(
        static_cast<std::size_t>(batch_size) *
        static_cast<std::size_t>(usable_seq_len) *
        static_cast<std::size_t>(n_embd)
    );

    self_attn_kernel<<<launch.blocks, launch.threads>>>(
        q,
        k_layer,
        v_layer,
        attn_out,
        batch_size,
        usable_seq_len,
        n_head,
        head_dim,
        n_embd
    );

    cuda_check(cudaGetLastError(), "launching self_attn_kernel");
}


void launch_vec_add(
    const double* left,
    const double* right,
    double* output,
    const int n
) {
    const auto launch = make_1d_launch(
        static_cast<std::size_t>(n)
    );

    add_vec_kernel<<<launch.blocks, launch.threads>>>(
        left,
        right, 
        output, 
        n
    );

    cuda_check(cudaGetLastError(), "launching add_vec_kernel");
}


void launch_relu(
    double* input, 
    int size
) {
    const auto launch = make_1d_launch(
        size
    );

    relu_kernel<<<launch.blocks, launch.threads>>>(
        input, 
        size
    );

    cuda_check(cudaGetLastError(), "launching relu_kernel");
}

void launch_transformer(const DeviceModel& device_model, DeviceWorkspace* workspace, const ModelConfig& config, const BatchTokens& batch) {
    /*
    embedding_lookup -> workspace.x

    for each layer:
        rmsnorm(x -> norm)
        linear(norm -> q)
        linear(norm -> k_cache[layer])
        linear(norm -> v_cache[layer])
        attention(q, k_cache[layer], v_cache[layer] -> attn_out)
        linear(attn_out -> x_tmp)
        residual_add(x, x_tmp -> x)

        rmsnorm(x -> norm)
        linear(norm -> mlp_hidden)
        relu(mlp_hidden)
        linear(mlp_hidden -> x_tmp)
        residual_add(x, x_tmp -> x)

    logits_and_loss(x -> logits/loss)
    */

    const int usable_seq_len = compute_usable_seq_len(config, batch);


    launch_embedding(device_model, workspace, config, batch);
    launch_rmsnorm(workspace->embeddings.ptr, workspace->x.ptr, config.n_embd, batch.batch_size, usable_seq_len);

    for (int layer_idx = 0; layer_idx < config.n_layer; ++layer_idx) {
        launch_rmsnorm(workspace->x.ptr, workspace->norm.ptr, config.n_embd, batch.batch_size, usable_seq_len);
        launch_linear(workspace->norm.ptr, workspace->q.ptr, device_model.attn_wq[layer_idx].ptr, config.n_embd, config.n_embd, batch.batch_size, usable_seq_len);
        
        //* Find the layer in the cache 
        double* k_layer = workspace->k_cache.ptr + layer_idx * batch.batch_size * usable_seq_len * config.n_embd;
        double* v_layer = workspace->v_cache.ptr + layer_idx * batch.batch_size * usable_seq_len * config.n_embd;
        
        launch_linear(workspace->norm.ptr, k_layer, device_model.attn_wk[layer_idx].ptr, config.n_embd, config.n_embd, batch.batch_size, usable_seq_len);
        launch_linear(workspace->norm.ptr, v_layer, device_model.attn_wv[layer_idx].ptr, config.n_embd, config.n_embd, batch.batch_size, usable_seq_len);
        launch_self_attn(
            workspace->q.ptr,
            k_layer,
            v_layer,
            workspace->attn_out.ptr,
            batch.batch_size,
            usable_seq_len,
            config.n_head,
            config.head_dim(),
            config.n_embd
        );
        
        launch_linear(workspace->attn_out.ptr, workspace->x_tmp.ptr, device_model.attn_wo[layer_idx].ptr, config.n_embd, config.n_embd, batch.batch_size, usable_seq_len);
        launch_vec_add(workspace->x.ptr, workspace->x_tmp.ptr, workspace->x_mid.ptr, static_cast<int>(workspace->x_mid.count));
        launch_rmsnorm(workspace->x_mid.ptr, workspace->x_norm2.ptr, config.n_embd, batch.batch_size, usable_seq_len);
        
        //* MLP Perceptron 
        launch_linear(workspace->x_norm2.ptr, workspace->mlp_hidden.ptr, device_model.mlp_fc1[layer_idx].ptr, config.n_embd, config.mlp_dim(), batch.batch_size, usable_seq_len);
        launch_relu(workspace->mlp_hidden.ptr, static_cast<int>(workspace->mlp_hidden.count)); 
        launch_linear(workspace->mlp_hidden.ptr, workspace->fc2.ptr, device_model.mlp_fc2[layer_idx].ptr, config.mlp_dim(), config.n_embd, batch.batch_size, usable_seq_len);
        launch_vec_add(workspace->x_mid.ptr, workspace->fc2.ptr, workspace->x.ptr, static_cast<int>(workspace->x.count));
    }
}

void launch_logits_and_loss(const DeviceModel& device_model, DeviceWorkspace* workspace, const ModelConfig& config, const BatchTokens& batch) {
    const int usable_seq_len = compute_usable_seq_len(config, batch);


    launch_linear(workspace->x.ptr, workspace->logits.ptr, device_model.lm_head.ptr, config.n_embd, config.vocab_size, batch.batch_size, usable_seq_len);
    cross_entropy_loss_kernel<<<1, 1>>>(
        workspace->logits.ptr,
        workspace->loss.ptr,
        workspace->tokens.ptr,
        batch.batch_size,
        batch.batch_seq_length,
        usable_seq_len,
        config.vocab_size
    );
    cuda_check(cudaGetLastError(), "launching cross_entropy_loss_kernel");
}

}  

DeviceModel upload_model_to_device(const Model& host_model) {
    return upload_model(host_model);
}

void free_device_model(DeviceModel* device_model) {
    if (device_model == nullptr) {
        return;
    }
    free_model(device_model);
    device_model->config = ModelConfig{};
}

KernelResult run_forward_batched(const DeviceModel& device_model, const BatchTokens& batch) {
    validate_batch(device_model.config, batch);
    const int usable_seq_len = compute_usable_seq_len(device_model.config, batch);

    DeviceWorkspace workspace = allocate_workspace(device_model.config, batch, usable_seq_len);
    KernelResult result;
    result.seq_len = usable_seq_len;

    launch_transformer(device_model, &workspace, device_model.config, batch);
    launch_logits_and_loss(device_model, &workspace, device_model.config, batch);
    cuda_check(cudaDeviceSynchronize(), "synchronizing CUDA kernels");
    result.logits.resize(workspace.logits.count);

    free_workspace(&workspace);
    return result;
}
