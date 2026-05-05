#include "kernel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct CliOptions {
    std::string mode;
    std::string dataset;
    std::string label = "custom";
    int num_steps = -1;
    int n_layer = -1;
    int n_embd = -1;
    int block_size = -1;
    int n_head = -1;
    int batch_size = 1;
    std::uint32_t seed = 42;
};

const std::string kDefaultFixtureDir = "training_data/fixtures/small_case";

struct BatchBucket {
    int seq_length = 0;
    int sequence_count = 0;
    std::vector<int> tokens;
};

BatchTokens make_repeated_batch(const std::vector<int>& tokens, int batch_size) {
    BatchTokens batch;
    batch.tokens.reserve(static_cast<std::size_t>(batch_size) * tokens.size());
    for (int idx = 0; idx < batch_size; ++idx) {
        batch.tokens.insert(batch.tokens.end(), tokens.begin(), tokens.end());
    }
    batch.batch_size = batch_size;
    batch.batch_seq_length = static_cast<int>(tokens.size());
    return batch;
}

BatchTokens make_batch_from_bucket(BatchBucket* bucket) {
    BatchTokens batch;
    batch.tokens.swap(bucket->tokens);
    batch.batch_size = bucket->sequence_count;
    batch.batch_seq_length = bucket->seq_length;
    bucket->sequence_count = 0;
    return batch;
}

std::string join_path(const std::string& dir, const std::string& filename) {
    if (dir.empty()) {
        return filename;
    }
    const char last = dir[dir.size() - 1];
    if (last == '/' || last == '\\') {
        return dir + filename;
    }
    return dir + "/" + filename;
}

std::string require_value(int argc, char** argv, int* index) {
    if (*index + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for option ") + argv[*index]);
    }
    ++(*index);
    return argv[*index];
}

CliOptions parse_cli(int argc, char** argv) {
    CliOptions options;
    for (int idx = 1; idx < argc; ++idx) {
        const std::string arg = argv[idx];
        if (arg == "--mode") {
            options.mode = require_value(argc, argv, &idx);
        } else if (arg == "--dataset") {
            options.dataset = require_value(argc, argv, &idx);
        } else if (arg == "--label") {
            options.label = require_value(argc, argv, &idx);
        } else if (arg == "--num-steps") {
            options.num_steps = std::stoi(require_value(argc, argv, &idx));
        } else if (arg == "--n-layer") {
            options.n_layer = std::stoi(require_value(argc, argv, &idx));
        } else if (arg == "--n-embd") {
            options.n_embd = std::stoi(require_value(argc, argv, &idx));
        } else if (arg == "--block-size") {
            options.block_size = std::stoi(require_value(argc, argv, &idx));
        } else if (arg == "--n-head") {
            options.n_head = std::stoi(require_value(argc, argv, &idx));
        } else if (arg == "--batch-size") {
            options.batch_size = std::stoi(require_value(argc, argv, &idx));
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(std::stoul(require_value(argc, argv, &idx)));
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (options.mode.empty()) {
        throw std::runtime_error("--mode is required");
    }
    if (options.batch_size < 1) {
        throw std::runtime_error("--batch-size must be at least 1");
    }
    return options;
}

std::vector<std::string> load_docs(const std::string& dataset_path) {
    std::ifstream input(dataset_path.c_str());
    if (!input) {
        throw std::runtime_error("failed to open dataset: " + dataset_path);
    }
    std::vector<std::string> docs;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            docs.push_back(line);
        }
    }
    return docs;
}

std::pair<std::string, std::unordered_map<char, int>> build_vocab(const std::vector<std::string>& docs) {
    std::string chars;
    for (const std::string& doc : docs) {
        chars += doc;
    }
    std::sort(chars.begin(), chars.end());
    chars.erase(std::unique(chars.begin(), chars.end()), chars.end());
    std::unordered_map<char, int> vocab;
    for (int idx = 0; idx < static_cast<int>(chars.size()); ++idx) {
        vocab[chars[idx]] = idx;
    }
    return {chars, vocab};
}

ModelConfig benchmark_config_from_options(const CliOptions& options, int vocab_size) {
    if (options.n_layer < 1) {
        throw std::runtime_error("--n-layer must be at least 1");
    }
    if (options.n_embd < 1) {
        throw std::runtime_error("--n-embd must be at least 1");
    }
    if (options.block_size < 1) {
        throw std::runtime_error("--block-size must be at least 1");
    }
    if (options.n_head < 1) {
        throw std::runtime_error("--n-head must be at least 1");
    }
    if (options.n_embd % options.n_head != 0) {
        throw std::runtime_error("--n-embd must be divisible by --n-head");
    }
    return ModelConfig{options.n_layer, options.n_embd, options.block_size, options.n_head, vocab_size};
}

std::vector<int> encode_doc(const std::string& doc, const std::unordered_map<char, int>& vocab, int bos_token_id) {
    std::vector<int> tokens;
    tokens.reserve(doc.size() + 2);
    tokens.push_back(bos_token_id);
    for (char ch : doc) {
        const auto it = vocab.find(ch);
        if (it == vocab.end()) {
            throw std::runtime_error(std::string("sample name contains character not in dataset: ") + ch);
        }
        tokens.push_back(it->second);
    }
    tokens.push_back(bos_token_id);
    return tokens;
}

std::vector<BatchTokens> build_length_bucketed_batches(
    const std::vector<std::string>& docs,
    int doc_count,
    const std::unordered_map<char, int>& vocab,
    int bos_token_id,
    int max_batch_size
) {
    std::vector<BatchTokens> batches;
    std::unordered_map<int, BatchBucket> buckets;
    std::vector<int> length_order;

    for (int doc_idx = 0; doc_idx < doc_count; ++doc_idx) {
        const std::vector<int> tokens = encode_doc(docs[doc_idx], vocab, bos_token_id);
        const int seq_length = static_cast<int>(tokens.size());
        BatchBucket& bucket = buckets[seq_length];
        if (bucket.seq_length == 0) {
            bucket.seq_length = seq_length;
            bucket.tokens.reserve(static_cast<std::size_t>(max_batch_size) * static_cast<std::size_t>(seq_length));
            length_order.push_back(seq_length);
        }

        bucket.tokens.insert(bucket.tokens.end(), tokens.begin(), tokens.end());
        ++bucket.sequence_count;
        if (bucket.sequence_count == max_batch_size) {
            batches.push_back(make_batch_from_bucket(&bucket));
            bucket.tokens.reserve(static_cast<std::size_t>(max_batch_size) * static_cast<std::size_t>(seq_length));
        }
    }

    for (int seq_length : length_order) {
        BatchBucket& bucket = buckets[seq_length];
        if (bucket.sequence_count > 0) {
            batches.push_back(make_batch_from_bucket(&bucket));
        }
    }

    return batches;
}

std::unordered_map<std::string, std::string> parse_manifest(const std::string& manifest_path) {
    std::ifstream input(manifest_path.c_str());
    if (!input) {
        throw std::runtime_error("failed to open manifest: " + manifest_path);
    }
    std::unordered_map<std::string, std::string> manifest;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const std::size_t pos = line.find('=');
        if (pos == std::string::npos) {
            throw std::runtime_error("invalid manifest line: " + line);
        }
        manifest[line.substr(0, pos)] = line.substr(pos + 1);
    }
    return manifest;
}

std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) {
            values.push_back(std::stoi(token));
        }
    }
    return values;
}

std::vector<float> read_f32_file(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open binary file: " + path);
    }
    input.seekg(0, std::ios::end);
    const std::streamsize file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (file_size % static_cast<std::streamsize>(sizeof(float)) != 0) {
        throw std::runtime_error("binary file is not float32-aligned: " + path);
    }
    std::vector<float> values(static_cast<std::size_t>(file_size / static_cast<std::streamsize>(sizeof(float))));
    input.read(reinterpret_cast<char*>(values.data()), file_size);
    return values;
}

std::vector<float> repeat_values(const std::vector<float>& values, int repeat_count) {
    std::vector<float> repeated;
    repeated.reserve(values.size() * static_cast<std::size_t>(repeat_count));
    for (int idx = 0; idx < repeat_count; ++idx) {
        repeated.insert(repeated.end(), values.begin(), values.end());
    }
    return repeated;
}

double compare_arrays(const std::string& label, const std::vector<double>& actual, const std::vector<float>& expected, double epsilon) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(label + " size mismatch");
    }
    double max_abs_error = 0.0;
    std::size_t max_idx = 0;
    for (std::size_t idx = 0; idx < actual.size(); ++idx) {
        const double abs_error = std::abs(static_cast<double>(actual[idx]) - static_cast<double>(expected[idx]));
        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
            max_idx = idx;
        }
    }
    std::cout << label << " max_abs_error=" << std::setprecision(8) << max_abs_error;
    if (!actual.empty()) {
        std::cout << " at_index=" << max_idx;
    }
    std::cout << '\n';
    if (max_abs_error > epsilon) {
        throw std::runtime_error(label + " exceeded validation epsilon");
    }
    return max_abs_error;
}

int run_validate(const CliOptions&) {
    const std::string manifest_path = join_path(kDefaultFixtureDir, "manifest.txt");
    const auto manifest = parse_manifest(manifest_path);
    ModelConfig config;
    config.n_layer = std::stoi(manifest.at("n_layer"));
    config.n_embd = std::stoi(manifest.at("n_embd"));
    config.block_size = std::stoi(manifest.at("block_size"));
    config.n_head = std::stoi(manifest.at("n_head"));
    config.vocab_size = std::stoi(manifest.at("vocab_size"));

    const std::vector<int> tokens = parse_int_list(manifest.at("token_ids"));
    const double epsilon = std::stod(manifest.at("validation_epsilon"));

    Model host_model = make_empty_model(config);
    load_model_from_f32(host_model, read_f32_file(join_path(kDefaultFixtureDir, manifest.at("weights_init_file"))));
    DeviceModel device_model = upload_model_to_device(host_model);
    std::vector<BatchTokens> batches;
    batches.push_back(make_repeated_batch(tokens, 1));
    try {
        KernelResult result;
        for (std::size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx) {
            result = run_forward_batched(device_model, batches[batch_idx]);
        }

        compare_arrays("logits", result.logits, repeat_values(read_f32_file(join_path(kDefaultFixtureDir, manifest.at("expected_logits_file"))), 1), epsilon);
        compare_arrays("loss", std::vector<double>{result.loss}, read_f32_file(join_path(kDefaultFixtureDir, manifest.at("expected_loss_file"))), epsilon);
        std::cout << "validation=pass\n";
    } catch (...) {
        free_device_model(&device_model);
        throw;
    }
    free_device_model(&device_model);
    return 0;
}

int run_benchmark(const CliOptions& options) {
    const auto benchmark_start = std::chrono::steady_clock::now();
    if (options.dataset.empty()) {
        throw std::runtime_error("--dataset is required for benchmark mode");
    }
    if (options.num_steps < 1) {
        throw std::runtime_error("--num-steps must be at least 1");
    }

    const std::vector<std::string> docs = load_docs(options.dataset);
    if (docs.empty()) {
        throw std::runtime_error("dataset is empty");
    }
    const std::pair<std::string, std::unordered_map<char, int>> vocab_result = build_vocab(docs);
    const std::string& uchars = vocab_result.first;
    const std::unordered_map<char, int>& vocab = vocab_result.second;
    const ModelConfig config = benchmark_config_from_options(options, static_cast<int>(uchars.size()) + 1);
    const int requested_steps = options.num_steps;
    const int steps = std::min(requested_steps, static_cast<int>(docs.size()));
    const std::vector<BatchTokens> batches = build_length_bucketed_batches(docs, steps, vocab, static_cast<int>(uchars.size()), options.batch_size);
    const Model host_model = initialize_model(config, options.seed);
    DeviceModel device_model = upload_model_to_device(host_model);
    double last_loss = 0.0;
    double mean_loss = 0.0;
    double weighted_loss_sum = 0.0;
    int loss_item_count = 0;
    double forward_pass_seconds_cumulative = 0.0;
    const std::string last_doc = steps > 0 ? docs[steps - 1] : "";
    try {
        for (std::size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx) {
            const BatchTokens& batch = batches[batch_idx];
            const auto forward_start = std::chrono::steady_clock::now();
            last_loss = run_forward_batched(device_model, batch).loss;
            const auto forward_end = std::chrono::steady_clock::now();
            forward_pass_seconds_cumulative += std::chrono::duration<double>(forward_end - forward_start).count();

            const int usable_seq_len = compute_usable_seq_len(config, batch);
            const int batch_loss_items = batch.batch_size * usable_seq_len;
            weighted_loss_sum += last_loss * static_cast<double>(batch_loss_items);
            loss_item_count += batch_loss_items;
        }
        if (loss_item_count > 0) {
            mean_loss = weighted_loss_sum / static_cast<double>(loss_item_count);
        }
    } catch (...) {
        free_device_model(&device_model);
        throw;
    }
    free_device_model(&device_model);
    const double total_program_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - benchmark_start).count();

    std::cout << "mode=benchmark "
              << "preset=" << options.label << ' '
              << "requested_steps=" << requested_steps << ' '
              << "steps=" << steps << ' '
              << "batch_size=" << options.batch_size << ' '
              << "batches=" << batches.size() << ' '
              << "last_doc=" << last_doc << ' '
              << "loss=" << std::setprecision(8) << last_loss << ' '
              << "mean_loss=" << std::setprecision(8) << mean_loss << ' '
              << "forward_pass_seconds_cumulative=" << std::setprecision(8) << forward_pass_seconds_cumulative << ' '
              << "total_program_seconds=" << std::setprecision(8) << total_program_seconds << ' '
              << "benchmark_status=forward_cuda\n";
    return 0;
}

} 

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_cli(argc, argv);
        if (options.mode == "validate") {
            return run_validate(options);
        }
        if (options.mode == "benchmark") {
            return run_benchmark(options);
        }
        throw std::runtime_error("unsupported mode: " + options.mode);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
