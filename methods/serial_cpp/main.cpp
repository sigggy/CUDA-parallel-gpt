#include "kernel.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
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
  std::filesystem::path dataset;
  std::string label = "custom";
  int num_steps = -1;
  int n_layer = -1;
  int n_embd = -1;
  int block_size = -1;
  int n_head = -1;
  std::uint32_t seed = 42;
};

const std::filesystem::path kDefaultFixtureDir =
    std::filesystem::path("training_data") / "fixtures" / "small_case";

std::string require_value(int argc, char **argv, int *index) {
  if (*index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for option ") +
                             argv[*index]);
  }
  ++(*index);
  return argv[*index];
}

CliOptions parse_cli(int argc, char **argv) {
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
    } else if (arg == "--seed") {
      options.seed = static_cast<std::uint32_t>(
          std::stoul(require_value(argc, argv, &idx)));
    } else {
      throw std::runtime_error("unknown option: " + arg);
    }
  }
  if (options.mode.empty()) {
    throw std::runtime_error("--mode is required");
  }
  return options;
}

std::vector<std::string> load_docs(const std::filesystem::path &dataset_path) {
  std::ifstream input(dataset_path);
  if (!input) {
    throw std::runtime_error("failed to open dataset: " +
                             dataset_path.string());
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

std::pair<std::string, std::unordered_map<char, int>>
build_vocab(const std::vector<std::string> &docs) {
  // Input: raw dataset strings.
  // Transformation: collect every character, sort, deduplicate, then assign
  // each char an integer ID. Output: the ordered character list plus a char ->
  // token_id lookup table.
  std::string chars;
  for (const std::string &doc : docs) {
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

ModelConfig benchmark_config_from_options(const CliOptions &options,
                                          int vocab_size) {
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
  return ModelConfig{options.n_layer, options.n_embd, options.block_size,
                     options.n_head, vocab_size};
}

std::vector<int> encode_doc(const std::string &doc,
                            const std::unordered_map<char, int> &vocab,
                            int bos_token_id) {
  // Input: one document like "anna" and the shared vocabulary mapping.
  // Transformation: add BOS at both ends and replace each character with its
  // token ID. Output: a token sequence suitable for next-token prediction.
  std::vector<int> tokens;
  tokens.reserve(doc.size() + 2);
  tokens.push_back(bos_token_id);
  for (char ch : doc) {
    const auto it = vocab.find(ch);
    if (it == vocab.end()) {
      throw std::runtime_error(
          std::string("sample name contains character not in dataset: ") + ch);
    }
    tokens.push_back(it->second);
  }
  tokens.push_back(bos_token_id);
  return tokens;
}

std::unordered_map<std::string, std::string>
parse_manifest(const std::filesystem::path &manifest_path) {
  std::ifstream input(manifest_path);
  if (!input) {
    throw std::runtime_error("failed to open manifest: " +
                             manifest_path.string());
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

std::vector<int> parse_int_list(const std::string &text) {
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

std::vector<float> read_f32_file(const std::filesystem::path &path) {
  // Input: a binary fixture file written as packed float32 values.
  // Transformation: read raw bytes, verify the size matches float32 alignment,
  // reinterpret as floats. Output: the numeric fixture values used for
  // validation.
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open binary file: " + path.string());
  }
  input.seekg(0, std::ios::end);
  const std::streamsize file_size = input.tellg();
  input.seekg(0, std::ios::beg);
  if (file_size % static_cast<std::streamsize>(sizeof(float)) != 0) {
    throw std::runtime_error("binary file is not float32-aligned: " +
                             path.string());
  }
  std::vector<float> values(static_cast<std::size_t>(
      file_size / static_cast<std::streamsize>(sizeof(float))));
  input.read(reinterpret_cast<char *>(values.data()), file_size);
  return values;
}

double compare_arrays(const std::string &label,
                      const std::vector<float> &actual,
                      const std::vector<float> &expected, double epsilon) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(label + " size mismatch");
  }
  double max_abs_error = 0.0;
  std::size_t max_idx = 0;
  for (std::size_t idx = 0; idx < actual.size(); ++idx) {
    const double abs_error =
        std::abs(static_cast<double>(actual[idx]) - static_cast<double>(expected[idx]));
    if (abs_error > max_abs_error) {
      max_abs_error = abs_error;
      max_idx = idx;
    }
  }
  std::cout << label << " max_abs_error=" << std::setprecision(8)
            << max_abs_error;
  if (!actual.empty()) {
    std::cout << " at_index=" << max_idx;
  }
  std::cout << '\n';
  if (max_abs_error > epsilon) {
    throw std::runtime_error(label + " exceeded validation epsilon");
  }
  return max_abs_error;
}

int run_validate(const CliOptions &) {
  // Input: manifest metadata plus the fixture files produced by the Python
  // reference. Transformation: rebuild the same model state and token sequence,
  // then run the C++ kernel. Output: max-error checks for logits and loss
  // against the reference outputs.
  const std::filesystem::path manifest_path = kDefaultFixtureDir / "manifest.txt";
  const auto manifest = parse_manifest(manifest_path);
  ModelConfig config;
  config.n_layer = std::stoi(manifest.at("n_layer"));
  config.n_embd = std::stoi(manifest.at("n_embd"));
  config.block_size = std::stoi(manifest.at("block_size"));
  config.n_head = std::stoi(manifest.at("n_head"));
  config.vocab_size = std::stoi(manifest.at("vocab_size"));

  const std::vector<int> tokens = parse_int_list(manifest.at("token_ids"));
  const double epsilon = std::stod(manifest.at("validation_epsilon"));

  Model model = make_empty_model(config);
  load_model_from_f32(model, read_f32_file(kDefaultFixtureDir /
                                           manifest.at("weights_init_file")));
  const KernelResult result = run_forward(model, tokens);

  compare_arrays(
      "logits", result.logits,
      read_f32_file(kDefaultFixtureDir / manifest.at("expected_logits_file")),
      epsilon);
  compare_arrays(
      "loss", {result.loss},
      read_f32_file(kDefaultFixtureDir / manifest.at("expected_loss_file")),
      epsilon);
  std::cout << "validation=pass\n";
  return 0;
}

int run_benchmark(const CliOptions &options) {
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
  const auto [uchars, vocab] = build_vocab(docs);
  const ModelConfig config =
      benchmark_config_from_options(options, static_cast<int>(uchars.size()) + 1);
  const int requested_steps = options.num_steps;
  const int steps = std::min(requested_steps, static_cast<int>(docs.size()));
  const Model model = initialize_model(config, options.seed);

  // Input: the dataset names in file order.
  // Transformation: tokenize each name and run one forward pass per example.
  // Output: the loss from the last processed document, which is printed for
  // benchmarking.
  float last_loss = 0.0f;
  double forward_pass_seconds_cumulative = 0.0;
  std::string last_doc;
  for (int step = 0; step < steps; ++step) {
    last_doc = docs[step];
    const std::vector<int> tokens =
        encode_doc(last_doc, vocab, static_cast<int>(uchars.size()));
    const auto forward_start = std::chrono::steady_clock::now();
    last_loss = run_forward(model, tokens).loss;
    const auto forward_end = std::chrono::steady_clock::now();
    forward_pass_seconds_cumulative +=
        std::chrono::duration<double>(forward_end - forward_start).count();
  }
  const double total_program_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    benchmark_start)
          .count();

  std::cout << "mode=benchmark "
            << "preset=" << options.label << ' '
            << "requested_steps=" << requested_steps << ' ' << "steps=" << steps
            << ' ' << "last_doc=" << last_doc << ' '
            << "loss=" << std::setprecision(8) << last_loss << ' '
            << "forward_pass_seconds_cumulative=" << std::setprecision(8)
            << forward_pass_seconds_cumulative << ' '
            << "total_program_seconds=" << std::setprecision(8)
            << total_program_seconds << '\n';
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const CliOptions options = parse_cli(argc, argv);
    if (options.mode == "validate") {
      return run_validate(options);
    }
    if (options.mode == "benchmark") {
      return run_benchmark(options);
    }
    throw std::runtime_error("unsupported mode: " + options.mode);
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
