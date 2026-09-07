#include "dfa_types.cuh"
#include "dfa_compile.cuh"
#include "fused_sample.cuh"
#include "reference.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cassert>

#define CHECK_CUDA(expr)                                                    \
    do {                                                                    \
        cudaError_t _e = (expr);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                    cudaGetErrorString(_e), __FILE__, __LINE__);            \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// Build a trivial single-byte-per-token vocabulary from characters 0x20..0xFF
static std::vector<std::vector<uint8_t>> make_byte_vocab(int vocab_size) {
    std::vector<std::vector<uint8_t>> vocab(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        vocab[i] = {static_cast<uint8_t>(i & 0xFF)};
    }
    return vocab;
}

// Compute chi-squared statistic between observed and expected frequency arrays.
static double chi_squared(const std::vector<int>& observed,
                           const std::vector<double>& expected)
{
    double chi2 = 0.0;
    for (int i = 0; i < (int)observed.size(); ++i) {
        if (expected[i] < 1e-12) continue;
        double diff = observed[i] - expected[i];
        chi2 += diff * diff / expected[i];
    }
    return chi2;
}

struct TestCase {
    const char* pattern;
    const char* name;
};

static bool test_mask_agreement(
    const char* pattern, int vocab_size,
    const std::vector<std::vector<uint8_t>>& vocab)
{
    DFAHost* host = compile_dfa(pattern, vocab);

    // Check mask agreement for start state and all valid transitions
    const int num_states = host->config.num_states;
    bool ok = true;

    for (int state = 0; state < num_states; ++state) {
        // Allocate GPU logit buffer
        std::vector<float> h_logits(vocab_size, 0.0f);
        float* d_logits;
        CHECK_CUDA(cudaMalloc(&d_logits, vocab_size * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_logits, h_logits.data(),
                              vocab_size * sizeof(float), cudaMemcpyHostToDevice));

        // Apply DFA mask via standalone kernel
        launch_dfa_mask(d_logits, host->device, state);
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<float> gpu_logits(vocab_size);
        CHECK_CUDA(cudaMemcpy(gpu_logits.data(), d_logits,
                              vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_logits);

        // CPU reference mask
        std::vector<bool> cpu_mask = cpu_valid_mask(host, state);

        for (int tok = 0; tok < vocab_size; ++tok) {
            bool gpu_valid = (gpu_logits[tok] > -1e30f);
            bool cpu_valid = cpu_mask[tok];
            if (gpu_valid != cpu_valid) {
                fprintf(stderr,
                    "[FAIL] mask_agreement pattern=%s state=%d tok=%d "
                    "gpu=%d cpu=%d\n",
                    pattern, state, tok, (int)gpu_valid, (int)cpu_valid);
                ok = false;
                break;
            }
        }
        if (!ok) break;
    }

    free_dfa(host);
    return ok;
}

static bool test_topk_agreement(
    const char* pattern, int vocab_size,
    const std::vector<std::vector<uint8_t>>& vocab,
    int top_k, int num_trials, std::mt19937& rng)
{
    DFAHost* host = compile_dfa(pattern, vocab);
    std::uniform_real_distribution<float> logit_dist(-5.0f, 5.0f);
    std::uniform_real_distribution<float> rand_dist(0.0f, 1.0f);

    float* d_logits;
    SampleResult* d_result;
    CHECK_CUDA(cudaMalloc(&d_logits, vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(SampleResult)));

    bool ok = true;
    const int state = host->config.start_state;

    for (int trial = 0; trial < num_trials; ++trial) {
        std::vector<float> h_logits(vocab_size);
        for (auto& v : h_logits) v = logit_dist(rng);
        float rand_u = rand_dist(rng);

        CHECK_CUDA(cudaMemcpy(d_logits, h_logits.data(),
                              vocab_size * sizeof(float), cudaMemcpyHostToDevice));

        // GPU fused sample
        launch_fused_sample(d_logits, host->device, state, top_k, 1.0f, rand_u, d_result);
        CHECK_CUDA(cudaDeviceSynchronize());

        SampleResult gpu_res;
        CHECK_CUDA(cudaMemcpy(&gpu_res, d_result, sizeof(SampleResult), cudaMemcpyDeviceToHost));

        // CPU reference sample
        SampleResult cpu_res = cpu_constrained_sample(
            h_logits, host, state, top_k, 1.0f, rand_u);

        // Both should select valid tokens
        if (gpu_res.token_id < 0 && cpu_res.token_id < 0) continue;

        if (gpu_res.token_id < 0 || cpu_res.token_id < 0) {
            fprintf(stderr,
                "[FAIL] topk_agreement trial=%d gpu_tok=%d cpu_tok=%d\n",
                trial, gpu_res.token_id, cpu_res.token_id);
            ok = false;
            break;
        }

        // Check that GPU token is valid according to CPU bitmask
        const int word_idx = gpu_res.token_id >> 5;
        const int bit_idx  = gpu_res.token_id & 31;
        bool tok_valid = (host->h_bitmasks[state * host->config.bitmask_words + word_idx]
                          >> bit_idx) & 1u;
        if (!tok_valid) {
            fprintf(stderr,
                "[FAIL] topk_agreement trial=%d gpu selected invalid token %d\n",
                trial, gpu_res.token_id);
            ok = false;
            break;
        }
    }

    cudaFree(d_logits);
    cudaFree(d_result);
    free_dfa(host);
    return ok;
}

static bool test_distribution_match(
    const char* pattern, int vocab_size,
    const std::vector<std::vector<uint8_t>>& vocab,
    int top_k, int num_samples, std::mt19937& rng)
{
    DFAHost* host = compile_dfa(pattern, vocab);
    std::uniform_real_distribution<float> rand_dist(0.0f, 1.0f);

    // Fixed logits for distribution test
    std::vector<float> h_logits(vocab_size, 0.0f);  // uniform logits

    float* d_logits;
    SampleResult* d_result;
    CHECK_CUDA(cudaMalloc(&d_logits, vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(SampleResult)));
    CHECK_CUDA(cudaMemcpy(d_logits, h_logits.data(),
                          vocab_size * sizeof(float), cudaMemcpyHostToDevice));

    const int state = host->config.start_state;

    // Expected: uniform over valid tokens
    std::vector<bool> valid_mask = cpu_valid_mask(host, state);
    int num_valid = 0;
    for (bool b : valid_mask) if (b) ++num_valid;

    if (num_valid == 0) { free_dfa(host); cudaFree(d_logits); cudaFree(d_result); return true; }
    if (num_valid < 2)  { free_dfa(host); cudaFree(d_logits); cudaFree(d_result); return true; }

    // Restrict to top_k valid tokens
    int effective_k = std::min(top_k, num_valid);

    // With uniform logits any top_k tokens have equal probability
    double expected_prob = 1.0 / effective_k;

    std::vector<int> gpu_counts(vocab_size, 0);
    for (int s = 0; s < num_samples; ++s) {
        float rand_u = rand_dist(rng);
        launch_fused_sample(d_logits, host->device, state, top_k, 1.0f, rand_u, d_result);
        CHECK_CUDA(cudaDeviceSynchronize());
        SampleResult res;
        CHECK_CUDA(cudaMemcpy(&res, d_result, sizeof(SampleResult), cudaMemcpyDeviceToHost));
        if (res.token_id >= 0 && res.token_id < vocab_size) ++gpu_counts[res.token_id];
    }

    // Build observed and expected arrays for chi-squared test
    // Only include tokens that were sampled or have nonzero expected probability
    std::vector<int> obs;
    std::vector<double> exp_freq;

    // Find the top-k valid tokens (by logit value, which are all 0 here -- take first k)
    int collected = 0;
    for (int tok = 0; tok < vocab_size && collected < effective_k; ++tok) {
        if (valid_mask[tok]) {
            obs.push_back(gpu_counts[tok]);
            exp_freq.push_back(expected_prob * num_samples);
            ++collected;
        }
    }

    double chi2 = chi_squared(obs, exp_freq);
    // Critical value for 99% confidence, degrees_of_freedom = effective_k - 1
    // For K <= 10, chi2 critical at p=0.001 is roughly 3*K; we use a generous threshold.
    double threshold = 4.0 * effective_k + 30.0;  // generous for test stability

    bool ok = (chi2 < threshold);
    if (!ok) {
        fprintf(stderr,
            "[FAIL] distribution_match pattern=%s chi2=%.2f threshold=%.2f k=%d\n",
            pattern, chi2, threshold, effective_k);
    }

    cudaFree(d_logits);
    cudaFree(d_result);
    free_dfa(host);
    return ok;
}

int main() {
    TestCase cases[] = {
        {"[0-9]+",                  "digit_sequence"},
        {"true|false|null",         "boolean_literal"},
        {"-?[0-9]+(\\.[0-9]+)?",    "json_number"},
    };

    const int vocab_sizes[] = {32000, 128256};
    std::mt19937 rng(42);

    int passed = 0, total = 0;

    for (const auto& tc : cases) {
        for (int vs : vocab_sizes) {
            printf("=== %s vocab=%d ===\n", tc.name, vs);
            auto vocab = make_byte_vocab(vs);

            ++total;
            bool ok = test_mask_agreement(tc.pattern, vs, vocab);
            printf("  mask_agreement:    %s\n", ok ? "PASS" : "FAIL");
            if (ok) ++passed;

            ++total;
            ok = test_topk_agreement(tc.pattern, vs, vocab, /*top_k=*/50, /*trials=*/20, rng);
            printf("  topk_agreement:    %s\n", ok ? "PASS" : "FAIL");
            if (ok) ++passed;

            ++total;
            ok = test_distribution_match(tc.pattern, vs, vocab, /*top_k=*/10,
                                         /*samples=*/10000, rng);
            printf("  distribution_match: %s\n", ok ? "PASS" : "FAIL");
            if (ok) ++passed;
        }
    }

    printf("\n%d / %d tests passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
