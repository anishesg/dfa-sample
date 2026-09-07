#include "dfa_types.cuh"
#include "dfa_compile.cuh"
#include "fused_sample.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <random>
#include <regex>
#include <algorithm>

#define CHECK_CUDA(expr)                                                    \
    do {                                                                    \
        cudaError_t _e = (expr);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                    cudaGetErrorString(_e), __FILE__, __LINE__);            \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// Vocabulary where token i is the byte sequence for character i (single-byte tokens).
static std::vector<std::vector<uint8_t>> make_byte_vocab(int vocab_size) {
    std::vector<std::vector<uint8_t>> vocab(vocab_size);
    for (int i = 0; i < vocab_size; ++i)
        vocab[i] = {static_cast<uint8_t>(i & 0xFF)};
    return vocab;
}

// Generate up to max_tokens tokens autoregressively under DFA constraint.
// Returns the concatenated byte string of selected tokens.
static std::vector<uint8_t> generate_sequence(
    DFAHost* host,
    int max_tokens,
    float temperature,
    int top_k,
    std::mt19937& rng)
{
    const int vocab_size = host->config.vocab_size;

    float* d_logits;
    SampleResult* d_result;
    CHECK_CUDA(cudaMalloc(&d_logits, vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(SampleResult)));

    std::uniform_real_distribution<float> logit_dist(-3.0f, 3.0f);
    std::uniform_real_distribution<float> rand_dist(0.01f, 0.99f);

    std::vector<uint8_t> output;
    int state = host->config.start_state;

    for (int step = 0; step < max_tokens; ++step) {
        // Check if any valid tokens exist from current state
        bool any_valid = false;
        for (int w = 0; w < host->config.bitmask_words; ++w) {
            if (host->h_bitmasks[state * host->config.bitmask_words + w]) {
                any_valid = true; break;
            }
        }
        if (!any_valid) break;

        // Random logits simulating model output
        std::vector<float> h_logits(vocab_size);
        for (auto& v : h_logits) v = logit_dist(rng);
        CHECK_CUDA(cudaMemcpy(d_logits, h_logits.data(),
                              vocab_size * sizeof(float), cudaMemcpyHostToDevice));

        float rand_u = rand_dist(rng);
        launch_fused_sample(d_logits, host->device, state, top_k, temperature, rand_u, d_result);
        CHECK_CUDA(cudaDeviceSynchronize());

        SampleResult res;
        CHECK_CUDA(cudaMemcpy(&res, d_result, sizeof(SampleResult), cudaMemcpyDeviceToHost));

        if (res.token_id < 0) break;

        // Decode token bytes and append
        const int bs = host->h_token_byte_offsets[res.token_id];
        const int be = host->h_token_byte_offsets[res.token_id + 1];
        for (int b = bs; b < be; ++b)
            output.push_back(host->h_token_bytes[b]);

        state = res.next_state;
        if (state < 0) break;
    }

    cudaFree(d_logits);
    cudaFree(d_result);
    return output;
}

struct SequenceTest {
    const char* dfa_pattern;   // pattern passed to compile_dfa
    const char* regex_pattern; // std::regex pattern for end-to-end validation
    const char* name;
    int max_tokens;
};

int main() {
    const int VOCAB_SIZE = 256;   // single-byte vocabulary
    const int TOP_K      = 20;
    const float TEMP     = 1.0f;
    std::mt19937 rng(99);

    auto vocab = make_byte_vocab(VOCAB_SIZE);

    SequenceTest tests[] = {
        {
            "[a-z]+",
            "^[a-z]+$",
            "lowercase_words",
            100
        },
        {
            "[0-9]+(\\.[0-9]+)?",
            "^[0-9]+(\\.[0-9]+)?$",
            "decimal_numbers",
            100
        },
        {
            "(GET|POST|PUT) /[a-z/]+ HTTP/1\\.[01]",
            "^(GET|POST|PUT) /[a-z/]+ HTTP/1\\.[01]$",
            "http_request_line",
            100
        },
    };

    int passed = 0, total = 0;

    for (auto& tc : tests) {
        printf("=== %s ===\n", tc.name);

        DFAHost* host = compile_dfa(tc.dfa_pattern, vocab);
        printf("  DFA: %d states, %d accepting\n",
               host->config.num_states, host->config.num_accepting);

        // Generate 5 sequences per pattern
        bool all_ok = true;
        for (int trial = 0; trial < 5; ++trial) {
            auto bytes = generate_sequence(host, tc.max_tokens, TEMP, TOP_K, rng);
            std::string text(bytes.begin(), bytes.end());

            // Validate against std::regex
            std::regex re(tc.regex_pattern);
            bool matches = std::regex_match(text, re);

            // Also validate via DFA
            bool dfa_ok = dfa_accepts_host(host, bytes.data(), (int)bytes.size());

            printf("  trial %d: \"%s\" len=%d regex=%s dfa=%s\n",
                   trial, text.substr(0, 40).c_str(), (int)text.size(),
                   matches ? "PASS" : "FAIL",
                   dfa_ok ? "PASS" : "FAIL");

            if (!matches || !dfa_ok) all_ok = false;
        }

        ++total;
        if (all_ok) ++passed;
        printf("  result: %s\n\n", all_ok ? "PASS" : "FAIL");

        free_dfa(host);
    }

    printf("%d / %d sequence tests passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
