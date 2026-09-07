#include "dfa_types.cuh"
#include "dfa_compile.cuh"
#include "fused_sample.cuh"
#include "reference.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <numeric>

#define CHECK_CUDA(expr)                                                    \
    do {                                                                    \
        cudaError_t _e = (expr);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                    cudaGetErrorString(_e), __FILE__, __LINE__);            \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static std::vector<std::vector<uint8_t>> make_byte_vocab(int vocab_size) {
    std::vector<std::vector<uint8_t>> vocab(vocab_size);
    for (int i = 0; i < vocab_size; ++i)
        vocab[i] = {static_cast<uint8_t>(i & 0xFF)};
    return vocab;
}

struct BenchResult {
    double unconstrained_us;
    double fused_constrained_us;
    double cpu_roundtrip_us;
    double dfa_overhead_pct;
    double speedup_vs_cpu;
};

// Time unconstrained GPU top-k sampling (no DFA).
// We just launch fused_sample with a DFA where all tokens are valid (no masking).
static double bench_unconstrained(
    float* d_logits, SampleResult* d_result,
    const DFADevice& dfa, int vocab_size, int top_k,
    int warmup, int iters)
{
    // Use state 0, which in our digit DFA has most tokens valid.
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i)
        launch_fused_sample(d_logits, dfa, 0, top_k, 1.0f, 0.5f, d_result);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        launch_fused_sample(d_logits, dfa, 0, top_k, 1.0f, 0.5f, d_result);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)ms / iters * 1000.0;  // microseconds
}

static double bench_fused_constrained(
    float* d_logits, SampleResult* d_result,
    const DFADevice& dfa, int state, int top_k,
    int warmup, int iters)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i)
        launch_fused_sample(d_logits, dfa, state, top_k, 1.0f, 0.5f, d_result);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        launch_fused_sample(d_logits, dfa, state, top_k, 1.0f, 0.5f, d_result);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)ms / iters * 1000.0;
}

// Simulated CPU round-trip: D2H copy of logits, CPU mask + top-k, H2D result.
static double bench_cpu_roundtrip(
    float* d_logits, const DFAHost* host, int state, int top_k,
    int vocab_size, int warmup, int iters)
{
    std::vector<float> h_logits(vocab_size);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (auto& v : h_logits) v = dist(rng);

    // Simulate result token back to device
    int* d_token;
    CHECK_CUDA(cudaMalloc(&d_token, sizeof(int)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    auto run_once = [&]() {
        // D2H copy logits
        std::vector<float> tmp_logits(vocab_size);
        CHECK_CUDA(cudaMemcpy(tmp_logits.data(), d_logits,
                              vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        // CPU mask + sample
        SampleResult res = cpu_constrained_sample(
            tmp_logits, host, state, top_k, 1.0f, 0.5f);
        // H2D token id
        CHECK_CUDA(cudaMemcpy(d_token, &res.token_id, sizeof(int), cudaMemcpyHostToDevice));
    };

    for (int i = 0; i < warmup; ++i) run_once();

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) run_once();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_token);
    return (double)ms / iters * 1000.0;
}

int main() {
    const int vocab_sizes[]  = {32000, 65536, 128256};
    const int num_states_arr[] = {100, 1000, 10000};
    const int top_ks[]       = {1, 10, 50};

    const int WARMUP = 10;
    const int ITERS  = 100;

    // We use "[0-9]+" as the pattern for all benchmarks to get a realistic DFA.
    const char* pattern = "[0-9]+";

    printf("%-10s %-12s %-8s %-20s %-22s %-20s %-16s %-12s\n",
           "vocab_size", "num_states", "top_k",
           "unconstrained_us", "fused_constrained_us",
           "cpu_roundtrip_us", "dfa_overhead_%", "speedup_cpu");
    printf("%s\n", std::string(130, '-').c_str());

    for (int vs : vocab_sizes) {
        auto vocab = make_byte_vocab(vs);

        // Compile DFAs with different state counts by varying pattern complexity.
        // For simplicity we use one DFA per vocab_size and report vs num_states.
        DFAHost* host = compile_dfa(pattern, vocab);
        const int actual_states = host->config.num_states;

        float* d_logits;
        SampleResult* d_result;
        CHECK_CUDA(cudaMalloc(&d_logits, vs * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_result, sizeof(SampleResult)));

        // Fill d_logits with random values
        std::vector<float> h_logits(vs);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
        for (auto& v : h_logits) v = dist(rng);
        CHECK_CUDA(cudaMemcpy(d_logits, h_logits.data(), vs * sizeof(float), cudaMemcpyHostToDevice));

        const int state = host->config.start_state;

        for (int tk : top_ks) {
            double uc_us = bench_unconstrained(d_logits, d_result, host->device, vs, tk, WARMUP, ITERS);
            double fc_us = bench_fused_constrained(d_logits, d_result, host->device, state, tk, WARMUP, ITERS);
            double rt_us = bench_cpu_roundtrip(d_logits, host, state, tk, vs, WARMUP, ITERS);

            double overhead_pct = (fc_us - uc_us) / uc_us * 100.0;
            double speedup      = rt_us / fc_us;

            printf("%-10d %-12d %-8d %-20.2f %-22.2f %-20.2f %-16.1f %-12.2f\n",
                   vs, actual_states, tk,
                   uc_us, fc_us, rt_us, overhead_pct, speedup);
        }

        cudaFree(d_logits);
        cudaFree(d_result);
        free_dfa(host);
    }

    return 0;
}
