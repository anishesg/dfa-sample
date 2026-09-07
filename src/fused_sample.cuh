#pragma once
#include "dfa_types.cuh"
#include <cuda_runtime.h>

struct SampleResult {
    int   token_id;
    int   next_state;
    float log_prob;
};

// Fused constrained top-k sampling.
// Performs DFA masking, temperature scaling, register-heap top-k, softmax, and
// categorical sampling in a single kernel launch with no intermediate global writes.
//
// d_logits:   (vocab_size,) float32 logit vector in device memory (read-only).
// dfa:        GPU-resident DFA with precomputed bitmasks and transition table.
// state:      Current DFA state index.
// top_k:      Number of top candidates to sample from (1-256).
// temperature: Softmax temperature (>0).
// rand_u:     Uniform random value in [0, 1) for categorical sampling.
// d_result:   Output SampleResult in device memory (token_id, next_state, log_prob).
void launch_fused_sample(
    const float* d_logits,
    const DFADevice& dfa,
    int state,
    int top_k,
    float temperature,
    float rand_u,
    SampleResult* d_result,
    cudaStream_t stream = 0);
