#pragma once
#include "dfa_types.cuh"
#include <cuda_runtime.h>

// Apply DFA validity mask to a float logit vector in global memory.
// Invalid token positions are set to -FLT_MAX.
// state: current DFA state index (0-based).
// stream: CUDA stream to launch on (0 = default).
void launch_dfa_mask(float* d_logits, const DFADevice& dfa, int state,
                     cudaStream_t stream = 0);
