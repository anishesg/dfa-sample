#include "dfa_types.cuh"
#include "dfa_mask.cuh"

#include <cuda_runtime.h>
#include <float.h>

// Warp-cooperative coalesced load of the bitmask for `state` into shared memory.
// smem must hold at least bitmask_words uint32 values.
__device__ __forceinline__
void load_bitmask_to_smem(const DFADevice& dfa, int state,
                           uint32_t* __restrict__ smem) {
    const int bw    = dfa.config.bitmask_words;
    const uint32_t* src = dfa.bitmasks + (size_t)state * bw;
    for (int i = threadIdx.x; i < bw; i += blockDim.x)
        smem[i] = src[i];
    __syncthreads();
}

// Apply DFA mask to logits in-place for the given state.
// Logits at invalid token positions are set to -INFINITY.
// Grid: (vocab_size + blockDim.x - 1) / blockDim.x blocks, each with BLOCK_SIZE threads.
__global__ void dfa_mask_kernel(
    float* __restrict__ logits,
    DFADevice dfa,
    int state,
    int vocab_size)
{
    extern __shared__ uint32_t smem_bitmask[];

    load_bitmask_to_smem(dfa, state, smem_bitmask);

    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= vocab_size) return;

    const int word_idx = gid >> 5;
    const int bit_idx  = gid & 31;
    const bool valid   = (smem_bitmask[word_idx] >> bit_idx) & 1u;
    if (!valid) logits[gid] = -FLT_MAX;
}

// Compute grid/block and launch dfa_mask_kernel.
void launch_dfa_mask(
    float* d_logits,
    const DFADevice& dfa,
    int state,
    cudaStream_t stream)
{
    const int vocab_size    = dfa.config.vocab_size;
    const int bitmask_words = dfa.config.bitmask_words;
    const int BLOCK         = 256;
    const int grid          = (vocab_size + BLOCK - 1) / BLOCK;
    const size_t smem_bytes = bitmask_words * sizeof(uint32_t);

    dfa_mask_kernel<<<grid, BLOCK, smem_bytes, stream>>>(
        d_logits, dfa, state, vocab_size);
}
