#include "fused_sample.cuh"
#include "dfa_types.cuh"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

// Maximum top_k supported by the register-resident heap.
static constexpr int MAX_K = 256;
static constexpr int BLOCK_SIZE = 256;

// ---- Register-resident min-heap ----
// Stores the top-k (value, token_id) pairs. The min is at index 0.
// We use a flat array heap with standard binary heap operations.
struct MinHeap {
    float vals[MAX_K];
    int   ids[MAX_K];
    int   size;
    int   capacity;

    __device__ __forceinline__ void init(int k) {
        size     = 0;
        capacity = k;
    }

    __device__ __forceinline__ void sift_down(int i) {
        while (true) {
            int left  = 2 * i + 1;
            int right = 2 * i + 2;
            int smallest = i;
            if (left  < size && vals[left]  < vals[smallest]) smallest = left;
            if (right < size && vals[right] < vals[smallest]) smallest = right;
            if (smallest == i) break;
            float tv = vals[i]; vals[i] = vals[smallest]; vals[smallest] = tv;
            int   ti = ids[i];  ids[i]  = ids[smallest];  ids[smallest]  = ti;
            i = smallest;
        }
    }

    __device__ __forceinline__ void sift_up(int i) {
        while (i > 0) {
            int parent = (i - 1) / 2;
            if (vals[parent] <= vals[i]) break;
            float tv = vals[i]; vals[i] = vals[parent]; vals[parent] = tv;
            int   ti = ids[i];  ids[i]  = ids[parent];  ids[parent]  = ti;
            i = parent;
        }
    }

    __device__ __forceinline__ void push(float v, int id) {
        if (size < capacity) {
            vals[size] = v;
            ids[size]  = id;
            ++size;
            sift_up(size - 1);
        } else if (v > vals[0]) {
            // Replace the min (root)
            vals[0] = v;
            ids[0]  = id;
            sift_down(0);
        }
    }

    // Sort the heap in descending order (in-place heapsort).
    __device__ __forceinline__ void sort_descending() {
        // Build max-heap from existing min-heap elements then extract
        // Simpler: just insertion-sort for small K (K <= 256 is fine in registers)
        for (int i = 1; i < size; ++i) {
            float kv = vals[i];
            int   ki = ids[i];
            int   j  = i - 1;
            while (j >= 0 && vals[j] < kv) {
                vals[j+1] = vals[j];
                ids[j+1]  = ids[j];
                --j;
            }
            vals[j+1] = kv;
            ids[j+1]  = ki;
        }
    }
};

// Warp-level max reduction
__device__ __forceinline__ float warp_max(float v) {
    for (int off = 16; off >= 1; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

// Warp-level sum reduction
__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off >= 1; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// ---- Fused kernel ----
// One thread block handles the entire vocabulary.
// Phase 1: load bitmask into shared memory.
// Phase 2: tile through logits; each thread maintains a local min-heap.
// Phase 3: merge per-thread heaps into a block-level sorted top-k list.
// Phase 4: softmax over top-k survivors using warp reductions.
// Phase 5: CDF-based categorical sampling.
// Phase 6: next-state lookup.
__global__ void fused_constrained_sample_kernel(
    const float* __restrict__ d_logits,
    DFADevice dfa,
    int state,
    int top_k,
    float inv_temperature,
    float rand_u,
    SampleResult* __restrict__ d_result)
{
    const int vocab_size    = dfa.config.vocab_size;
    const int bitmask_words = dfa.config.bitmask_words;

    // Shared memory layout:
    //   [0 .. bitmask_words)          : bitmask for current state
    //   [bitmask_words .. bitmask_words + top_k)  : merged top-k values (float)
    //   [bitmask_words + top_k .. bitmask_words + 2*top_k) : merged top-k ids (int)
    extern __shared__ uint32_t smem[];
    uint32_t* smem_bitmask = smem;
    float*    smem_vals    = (float*)(smem + bitmask_words);
    int*      smem_ids     = (int*)(smem_vals + top_k);

    // Phase 1: cooperative bitmask load
    const uint32_t* bm_src = dfa.bitmasks + (size_t)state * bitmask_words;
    for (int i = threadIdx.x; i < bitmask_words; i += blockDim.x)
        smem_bitmask[i] = bm_src[i];
    __syncthreads();

    // Phase 2: build per-thread min-heap over valid logits
    MinHeap heap;
    heap.init(top_k);

    for (int tok = threadIdx.x; tok < vocab_size; tok += blockDim.x) {
        const bool valid = (smem_bitmask[tok >> 5] >> (tok & 31)) & 1u;
        if (!valid) continue;
        float scaled = d_logits[tok] * inv_temperature;
        heap.push(scaled, tok);
    }

    // Phase 3: merge per-thread heaps into shared memory top-k.
    // Strategy: each thread writes its heap root (the min) to shared memory for
    // a parallel reduction. Repeated top_k times. This is O(top_k * blockDim) per
    // block -- acceptable for top_k <= 256 and BLOCK_SIZE=256.
    //
    // Simpler and correct: serialize merging via a global shared-memory heap.
    // Use a tournament: all threads atomically contribute their best candidates.
    //
    // We use the following approach:
    //   a. Each thread writes its heap, sorted descending, to a per-thread slot in smem.
    //   b. Thread 0 does a k-way merge.
    // For BLOCK_SIZE=256 and K=256 this is fine.
    //
    // Allocate per-thread heap slots in smem beyond the top_k result area.
    // Actually we keep it simpler: each thread writes its local heap to smem in
    // a strided layout, then a single-pass scan by thread 0 collects top-k.

    heap.sort_descending();

    // Write local heap to per-thread region of smem (beyond the bitmask + result area).
    // smem layout: [bitmask_words | top_k floats | top_k ints | blockDim.x * top_k floats | blockDim.x * top_k ints]
    float* local_vals_smem = smem_vals + top_k + (size_t)threadIdx.x * top_k;
    int*   local_ids_smem  = smem_ids  + top_k + (size_t)threadIdx.x * top_k;

    for (int i = 0; i < heap.size && i < top_k; ++i) {
        local_vals_smem[i] = heap.vals[i];
        local_ids_smem[i]  = heap.ids[i];
    }
    // Pad remaining with -FLT_MAX
    for (int i = heap.size; i < top_k; ++i) {
        local_vals_smem[i] = -FLT_MAX;
        local_ids_smem[i]  = -1;
    }
    __syncthreads();

    // Phase 3b: thread 0 merges all per-thread heaps into smem_vals/smem_ids[0..top_k)
    if (threadIdx.x == 0) {
        // Global min-heap of size top_k for merging
        MinHeap global_heap;
        global_heap.init(top_k);

        for (int t = 0; t < blockDim.x; ++t) {
            float* tv = smem_vals + top_k + (size_t)t * top_k;
            int*   ti = smem_ids  + top_k + (size_t)t * top_k;
            for (int i = 0; i < top_k; ++i) {
                if (ti[i] < 0) break;
                global_heap.push(tv[i], ti[i]);
            }
        }
        global_heap.sort_descending();
        for (int i = 0; i < global_heap.size; ++i) {
            smem_vals[i] = global_heap.vals[i];
            smem_ids[i]  = global_heap.ids[i];
        }
        for (int i = global_heap.size; i < top_k; ++i) {
            smem_vals[i] = -FLT_MAX;
            smem_ids[i]  = -1;
        }
    }
    __syncthreads();

    // Phase 4: softmax over top-k survivors (parallel across lanes)
    // Use warp-level reductions for max and sum.
    int k_actual = 0;
    for (int i = 0; i < top_k; ++i)
        if (smem_ids[i] >= 0) ++k_actual;

    if (k_actual == 0) {
        // No valid tokens: return -1 as error
        if (threadIdx.x == 0) {
            d_result->token_id  = -1;
            d_result->next_state = DFA_DEAD_STATE;
            d_result->log_prob   = -FLT_MAX;
        }
        return;
    }

    // Each thread handles one top-k entry
    float local_val  = (threadIdx.x < k_actual) ? smem_vals[threadIdx.x] : -FLT_MAX;
    float global_max = warp_max(local_val);
    // Broadcast max across all warps via smem
    __shared__ float smem_max;
    if (threadIdx.x == 0) smem_max = -FLT_MAX;
    __syncthreads();
    if (threadIdx.x % 32 == 0) atomicMax((int*)&smem_max, __float_as_int(global_max));
    __syncthreads();
    global_max = __int_as_float(*(int*)&smem_max);

    float exp_val = (threadIdx.x < k_actual) ? expf(local_val - global_max) : 0.0f;
    float sum_exp = warp_sum(exp_val);
    __shared__ float smem_sum;
    if (threadIdx.x == 0) smem_sum = 0.0f;
    __syncthreads();
    if (threadIdx.x % 32 == 0) atomicAdd(&smem_sum, sum_exp);
    __syncthreads();
    float total_sum = smem_sum;

    // Write probabilities back to smem_vals for CDF scan
    if (threadIdx.x < k_actual)
        smem_vals[threadIdx.x] = exp_val / total_sum;
    __syncthreads();

    // Phase 5: CDF-based sampling (thread 0 only, sequential scan)
    if (threadIdx.x == 0) {
        float cdf = 0.0f;
        int selected = smem_ids[0];
        float selected_prob = smem_vals[0];
        for (int i = 0; i < k_actual; ++i) {
            cdf += smem_vals[i];
            if (rand_u <= cdf) {
                selected      = smem_ids[i];
                selected_prob = smem_vals[i];
                break;
            }
        }

        // Phase 6: next-state lookup via transition table
        int next = get_next_state(dfa, state, selected);

        d_result->token_id   = selected;
        d_result->next_state = next;
        d_result->log_prob   = logf(fmaxf(selected_prob, 1e-30f));
    }
}

void launch_fused_sample(
    const float* d_logits,
    const DFADevice& dfa,
    int state,
    int top_k,
    float temperature,
    float rand_u,
    SampleResult* d_result,
    cudaStream_t stream)
{
    if (top_k > MAX_K) top_k = MAX_K;
    if (top_k < 1)     top_k = 1;

    const int bitmask_words = dfa.config.bitmask_words;
    // smem: bitmask | top_k result vals | top_k result ids |
    //        BLOCK_SIZE * top_k local vals | BLOCK_SIZE * top_k local ids
    const size_t smem_bytes =
        bitmask_words   * sizeof(uint32_t)    +  // bitmask
        top_k           * sizeof(float)       +  // merged vals
        top_k           * sizeof(int)         +  // merged ids
        BLOCK_SIZE * top_k * sizeof(float)    +  // per-thread vals
        BLOCK_SIZE * top_k * sizeof(int);        // per-thread ids

    const float inv_temp = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;

    fused_constrained_sample_kernel<<<1, BLOCK_SIZE, smem_bytes, stream>>>(
        d_logits, dfa, state, top_k, inv_temp, rand_u, d_result);
}
