#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// Maximum vocabulary size supported (128K for large tokenizers)
static constexpr int MAX_VOCAB_SIZE = 131072;
// Maximum bytes per token in byte-sequence representation
static constexpr int MAX_TOKEN_BYTES = 32;
// Maximum alphabet size for DFA transitions (byte values 0-255)
static constexpr int ALPHABET_SIZE = 256;

struct DFAConfig {
    int num_states;
    int vocab_size;
    int start_state;
    int num_accepting;
    // ceil(vocab_size / 32) words per bitmask row
    int bitmask_words;
};

// All pointer members point to device memory
struct DFADevice {
    // Transition table: [num_states * ALPHABET_SIZE] -> next_state (or -1 for dead state)
    // Row-major: transitions[state * ALPHABET_SIZE + byte_val]
    int32_t* transitions;

    // Per-state vocabulary bitmasks: [num_states * bitmask_words] uint32
    // Bit i of row s is set if token i is valid from state s.
    // Row-major: bitmasks[state * bitmask_words + (token_id / 32)]
    uint32_t* bitmasks;

    // Token byte sequences packed as flat arrays
    // token_bytes[token_byte_offsets[i] .. token_byte_offsets[i+1]) is the byte sequence for token i
    uint8_t* token_bytes;
    int32_t* token_byte_offsets;  // length vocab_size + 1

    DFAConfig config;
};

// Check whether token_id is valid from the given DFA state using the precomputed bitmask.
__device__ __forceinline__
bool is_token_valid(const DFADevice& dfa, int state, int token_id) {
    const int word_idx = token_id >> 5;       // token_id / 32
    const int bit_idx  = token_id & 31;       // token_id % 32
    const uint32_t word = dfa.bitmasks[state * dfa.config.bitmask_words + word_idx];
    return (word >> bit_idx) & 1u;
}

// Advance the DFA through one byte, returning the next state.
// Returns -1 if the transition is undefined (dead state).
__device__ __forceinline__
int dfa_step(const DFADevice& dfa, int state, uint8_t byte_val) {
    if (state < 0) return -1;
    return dfa.transitions[state * ALPHABET_SIZE + byte_val];
}

// Compute the DFA state reached by consuming the byte sequence of token_id from start_state.
// Returns -1 if the token leads to a dead state (should match is_token_valid == false).
__device__ __forceinline__
int get_next_state(const DFADevice& dfa, int state, int token_id) {
    const int seq_start = dfa.token_byte_offsets[token_id];
    const int seq_end   = dfa.token_byte_offsets[token_id + 1];
    int cur = state;
    for (int i = seq_start; i < seq_end && cur >= 0; ++i) {
        cur = dfa_step(dfa, cur, dfa.token_bytes[i]);
    }
    return cur;
}

// Host-side mirror of DFADevice that owns pinned/device allocations.
struct DFAHost {
    DFADevice device;          // struct with device pointers, ready to copy to GPU
    DFAConfig config;

    // Flat host-side copies for inspection/debugging
    int32_t* h_transitions;    // [num_states * ALPHABET_SIZE]
    uint32_t* h_bitmasks;      // [num_states * bitmask_words]
    uint8_t*  h_token_bytes;
    int32_t*  h_token_byte_offsets;
    int       total_token_bytes;
};

// Sentinel representing a dead/invalid DFA state
static constexpr int DFA_DEAD_STATE = -1;
