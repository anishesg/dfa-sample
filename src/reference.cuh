#pragma once

#include "dfa_types.cuh"
#include "dfa_compile.cuh"
#include "fused_sample.cuh"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <cfloat>

// CPU reference implementation of constrained top-k sampling.
// Produces identical results to the GPU fused kernel given the same inputs.
//
// logits:      Host-side logit vector of length vocab_size.
// host:        Compiled DFA (host arrays).
// state:       Current DFA state.
// top_k:       Number of top candidates.
// temperature: Softmax temperature (>0).
// rand_u:      Uniform random value in [0, 1) for categorical sampling.
//
// Returns (token_id, next_state, log_prob).
static SampleResult cpu_constrained_sample(
    const std::vector<float>& logits,
    const DFAHost* host,
    int state,
    int top_k,
    float temperature,
    float rand_u)
{
    const int vocab_size    = host->config.vocab_size;
    const int bitmask_words = host->config.bitmask_words;
    const float inv_temp    = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;

    // Step 1: collect valid (scaled_logit, token_id) pairs
    std::vector<std::pair<float, int>> valid;
    valid.reserve(vocab_size);

    for (int tok = 0; tok < vocab_size; ++tok) {
        const int word_idx = tok >> 5;
        const int bit_idx  = tok & 31;
        bool is_valid = (host->h_bitmasks[state * bitmask_words + word_idx] >> bit_idx) & 1u;
        if (is_valid) {
            valid.emplace_back(logits[tok] * inv_temp, tok);
        }
    }

    if (valid.empty()) {
        return {-1, DFA_DEAD_STATE, -FLT_MAX};
    }

    // Step 2: partial sort to get top-k largest (descending)
    int k = std::min(top_k, (int)valid.size());
    std::partial_sort(valid.begin(), valid.begin() + k, valid.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    valid.resize(k);

    // Step 3: softmax over the k survivors
    float max_val = valid[0].first;
    std::vector<float> probs(k);
    float sum_exp = 0.0f;
    for (int i = 0; i < k; ++i) {
        probs[i] = std::exp(valid[i].first - max_val);
        sum_exp += probs[i];
    }
    for (int i = 0; i < k; ++i) probs[i] /= sum_exp;

    // Step 4: CDF-based categorical sampling
    float cdf = 0.0f;
    int selected     = valid[0].second;
    float sel_prob   = probs[0];
    for (int i = 0; i < k; ++i) {
        cdf += probs[i];
        if (rand_u <= cdf) {
            selected = valid[i].second;
            sel_prob = probs[i];
            break;
        }
    }

    // Step 5: next-state lookup via host transition table
    int cur = state;
    const int seq_start = host->h_token_byte_offsets[selected];
    const int seq_end   = host->h_token_byte_offsets[selected + 1];
    for (int bi = seq_start; bi < seq_end && cur >= 0; ++bi)
        cur = host->h_transitions[cur * ALPHABET_SIZE + host->h_token_bytes[bi]];

    SampleResult result;
    result.token_id   = selected;
    result.next_state = cur;
    result.log_prob   = std::log(std::max(sel_prob, 1e-30f));
    return result;
}

// Apply DFA mask to a host-side logit vector in-place.
// Sets invalid positions to -FLT_MAX.
static void cpu_apply_dfa_mask(
    std::vector<float>& logits,
    const DFAHost* host,
    int state)
{
    const int vocab_size    = host->config.vocab_size;
    const int bitmask_words = host->config.bitmask_words;
    for (int tok = 0; tok < vocab_size; ++tok) {
        const int word_idx = tok >> 5;
        const int bit_idx  = tok & 31;
        bool valid = (host->h_bitmasks[state * bitmask_words + word_idx] >> bit_idx) & 1u;
        if (!valid) logits[tok] = -FLT_MAX;
    }
}

// Return a host-side bitmask vector for a given state (for comparison).
static std::vector<bool> cpu_valid_mask(const DFAHost* host, int state) {
    const int vocab_size    = host->config.vocab_size;
    const int bitmask_words = host->config.bitmask_words;
    std::vector<bool> mask(vocab_size);
    for (int tok = 0; tok < vocab_size; ++tok) {
        const int word_idx = tok >> 5;
        const int bit_idx  = tok & 31;
        mask[tok] = (host->h_bitmasks[state * bitmask_words + word_idx] >> bit_idx) & 1u;
    }
    return mask;
}
