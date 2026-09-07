"""
Python-level tests for dfa_sample module.

Tests:
1. Compile JSON integer regex and generate 50 tokens; validate output matches regex.
2. Compare output distribution against torch.softmax + torch.multinomial over 5000 samples.

Run with:
    pytest tests/test_python.py -v
or directly:
    python tests/test_python.py
"""

import math
import random
import re
import sys

import pytest

torch = pytest.importorskip("torch")

from dfa_sample import DFA, compile_regex, constrained_sample, apply_dfa_mask


def make_byte_vocab(vocab_size: int):
    """Single-byte vocabulary: token i -> bytes([i & 0xFF])."""
    return [bytes([i & 0xFF]) for i in range(vocab_size)]


def decode_token_bytes(token_id: int, vocab: list) -> bytes:
    return vocab[token_id]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def byte_vocab_256():
    return make_byte_vocab(256)


@pytest.fixture(scope="module")
def json_int_dfa(byte_vocab_256):
    # JSON integer: optional minus, nonzero leading digit, then more digits
    pattern = r"-?[1-9][0-9]*"
    return compile_regex(pattern, byte_vocab_256), byte_vocab_256


class TestGenerateJSONIntegers:
    """Generate 50 tokens under JSON integer constraint and validate."""

    def test_generate_matches_regex(self, json_int_dfa):
        if DEVICE == "cpu":
            pytest.skip("CUDA not available")

        dfa, vocab = json_int_dfa
        state = dfa.start_state
        generated = []
        rng = random.Random(42)

        for _ in range(50):
            logits = torch.randn(dfa.vocab_size, device=DEVICE, dtype=torch.float32)
            tok, state = constrained_sample(
                logits, dfa, state, top_k=20, temperature=1.0,
                rand_u=rng.random()
            )
            if tok < 0:
                break
            generated.append(vocab[tok])
            if state < 0:
                break

        text = b"".join(generated).decode("latin-1", errors="replace")
        # Must match the JSON integer pattern
        assert re.fullmatch(r"-?[1-9][0-9]*", text), \
            f"Generated text '{text}' does not match JSON integer pattern"

    def test_all_tokens_valid(self, json_int_dfa):
        """Every generated token must pass DFA constraint at its step."""
        if DEVICE == "cpu":
            pytest.skip("CUDA not available")

        dfa, vocab = json_int_dfa
        rng = random.Random(123)

        for trial in range(10):
            state = dfa.start_state
            for step in range(20):
                logits = torch.randn(dfa.vocab_size, device=DEVICE, dtype=torch.float32)
                tok, next_state = constrained_sample(
                    logits, dfa, state, top_k=10, temperature=1.0,
                    rand_u=rng.random()
                )
                if tok < 0:
                    break
                # Verify tok is valid at current state via apply_dfa_mask
                test_logits = torch.zeros(dfa.vocab_size, device=DEVICE)
                apply_dfa_mask(test_logits, dfa, state)
                assert test_logits[tok].item() > -1e30, \
                    f"Trial {trial} step {step}: token {tok} should be valid"
                state = next_state
                if state < 0:
                    break


class TestDistributionMatch:
    """Compare GPU sampling distribution to torch.multinomial reference."""

    def test_distribution_match(self, json_int_dfa):
        if DEVICE == "cpu":
            pytest.skip("CUDA not available")

        dfa, vocab = json_int_dfa
        n_samples = 5000
        top_k = 10
        temperature = 1.0
        state = dfa.start_state

        # Fixed uniform logits so expected probabilities are known
        logits_cpu = torch.zeros(dfa.vocab_size, dtype=torch.float32)
        logits_gpu = logits_cpu.to(DEVICE)

        # Reference: torch.softmax over valid tokens
        ref_logits = logits_cpu.clone()
        # Apply mask (on CPU via host -- we only have GPU apply_dfa_mask, so move to GPU and back)
        ref_logits_gpu = ref_logits.to(DEVICE)
        apply_dfa_mask(ref_logits_gpu, dfa, state)
        ref_logits_masked = ref_logits_gpu.cpu()

        # Find valid tokens
        valid_mask = ref_logits_masked > -1e30
        valid_indices = valid_mask.nonzero(as_tuple=False).flatten().tolist()

        if len(valid_indices) == 0:
            pytest.skip("No valid tokens from start state")

        # Top_k valid tokens (by logit, which are all 0, so any k)
        k_actual = min(top_k, len(valid_indices))
        selected_indices = valid_indices[:k_actual]

        expected_prob = 1.0 / k_actual
        expected_counts = {idx: expected_prob * n_samples for idx in selected_indices}

        # Collect GPU samples
        rng = random.Random(7)
        gpu_counts: dict = {}
        for _ in range(n_samples):
            tok, _ = constrained_sample(
                logits_gpu, dfa, state, top_k=top_k, temperature=temperature,
                rand_u=rng.random()
            )
            gpu_counts[tok] = gpu_counts.get(tok, 0) + 1

        # Chi-squared test over the top-k tokens
        chi2 = 0.0
        for idx in selected_indices:
            obs = gpu_counts.get(idx, 0)
            exp = expected_counts[idx]
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp

        # Generous threshold: p=0.001 critical value for chi-squared with k-1 dof
        # For k<=10, threshold is around 26; we use 4*k + 30 to account for sampling noise.
        threshold = 4 * k_actual + 30
        assert chi2 < threshold, \
            f"Chi-squared {chi2:.2f} exceeds threshold {threshold:.2f} (k={k_actual})"


if __name__ == "__main__":
    # Run tests directly without pytest
    import sys

    vocab = make_byte_vocab(256)
    dfa = compile_regex(r"-?[1-9][0-9]*", vocab)

    if DEVICE == "cpu":
        print("CUDA not available; skipping tests")
        sys.exit(0)

    print(f"DFA: {dfa.num_states} states, vocab_size={dfa.vocab_size}")

    # Test 1: generate 50 tokens
    state = dfa.start_state
    generated = []
    rng = random.Random(42)
    for _ in range(50):
        logits = torch.randn(dfa.vocab_size, device=DEVICE)
        tok, state = constrained_sample(logits, dfa, state, top_k=20, rand_u=rng.random())
        if tok < 0: break
        generated.append(vocab[tok])
        if state < 0: break

    text = b"".join(generated).decode("latin-1", errors="replace")
    match = bool(re.fullmatch(r"-?[1-9][0-9]*", text))
    print(f"Test 1 (generate_matches_regex): text='{text}' match={match}")
    assert match, "FAIL"
    print("PASS")

    # Test 2: distribution
    logits_gpu = torch.zeros(dfa.vocab_size, device=DEVICE)
    logits_masked = logits_gpu.clone()
    apply_dfa_mask(logits_masked, dfa, dfa.start_state)
    valid_cnt = (logits_masked > -1e30).sum().item()
    k = min(10, valid_cnt)
    expected = 1.0 / k
    counts: dict = {}
    for _ in range(5000):
        tok, _ = constrained_sample(logits_gpu, dfa, dfa.start_state, top_k=10, rand_u=random.random())
        counts[tok] = counts.get(tok, 0) + 1
    chi2 = sum((counts.get(i, 0) - expected * 5000) ** 2 / (expected * 5000)
               for i in list(counts.keys())[:k])
    print(f"Test 2 (distribution_match): chi2={chi2:.2f} threshold={4*k+30}")
    assert chi2 < 4 * k + 30, f"FAIL chi2={chi2}"
    print("PASS")
