"""
dfa_sample: GPU-resident DFA constrained decoding with fused top-k sampling.

Typical usage:
    from dfa_sample import compile_regex, constrained_sample

    dfa = compile_regex(r'-?[1-9][0-9]*', tokenizer.get_vocab_bytes())
    logits = model(input_ids).logits[0, -1]  # (vocab_size,) float32 on GPU
    token_id, next_state = constrained_sample(logits, dfa, state=0, top_k=50)
"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch

try:
    import _dfa_sample_ext as _ext
    _EXTENSION_AVAILABLE = True
except ImportError:
    _EXTENSION_AVAILABLE = False
    _ext = None


def _require_extension() -> None:
    if not _EXTENSION_AVAILABLE:
        raise RuntimeError(
            "dfa_sample C++/CUDA extension not built. Run: pip install -e ."
        )


class DFA:
    """Compiled DFA handle wrapping the C++ DFAHandle object."""

    def __init__(self, handle: object) -> None:
        self._handle = handle

    @property
    def num_states(self) -> int:
        return self._handle.num_states

    @property
    def vocab_size(self) -> int:
        return self._handle.vocab_size

    @property
    def start_state(self) -> int:
        return self._handle.start_state

    def _cpp_handle(self) -> object:
        return self._handle


def compile_regex(pattern: str, vocab_bytes: List[bytes]) -> DFA:
    """Compile a regex pattern and tokenizer vocabulary into a GPU-resident DFA.

    Args:
        pattern:     Regular expression string (Python re syntax subset).
        vocab_bytes: List of bytes objects, one per token, indexed by token ID.

    Returns:
        DFA object backed by device-memory bitmask tables.
    """
    _require_extension()
    if not vocab_bytes:
        raise ValueError("vocab_bytes must be non-empty")
    handle = _ext.compile_dfa(pattern, vocab_bytes)
    return DFA(handle)


def constrained_sample(
    logits: torch.Tensor,
    dfa: DFA,
    state: int,
    top_k: int = 50,
    temperature: float = 1.0,
    rand_u: float | None = None,
) -> Tuple[int, int]:
    """Fused constrained top-k sampling in a single GPU kernel.

    Args:
        logits:      Float32 CUDA tensor of shape (vocab_size,).
        dfa:         Compiled DFA handle from compile_regex().
        state:       Current DFA state index.
        top_k:       Number of top candidates to sample from.
        temperature: Softmax temperature.
        rand_u:      Uniform random value in [0, 1); sampled if None.

    Returns:
        (token_id, next_state) tuple.
    """
    _require_extension()

    if not logits.is_cuda:
        raise ValueError("logits must be a CUDA tensor")
    if logits.dtype != torch.float32:
        logits = logits.float()
    if not logits.is_contiguous():
        logits = logits.contiguous()

    if rand_u is None:
        rand_u = random.random()

    rand_u = float(rand_u)
    rand_u = max(0.0, min(rand_u, 0.9999999))

    token_id, next_state = _ext.fused_constrained_sample(
        logits, dfa._cpp_handle(), state, top_k, float(temperature), rand_u
    )
    return token_id, next_state


def apply_dfa_mask(logits: torch.Tensor, dfa: DFA, state: int) -> torch.Tensor:
    """Apply DFA validity mask to logits in-place.

    Sets invalid token positions to -inf. Returns logits for chaining.
    """
    _require_extension()

    if not logits.is_cuda:
        raise ValueError("logits must be a CUDA tensor")
    if logits.dtype != torch.float32:
        raise ValueError("logits must be float32")
    if not logits.is_contiguous():
        logits = logits.contiguous()

    _ext.dfa_mask(logits, dfa._cpp_handle(), state)
    return logits


__all__ = ["DFA", "compile_regex", "constrained_sample", "apply_dfa_mask"]
