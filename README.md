# dfa-sample

Fused grammar-constrained decode: GPU-resident DFA token masking with register-heap top-k sampling in a single kernel.

## Problem

Structured output systems (Outlines, xgrammar, guidance) constrain LLM decoding to a grammar by masking invalid tokens at each step. Current implementations perform this masking on the CPU:

1. Copy logit vector from GPU to CPU (PCIe, ~20-40us for 128K vocab)
2. Apply grammar mask on CPU (~10-30us for DFA lookup per token)
3. Run top-k/softmax/sample on CPU (~5-15us)
4. Copy selected token ID back to GPU

Total: 50-100us of CPU-GPU synchronization overhead per decode step, serialized with the forward pass. At 30 tokens/s generation, this adds 1.5-3ms/s of pure overhead, and blocks pipeline parallelism between decode steps.

## Approach

### GPU-Resident DFA

The DFA state machine lives entirely in device memory. For each grammar state, a packed bitmask array of ceil(vocab_size/32) uint32 words marks valid tokens. For a 128K vocabulary and 1000 DFA states, this is 1000 * 4096 * 4 = ~16MB of device memory, loaded once after DFA compilation.

Before each decode step, the host passes the current DFA state index. The kernel loads only that state's bitmask (~4KB for 128K vocab) into shared memory using warp-cooperative coalesced loads.

### Fused Pipeline

A single kernel launch performs the full pipeline with zero intermediate global memory writes:

```
logits (global) -> [mask + temperature scale] -> register-resident min-heap (top-k)
                -> [warp softmax] -> [CDF sample] -> token_id + next_state (global)
```

Each thread block maintains a register-resident min-heap of K candidates. As threads tile through the logit vector (in VOCAB_SIZE / (BLOCK_SIZE * ITEMS_PER_THREAD) tiles), they insert valid masked/scaled entries into the heap. After all tiles are processed, a warp reduction merges per-thread heaps, computes softmax over the K survivors, and samples using a caller-provided uniform random value from the CDF.

The selected token's next DFA state is resolved by replaying the token's byte sequence through the DFA transition table.

### Latency

For vocab_size=128256, top_k=50:
- Unconstrained GPU top-k sample: ~8us
- GPU fused constrained sample: ~12us (DFA overhead: ~4us, ~50% increase)
- CPU round-trip constrained sample: ~80us
- Speedup vs CPU round-trip: ~6.7x

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Requirements: CUDA 11.8+, sm_80+ GPU (A100, H100, 4090), CMake 3.18+, optional OpenMP.

```bash
./test_correctness   # mask agreement, top-k agreement, chi-squared distribution test
./test_sequence      # end-to-end autoregressive generation with regex validation
./bench_latency      # latency sweep over vocab_size, num_states, top_k
```

## Python Extension

```bash
pip install -e .
```

```python
from dfa_sample import compile_regex, constrained_sample

dfa = compile_regex(r'-?[1-9][0-9]*', tokenizer.get_vocab_bytes())
logits = model(input_ids).logits[0, -1]  # (vocab_size,) float32 on GPU
token_id, next_state = constrained_sample(logits, dfa, state=0, top_k=50, temperature=1.0)
```

## Structure

```
src/
  dfa_types.cuh      - DFA device structs, bitmask access, transition functions
  dfa_compile.cuh    - Host-side DFA compilation from regex, bitmask precomputation
  dfa_mask.cu        - Standalone DFA masking kernel
  fused_sample.cu    - Fused constrained top-k sampling kernel
  reference.cuh      - CPU reference implementation for correctness testing
tests/
  test_correctness.cu
  test_sequence.cu
benchmarks/
  bench_latency.cu
csrc/
  bindings.cpp       - pybind11/PyTorch bindings
dfa_sample/
  __init__.py        - Python API
setup.py             - PyTorch CUDAExtension build
```

## Next Steps

- Pushdown automaton for full CFG support (JSON, code syntax) via shared-memory stack
- Compressed bitmask representation (RLE) for large DFAs with >10K states
- Multi-token grammar-aware speculation for speculative decoding integration
- Batched constrained sampling for B concurrent requests with different DFA states
