#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dfa_types.cuh"
#include "dfa_compile.cuh"
#include "dfa_mask.cuh"
#include "fused_sample.cuh"

#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

namespace py = pybind11;

// ---- Python-facing DFA handle ----

struct DFAHandle {
    std::unique_ptr<DFAHost, decltype(&free_dfa)> ptr;

    explicit DFAHandle(DFAHost* h) : ptr(h, free_dfa) {}

    DFAHost* get() const { return ptr.get(); }

    int num_states()  const { return ptr->config.num_states; }
    int vocab_size()  const { return ptr->config.vocab_size; }
    int start_state() const { return ptr->config.start_state; }
};

// ---- compile_dfa binding ----

static std::shared_ptr<DFAHandle> py_compile_dfa(
    const std::string& pattern,
    const std::vector<py::bytes>& vocab_bytes)
{
    std::vector<std::vector<uint8_t>> vocab;
    vocab.reserve(vocab_bytes.size());
    for (const auto& b : vocab_bytes) {
        std::string s = b;
        vocab.emplace_back(s.begin(), s.end());
    }
    DFAHost* h = compile_dfa(pattern, vocab);
    return std::make_shared<DFAHandle>(h);
}

// ---- dfa_mask binding ----

static void py_dfa_mask(torch::Tensor logits,
                         const std::shared_ptr<DFAHandle>& dfa,
                         int state)
{
    TORCH_CHECK(logits.is_cuda(),     "logits must be a CUDA tensor");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(logits.numel() == dfa->vocab_size(),
                "logits size must match DFA vocab_size");
    TORCH_CHECK(state >= 0 && state < dfa->num_states(), "invalid state");

    launch_dfa_mask(logits.data_ptr<float>(),
                    dfa->get()->device,
                    state,
                    c10::cuda::getCurrentCUDAStream());
}

// ---- fused_constrained_sample binding ----

static std::tuple<int, int> py_fused_constrained_sample(
    torch::Tensor logits,
    const std::shared_ptr<DFAHandle>& dfa,
    int state,
    int top_k,
    float temperature,
    float rand_u)
{
    TORCH_CHECK(logits.is_cuda(),       "logits must be a CUDA tensor");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(logits.numel() == dfa->vocab_size(),
                "logits size must match DFA vocab_size");
    TORCH_CHECK(state >= 0 && state < dfa->num_states(), "invalid state");
    TORCH_CHECK(top_k >= 1 && top_k <= 256, "top_k must be in [1, 256]");
    TORCH_CHECK(temperature > 0.0f, "temperature must be positive");
    TORCH_CHECK(rand_u >= 0.0f && rand_u < 1.0f, "rand_u must be in [0, 1)");

    // Allocate result on device
    SampleResult* d_result;
    cudaMalloc(&d_result, sizeof(SampleResult));

    launch_fused_sample(
        logits.data_ptr<float>(),
        dfa->get()->device,
        state, top_k, temperature, rand_u,
        d_result,
        c10::cuda::getCurrentCUDAStream());

    SampleResult h_result;
    cudaMemcpy(&h_result, d_result, sizeof(SampleResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return {h_result.token_id, h_result.next_state};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GPU-resident DFA constrained decoding with fused top-k sampling";

    py::class_<DFAHandle, std::shared_ptr<DFAHandle>>(m, "DFAHandle")
        .def_property_readonly("num_states",  &DFAHandle::num_states)
        .def_property_readonly("vocab_size",  &DFAHandle::vocab_size)
        .def_property_readonly("start_state", &DFAHandle::start_state);

    m.def("compile_dfa",
          &py_compile_dfa,
          py::arg("pattern"),
          py::arg("vocab_bytes"),
          "Compile a regex pattern and tokenizer vocabulary into a GPU-resident DFA.\n"
          "vocab_bytes: list of bytes objects, one per token (by token ID).\n"
          "Returns a DFAHandle opaque object.");

    m.def("dfa_mask",
          &py_dfa_mask,
          py::arg("logits"),
          py::arg("dfa"),
          py::arg("state"),
          "Apply DFA validity mask to a float32 logit tensor in-place (CUDA).\n"
          "Invalid token positions are set to -inf.");

    m.def("fused_constrained_sample",
          &py_fused_constrained_sample,
          py::arg("logits"),
          py::arg("dfa"),
          py::arg("state"),
          py::arg("top_k"),
          py::arg("temperature"),
          py::arg("rand_u"),
          "Fused constrained top-k sampling in a single kernel launch.\n"
          "Returns (token_id, next_state).");
}
