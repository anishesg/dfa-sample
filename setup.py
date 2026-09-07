import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Detect CUDA architecture from torch or use sm_80+ default
def get_cuda_archs():
    archs = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0")
    return archs

src_dir = os.path.join(os.path.dirname(__file__), "src")
csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")

# Source files: compile src/*.cu + csrc/bindings.cpp
cuda_sources = [
    os.path.join(src_dir,  "dfa_mask.cu"),
    os.path.join(src_dir,  "fused_sample.cu"),
    os.path.join(csrc_dir, "bindings.cpp"),
]

# dfa_compile.cuh is header-only; dfa_mask.cu and fused_sample.cu compile the kernels.
# We also add a stub .cu for the compile functions used from Python.
compile_stub = os.path.join(src_dir, "dfa_compile_stub.cu")
if not os.path.exists(compile_stub):
    with open(compile_stub, "w") as f:
        f.write('#include "dfa_compile.cuh"\n')
        f.write('// Instantiation stub for DFA compilation\n')
cuda_sources.insert(0, compile_stub)

nvcc_flags = [
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-std=c++17",
    "--use_fast_math",
    "--ptxas-options=-v",
]

arch_list = get_cuda_archs()
for arch in arch_list.split(";"):
    a = arch.strip().replace(".", "")
    nvcc_flags += [f"-gencode=arch=compute_{a},code=sm_{a}"]

setup(
    name="dfa_sample",
    version="0.1.0",
    description="Fused grammar-constrained decode: GPU-resident DFA token masking with register-heap top-k sampling",
    packages=find_packages(exclude=["tests", "benchmarks"]),
    ext_modules=[
        CUDAExtension(
            name="_dfa_sample_ext",
            sources=cuda_sources,
            include_dirs=[src_dir, csrc_dir],
            extra_compile_args={
                "cxx":  ["-O3", "-std=c++17", "-fopenmp"],
                "nvcc": nvcc_flags,
            },
            extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0"],
)
