import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softsign activation
softsign_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void softsign_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t xi = x[idx];
        y[idx] = xi / (1 + fabs(xi));
    }
}

std::tuple<torch::Tensor> softsign_forward(torch::Tensor x) {
    const auto size = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    if (x.scalar_type() == torch::kFloat32) {
        softsign_kernel<float><<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);
    } else if (x.scalar_type() == torch::kFloat16) {
        softsign_kernel<__half><<<num_blocks, block_size>>>(x.data_ptr<__half>(), y.data_ptr<__half>(), size);
    } else {
        throw std::runtime_error("Unsupported tensor type");
    }

    return std::make_tuple(y);
}
"""

softsign_cpp_source = (
    "std::tuple<torch::Tensor> softsign_forward(torch::Tensor x);"
)

# Compile the inline CUDA code for Softsign
softsign = load_inline(
    name="softsign",
    cpp_sources=softsign_cpp_source,
    cuda_sources=softsign_source,
    functions=["softsign_forward"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["--expt-relaxed-constexpr"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softsign_forward = softsign

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softsign_forward.softsign_forward(x)[0]