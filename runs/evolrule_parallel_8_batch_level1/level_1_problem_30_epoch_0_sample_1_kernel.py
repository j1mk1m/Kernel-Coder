import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softsign_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softsign_kernel(const float* x, float* y, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float xi = x[idx];
        float denom = 1.0f + fabsf(xi);
        y[idx] = xi / denom;
    }
}

torch::Tensor softsign_cuda(torch::Tensor x) {
    auto output = torch::empty_like(x);
    int num_elements = x.numel();

    const int threads_per_block = 256;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    softsign_kernel<<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), output.data_ptr<float>(), num_elements);

    return output;
}
"""

softsign_cpp = """
extern "C" {
    torch::Tensor softsign_cuda(torch::Tensor x);
}
"""

# Compile the CUDA code
softsign = load_inline(
    name="softsign",
    cpp_sources=softsign_cpp,
    cuda_sources=softsign_source,
    functions=["softsign_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softsign_cuda = softsign

    def forward(self, x):
        return self.softsign_cuda.softsign_cuda(x)