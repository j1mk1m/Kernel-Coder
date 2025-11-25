import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for SELU
selu_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void selu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        const float alpha = 1.6732632423543772848170429916717f;
        const float lambda = 1.0507009873554804934193349852946f;
        if (xi > 0.0f) {
            out[idx] = lambda * xi;
        } else {
            out[idx] = lambda * alpha * (expf(xi) - 1.0f);
        }
    }
}

torch::Tensor selu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int threads_per_block = 256;
    const int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    selu_kernel<<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

selu_cpp_source = "torch::Tensor selu_cuda(torch::Tensor x);"

# Compile the inline CUDA code for SELU
selu = load_inline(
    name="selu",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_source,
    functions=["selu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.selu = selu  # Store the CUDA module

    def forward(self, x):
        return self.selu.selu_cuda(x)