import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for LeakyReLU
leakyrelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leakyrelu_kernel(const float* x, float* out, int size, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] > 0 ? x[idx] : x[idx] * negative_slope;
    }
}

torch::Tensor leakyrelu_cuda(torch::Tensor x, float negative_slope) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    leakyrelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size, negative_slope);

    return out;
}
"""

leakyrelu_cpp_source = (
    "torch::Tensor leakyrelu_cuda(torch::Tensor x, float negative_slope);"
)

# Compile the inline CUDA code for LeakyReLU
leakyrelu = load_inline(
    name="leakyrelu",
    cpp_sources=leakyrelu_cpp_source,
    cuda_sources=leakyrelu_source,
    functions=["leakyrelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
        self.leakyrelu = leakyrelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leakyrelu.leakyrelu_cuda(x, self.negative_slope)