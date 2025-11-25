import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for division
division_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void division_kernel(float* input, float divisor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] /= divisor;
    }
}

torch::Tensor division_cuda(torch::Tensor input, float divisor) {
    auto size = input.numel();
    auto out = input.clone();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    division_kernel<<<num_blocks, block_size>>>(out.data_ptr<float>(), divisor, size);

    return out;
}
"""

division_cpp_source = (
    "torch::Tensor division_cuda(torch::Tensor input, float divisor);"
)

# Compile the inline CUDA code for division
division = load_inline(
    name="division",
    cpp_sources=division_cpp_source,
    cuda_sources=division_source,
    functions=["division_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for LeakyReLU
leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = input[idx] > 0 ? input[idx] : input[idx] * 0.01f;
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto out = input.clone();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    leaky_relu_kernel<<<num_blocks, block_size>>>(out.data_ptr<float>(), size);

    return out;
}
"""

leaky_relu_cpp_source = (
    "torch::Tensor leaky_relu_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for LeakyReLU
leaky_relu = load_inline(
    name="leaky_relu",
    cpp_sources=leaky_relu_cpp_source,
    cuda_sources=leaky_relu_source,
    functions=["leaky_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor

    def forward(self, x):
        x = self.conv(x)
        x = division.division_cuda(x, self.divisor)
        x = leaky_relu.leaky_relu_cuda(x)
        return x