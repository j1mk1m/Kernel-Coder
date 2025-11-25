import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for minimum operation
min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void min_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx];
    }
}

torch::Tensor min_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    min_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

min_cpp_source = (
    "torch::Tensor min_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for minimum operation
min_op = load_inline(
    name="min_op",
    cpp_sources=min_cpp_source,
    cuda_sources=min_source,
    functions=["min_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for tanh operation
tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

torch::Tensor tanh_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    tanh_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

tanh_cpp_source = (
    "torch::Tensor tanh_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for tanh operation
tanh_op = load_inline(
    name="tanh_op",
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_source,
    functions=["tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.min_op = min_op
        self.tanh_op = tanh_op

    def forward(self, x):
        x = self.conv(x)
        x = self.min_op.min_cuda(x) # Apply minimum operation along the channel dimension
        x = self.tanh_op.tanh_cuda(x)
        x = self.tanh_op.tanh_cuda(x)
        return x

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]