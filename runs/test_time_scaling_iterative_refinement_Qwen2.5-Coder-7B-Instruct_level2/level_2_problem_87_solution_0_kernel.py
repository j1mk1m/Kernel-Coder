import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Mish activation
mish_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mish_activation_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = x[idx] * tanh(log1p(exp(x[idx])));
    }
}

torch::Tensor mish_activation_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::clone(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mish_activation_kernel<<<num_blocks, block_size>>>(out.data_ptr<float>(), size);

    return out;
}
"""

mish_activation_cpp_source = (
    "torch::Tensor mish_activation_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for Mish activation
mish_activation = load_inline(
    name="mish_activation",
    cpp_sources=mish_activation_cpp_source,
    cuda_sources=mish_activation_source,
    functions=["mish_activation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for subtraction
subtraction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void subtraction_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

torch::Tensor subtraction_cuda(torch::Tensor a, const float* b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    subtraction_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b, out.data_ptr<float>(), size);

    return out;
}
"""

subtraction_cpp_source = (
    "torch::Tensor subtraction_cuda(torch::Tensor a, const float* b);"
)

# Compile the inline CUDA code for subtraction
subtraction = load_inline(
    name="subtraction",
    cpp_sources=subtraction_cpp_source,
    cuda_sources=subtraction_source,
    functions=["subtraction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.mish_activation = mish_activation
        self.subtraction = subtraction

    def forward(self, x):
        x = self.conv(x)
        x = self.subtraction.subtraction_cuda(x, &self.subtract_value_1)
        x = self.subtraction.subtraction_cuda(x, &self.subtract_value_2)
        x = self.mish_activation.mish_activation_cuda(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]