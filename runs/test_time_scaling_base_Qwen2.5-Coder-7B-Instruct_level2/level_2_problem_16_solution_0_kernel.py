import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Mish activation
mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mish_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = x[idx] * tanh(log1p(exp(x[idx])));
    }
}

void mish_cuda(torch::Tensor x) {
    auto size = x.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), size);
}
"""

mish_cpp_source = (
    "void mish_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for Mish activation
mish = load_inline(
    name="mish",
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_source,
    functions=["mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Hardtanh activation
hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel(float* x, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = max(min(x[idx], max_val), min_val);
    }
}

void hardtanh_cuda(torch::Tensor x, float min_val, float max_val) {
    auto size = x.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardtanh_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), size, min_val, max_val);
}
"""

hardtanh_cpp_source = (
    "void hardtanh_cuda(torch::Tensor x, float min_val, float max_val);"
)

# Compile the inline CUDA code for Hardtanh activation
hardtanh = load_inline(
    name="hardtanh",
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_source,
    functions=["hardtanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        x = self.conv_transpose(x)
        mish_cuda(x) # Mish activation implemented in CUDA
        x = x + self.add_value
        hardtanh_cuda(x, -1, 1) # Hardtanh activation implemented in CUDA
        x = x * self.scale # Scaling
        return x