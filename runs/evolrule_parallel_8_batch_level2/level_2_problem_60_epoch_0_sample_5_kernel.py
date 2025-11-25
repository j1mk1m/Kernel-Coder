import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Swish activation
swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        y[idx] = xi / (1.0f + expf(-xi));
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

swish_cpp_header = "torch::Tensor swish_cuda(torch::Tensor x);"
swish_module = load_inline(
    name="swish_cuda",
    cpp_sources=swish_cpp_header,
    cuda_sources=swish_source,
    functions=["swish_cuda"],
    verbose=True,
)

# Define the custom CUDA kernel for HardSwish activation
hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void hardswish_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float temp = xi + 3.0f;
        temp = fmaxf(temp, 0.0f);
        temp = fminf(temp, 6.0f);
        y[idx] = xi * temp / 6.0f;
    }
}

torch::Tensor hardswish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardswish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

hardswish_cpp_header = "torch::Tensor hardswish_cuda(torch::Tensor x);"
hardswish_module = load_inline(
    name="hardswish_cuda",
    cpp_sources=hardswish_cpp_header,
    cuda_sources=hardswish_source,
    functions=["hardswish_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.swish = swish_module
        self.hardswish = hardswish_module

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.swish.swish_cuda(x)
        x = self.group_norm(x)
        x = self.hardswish.hardswish_cuda(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
groups = 4
eps = 1e-5

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, eps]