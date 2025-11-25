import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise subtraction
elementwise_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_subtract_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

torch::Tensor elementwise_subtract_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_subtract_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_subtract_cpp_source = (
    "torch::Tensor elementwise_subtract_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise subtraction
elementwise_subtract = load_inline(
    name="elementwise_subtract",
    cpp_sources=elementwise_subtract_cpp_source,
    cuda_sources=elementwise_subtract_source,
    functions=["elementwise_subtract_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Mish activation
mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ float mish(float x) {
    return x * tanh(log(expf(x) + 1));
}

__global__ void mish_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = mish(x[idx]);
    }
}

torch::Tensor mish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

mish_cpp_source = (
    "torch::Tensor mish_cuda(torch::Tensor x);"
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


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.elementwise_subtract = elementwise_subtract
        self.mish = mish

    def forward(self, x):
        x = self.conv(x)
        x = self.elementwise_subtract.elementwise_subtract_cuda(x, self.subtract_value_1)
        x = self.elementwise_subtract.elementwise_subtract_cuda(x, self.subtract_value_2)
        x = self.mish.mish_cuda(x)
        return x


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]