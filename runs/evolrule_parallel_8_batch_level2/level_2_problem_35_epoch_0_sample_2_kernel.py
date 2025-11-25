import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused subtraction + hardswish CUDA kernel
subtract_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void subtract_hardswish_kernel(
    const float* input, float subtract_val, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] - subtract_val;
        float tmp = x + 3.0f;
        tmp = fmaxf(tmp, 0.0f);
        tmp = fminf(tmp, 6.0f);
        output[idx] = x * tmp / 6.0f;
    }
}

torch::Tensor subtract_hardswish_cuda(
    torch::Tensor input, float subtract_val) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    subtract_hardswish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), subtract_val, output.data_ptr<float>(), size);

    return output;
}
"""

subtract_hardswish_cpp = "torch::Tensor subtract_hardswish_cuda(torch::Tensor input, float subtract_val);"

# Define the custom mish activation CUDA kernel
mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void mish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float exp_x = expf(x);
        float softplus = logf(exp_x + 1.0f);
        output[idx] = x * tanhf(softplus);
    }
}

torch::Tensor mish_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mish_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

mish_cpp = "torch::Tensor mish_cuda(torch::Tensor input);"

# Compile the CUDA extensions
subtract_hardswish = load_inline(
    name="subtract_hardswish",
    cpp_sources=subtract_hardswish_cpp,
    cuda_sources=subtract_hardswish_source,
    functions=["subtract_hardswish_cuda"],
    verbose=True
)

mish_activation = load_inline(
    name="mish_activation",
    cpp_sources=mish_cpp,
    cuda_sources=mish_source,
    functions=["mish_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        self.subtract_hardswish = subtract_hardswish
        self.mish = mish_activation

    def forward(self, x):
        x = self.conv(x)
        # Apply fused subtraction and hardswish
        x = self.subtract_hardswish.subtract_hardswish_cuda(x, self.subtract_value)
        x = self.pool(x)
        # Apply custom mish activation
        x = self.mish.mish_cuda(x)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]

# Global variables from original code
batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2