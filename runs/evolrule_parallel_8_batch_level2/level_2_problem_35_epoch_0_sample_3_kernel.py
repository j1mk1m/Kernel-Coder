import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused subtract + HardSwish CUDA kernel
subtract_and_hardswish_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void subtract_and_hardswish_kernel(
    const float* input, float subtract_val, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = input[idx] - subtract_val;
        float z = temp + 3.0f;
        float relu6_z = fmaxf(0.0f, fminf(z, 6.0f));
        output[idx] = temp * relu6_z / 6.0f;
    }
}

torch::Tensor subtract_and_hardswish_cuda(
    torch::Tensor input, float subtract_val) {
    input = input.contiguous();
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    subtract_and_hardswish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), subtract_val, output.data_ptr<float>(), size);
    return output;
}
"""

subtract_and_hardswish_cpp = "torch::Tensor subtract_and_hardswish_cuda(torch::Tensor input, float subtract_val);"

# Define the Mish CUDA kernel
mish_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void mish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x_val = input[idx];
        if (x_val > 20.f) {
            output[idx] = x_val;
        } else if (x_val < -20.f) {
            output[idx] = 0.f;
        } else {
            float exp_x = expf(x_val);
            float softplus_x = logf(1.f + exp_x);
            float tanh_softplus = tanhf(softplus_x);
            output[idx] = x_val * tanh_softplus;
        }
    }
}

torch::Tensor mish_cuda(torch::Tensor input) {
    input = input.contiguous();
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    mish_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

mish_cpp = "torch::Tensor mish_cuda(torch::Tensor input);"

# Compile the CUDA kernels
subtract_and_hardswish = load_inline(
    name="subtract_and_hardswish",
    cuda_sources=subtract_and_hardswish_source,
    cpp_sources=subtract_and_hardswish_cpp,
    functions=["subtract_and_hardswish_cuda"],
    verbose=True,
)

mish = load_inline(
    name="mish",
    cuda_sources=mish_source,
    cpp_sources=mish_cpp,
    functions=["mish_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        self.subtract_and_hardswish = subtract_and_hardswish
        self.mish = mish

    def forward(self, x):
        x = self.conv(x)
        x = self.subtract_and_hardswish.subtract_and_hardswish_cuda(x, self.subtract_value)
        x = self.pool(x)
        x = self.mish.mish_cuda(x)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]

# Global variables from the original setup
batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2