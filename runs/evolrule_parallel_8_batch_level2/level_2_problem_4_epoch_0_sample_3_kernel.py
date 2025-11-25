import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for applying Mish twice
double_mish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <math.h>

__global__ void double_mish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // First Mish
        float exp_x = expf(x);
        float softplus_x = logf(1.0f + exp_x);
        float tanh_softplus_x = tanhf(softplus_x);
        float y = x * tanh_softplus_x;
        // Second Mish on y
        float exp_y = expf(y);
        float softplus_y = logf(1.0f + exp_y);
        float tanh_softplus_y = tanhf(softplus_y);
        output[idx] = y * tanh_softplus_y;
    }
}

torch::Tensor double_mish_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int threads_per_block = 256;
    const int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    double_mish_kernel<<<blocks_per_grid, threads_per_block>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

cpp_source = """
extern "C" {
    torch::Tensor double_mish_cuda(torch::Tensor input);
}
"""

double_mish = load_inline(
    name="double_mish",
    cpp_sources=cpp_source,
    cuda_sources=double_mish_source,
    functions=["double_mish_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.double_mish = double_mish  # The loaded CUDA kernel

    def forward(self, x):
        x = self.conv(x)
        x = self.double_mish.double_mish_cuda(x)
        return x

# Helper functions
batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]