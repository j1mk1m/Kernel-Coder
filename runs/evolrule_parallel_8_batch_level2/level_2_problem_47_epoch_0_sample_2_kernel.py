import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused Mish-Tanh CUDA kernel
fused_mish_tanh_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void fused_mish_tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float softplus_x = log1pf(expf(x));
        float mish = x * tanhf(softplus_x);
        output[idx] = tanhf(mish);
    }
}

torch::Tensor fused_mish_tanh_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    const int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    fused_mish_tanh_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}
"""

cpp_source = "torch::Tensor fused_mish_tanh_cuda(torch::Tensor input);"

# Compile the fused kernel
fused_mish_tanh = load_inline(
    name="fused_mish_tanh",
    cuda_sources=fused_mish_tanh_source,
    cpp_sources=cpp_source,
    functions=["fused_mish_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.fused_mish_tanh = fused_mish_tanh  # Load the fused kernel module

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_mish_tanh.fused_mish_tanh_cuda(x)
        return x

# Global variables required for initialization and input generation
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 32, 64, 64
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]