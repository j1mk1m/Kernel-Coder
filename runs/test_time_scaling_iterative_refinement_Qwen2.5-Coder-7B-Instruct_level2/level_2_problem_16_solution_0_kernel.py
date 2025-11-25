import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Mish activation
mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mish_forward_kernel(const float* input, float* output, int elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < elements) {
        float val = input[tid];
        output[tid] = val * tanh(log1p(exp(val)));
    }
}

torch::Tensor mish_forward_cuda(torch::Tensor input) {
    int elements = input.numel();
    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    const int blocks_per_grid = (elements + threads_per_block - 1) / threads_per_block;

    mish_forward_kernel<<<blocks_per_grid, threads_per_block>>>(input.data_ptr<float>(), output.data_ptr<float>(), elements);
    return output;
}
"""

mish_cpp_source = (
    "torch::Tensor mish_forward_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Mish activation
mish = load_inline(
    name="mish",
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_source,
    functions=["mish_forward_cuda"],
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
        self.mish_forward = mish

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.mish_forward.mish_forward_cuda(x)  # Using custom CUDA kernel for Mish
        x = x + self.add_value
        x = torch.nn.functional.hardtanh(x, min_val=-1, max_val=1)
        x = x * self.scale
        return x

batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]