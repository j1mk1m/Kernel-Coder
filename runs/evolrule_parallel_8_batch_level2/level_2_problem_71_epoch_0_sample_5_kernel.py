import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused division and LeakyReLU CUDA kernel (in-place)
fused_div_leakyrelu_inplace_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_div_leakyrelu_inplace_kernel(
    float* data,
    float divisor,
    float negative_slope,
    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx] / divisor;
        data[idx] = (val > 0) ? val : val * negative_slope;
    }
}

torch::Tensor fused_div_leakyrelu_inplace_cuda(torch::Tensor input, float divisor, float negative_slope) {
    auto size = input.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_div_leakyrelu_inplace_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        divisor,
        negative_slope,
        size
    );

    return input;
}
"""

fused_div_leakyrelu_inplace_cpp_source = """
torch::Tensor fused_div_leakyrelu_inplace_cuda(torch::Tensor input, float divisor, float negative_slope);
"""

# Compile the fused in-place kernel
fused_div_leakyrelu_inplace = load_inline(
    name="fused_div_leakyrelu_inplace",
    cpp_sources=fused_div_leakyrelu_inplace_cpp_source,
    cuda_sources=fused_div_leakyrelu_inplace_source,
    functions=["fused_div_leakyrelu_inplace_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.fused_div_leakyrelu_inplace = fused_div_leakyrelu_inplace

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_div_leakyrelu_inplace.fused_div_leakyrelu_inplace_cuda(x, self.divisor, 0.01)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
divisor = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]