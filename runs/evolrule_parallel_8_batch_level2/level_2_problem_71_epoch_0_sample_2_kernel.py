import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for division and LeakyReLU
fused_div_leakyrelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_div_leakyrelu_kernel(const float* input, float divisor, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] / divisor;
        output[idx] = (val > 0.0f) ? val : 0.01f * val;
    }
}

torch::Tensor fused_div_leakyrelu_cuda(torch::Tensor input, float divisor) {
    int64_t size = input.numel();
    torch::Tensor output = torch::empty_like(input);
    
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    fused_div_leakyrelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), divisor, output.data_ptr<float>(), size);

    return output;
}
"""

# Declaration for the C++ function
fused_div_leakyrelu_cpp_source = (
    "torch::Tensor fused_div_leakyrelu_cuda(torch::Tensor input, float divisor);"
)

# Compile the CUDA kernel
fused_div_leakyrelu = load_inline(
    name="fused_div_leakyrelu",
    cpp_sources=fused_div_leakyrelu_cpp_source,
    cuda_sources=fused_div_leakyrelu_source,
    functions=["fused_div_leakyrelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.fused_div_leakyrelu = fused_div_leakyrelu  # The loaded module

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_div_leakyrelu.fused_div_leakyrelu_cuda(x, self.divisor)
        return x

# Keep the same input functions
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