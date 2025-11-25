import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused ReLU + HardSwish CUDA kernel
fused_relu_hardswish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_relu_hardswish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Apply ReLU
        x = fmaxf(x, 0.0f);
        // Compute HardSwish
        if (x <= 3.0f) {
            output[idx] = x * (x + 3.0f) / 6.0f;
        } else {
            output[idx] = x;
        }
    }
}

torch::Tensor fused_relu_hardswish_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (input.numel() + block_size - 1) / block_size;
    fused_relu_hardswish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), input.numel()
    );
    return output;
}
"""

# The C++ header for the kernel
fused_relu_hardswish_cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_relu_hardswish_cuda(torch::Tensor input);
"""

# Compile the CUDA kernel
fused_relu_hardswish = load_inline(
    name="fused_relu_hardswish",
    cpp_sources=[fused_relu_hardswish_cpp_source],
    cuda_sources=[fused_relu_hardswish_source],
    functions=["fused_relu_hardswish_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # The fused activation is a function from the loaded extension
        self.fused_relu_hardswish = fused_relu_hardswish.fused_relu_hardswish_cuda

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_relu_hardswish(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]