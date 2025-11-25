import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused subtract and hardswish CUDA kernel
fused_subtract_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_subtract_hardswish_kernel(
    const float* input, const float subtract_val, float* output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float x = input[idx] - subtract_val;
        float tmp = x + 3.0f;
        tmp = fmaxf(tmp, 0.0f); // ReLU6
        tmp = fminf(tmp, 6.0f); // Clamp to 6
        output[idx] = x * tmp / 6.0f;
    }
}

torch::Tensor fused_subtract_hardswish_cuda(torch::Tensor input, float subtract_val) {
    int num_elements = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_subtract_hardswish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        subtract_val,
        output.data_ptr<float>(),
        num_elements
    );

    return output;
}
"""

fused_subtract_hardswish_cpp_source = (
    "torch::Tensor fused_subtract_hardswish_cuda(torch::Tensor input, float subtract_val);"
)

# Compile the fused kernel
fused_subtract_hardswish = load_inline(
    name="fused_subtract_hardswish",
    cpp_sources=fused_subtract_hardswish_cpp_source,
    cuda_sources=fused_subtract_hardswish_source,
    functions=["fused_subtract_hardswish_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        self.fused_subtract_hardswish = fused_subtract_hardswish

    def forward(self, x):
        x = self.conv(x)
        # Apply fused subtract and hardswish
        x = self.fused_subtract_hardswish.fused_subtract_hardswish_cuda(x, self.subtract_value)
        x = self.pool(x)
        x = torch.nn.functional.mish(x)
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]