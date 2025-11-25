import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define fused Subtract + Hardswish kernel
fused_subtract_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_subtract_hardswish_kernel(
    const float* input,
    const float subtract_value,
    float* output,
    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = input[idx] - subtract_value;
        float temp_plus_3 = temp + 3.0f;
        float relu6_val = fmaxf(fminf(temp_plus_3, 6.0f), 0.0f);
        output[idx] = temp * relu6_val / 6.0f;
    }
}

torch::Tensor fused_subtract_hardswish_cuda(
    torch::Tensor input,
    float subtract_value) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_subtract_hardswish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        subtract_value,
        output.data_ptr<float>(),
        size
    );

    return output;
}
"""

# Define Mish activation kernel
mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void mish_kernel(
    const float* input,
    float* output,
    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float exp_x = expf(x);
        float softplus = logf(1.0f + exp_x);
        float tanh_soft = tanhf(softplus);
        output[idx] = x * tanh_soft;
    }
}

torch::Tensor mish_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}
"""

# Compile fused Subtract + Hardswish kernel
fused_subtract_hardswish_cpp_src = (
    "torch::Tensor fused_subtract_hardswish_cuda(torch::Tensor input, float subtract_value);"
)
fused_subtract_hardswish = load_inline(
    name="fused_subtract_hardswish",
    cpp_sources=fused_subtract_hardswish_cpp_src,
    cuda_sources=fused_subtract_hardswish_source,
    functions=["fused_subtract_hardswish_cuda"],
    verbose=True,
)

# Compile Mish kernel
mish_cpp_src = "torch::Tensor mish_cuda(torch::Tensor input);"
mish = load_inline(
    name="mish",
    cpp_sources=mish_cpp_src,
    cuda_sources=mish_source,
    functions=["mish_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(pool_kernel_size)
        self.subtract_value = subtract_value
        self.fused_subtract_hardswish = fused_subtract_hardswish
        self.mish = mish

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_subtract_hardswish.fused_subtract_hardswish_cuda(x, self.subtract_value)
        x = self.pool(x)
        x = self.mish.mish_cuda(x)
        return x