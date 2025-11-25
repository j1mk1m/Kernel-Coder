import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for subtraction and Mish activation
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_subtract_mish_kernel(
    const float* input,
    float* output,
    float s,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = input[idx] - s;
        float exp_temp = expf(temp);
        float softplus_val = logf(1.0f + exp_temp);
        float tanh_softplus = tanhf(softplus_val);
        output[idx] = temp * tanh_softplus;
    }
}

torch::Tensor fused_subtract_mish_cuda(
    torch::Tensor input,
    float subtract_value_1,
    float subtract_value_2
) {
    float s = subtract_value_1 + subtract_value_2;
    auto output = torch::empty_like(input);
    const int size = input.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_subtract_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        s,
        size
    );

    return output;
}
"""

fused_kernel_cpp_source = (
    "torch::Tensor fused_subtract_mish_cuda(torch::Tensor input, float subtract_value_1, float subtract_value_2);"
)

# Compile the inline CUDA code
fused_subtract_mish = load_inline(
    name="fused_subtract_mish",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_subtract_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.fused_subtract_mish = fused_subtract_mish  # The loaded CUDA kernel

    def forward(self, x):
        x = self.conv(x)
        # Apply fused subtraction and Mish activation
        x = self.fused_subtract_mish.fused_subtract_mish_cuda(
            x, self.subtract_value_1, self.subtract_value_2
        )
        return x

# The get_inputs and get_init_inputs functions remain the same as in the original code
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]