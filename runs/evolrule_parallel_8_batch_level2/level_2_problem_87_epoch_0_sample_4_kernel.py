import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Global variables as per original code
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

# Define the fused subtraction and Mish activation CUDA kernel
fused_sub_mish_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void fused_sub_mish_kernel(const float* input, float* output, int size, float subtract_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] - subtract_val;
        // Compute mish(x) = x * tanh(softplus(x))
        float exp_x = expf(x);
        float softplus = logf(1.0f + exp_x);
        float tanh_softplus = tanhf(softplus);
        output[idx] = x * tanh_softplus;
    }
}

torch::Tensor fused_sub_mish_cuda(torch::Tensor input, float subtract_val) {
    auto output = torch::empty_like(input);
    int size = input.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_sub_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        subtract_val
    );

    return output;
}
"""

fused_sub_mish_cpp_source = (
    "torch::Tensor fused_sub_mish_cuda(torch::Tensor input, float subtract_val);"
)

# Compile the inline CUDA code
fused_sub_mish = load_inline(
    name="fused_sub_mish",
    cpp_sources=[fused_sub_mish_cpp_source],
    cuda_sources=[fused_sub_mish_source],
    functions=["fused_sub_mish_cuda"],
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
        self.fused_sub_mish = fused_sub_mish

    def forward(self, x):
        x = self.conv(x)
        subtract_val = self.subtract_value_1 + self.subtract_value_2
        x = self.fused_sub_mish.fused_sub_mish_cuda(x, subtract_val)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]