import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused Mish + Tanh activation CUDA kernel
fused_activation_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void fused_activation_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float softplus_x;
        if (x > 18.0f) {
            softplus_x = x;
        } else if (x < -30.0f) {
            softplus_x = expf(x);
        } else {
            softplus_x = logf(1.0f + expf(x));
        }
        float mish_x = x * tanhf(softplus_x);
        output[idx] = tanhf(mish_x);
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (input.numel() + block_size - 1) / block_size;
    fused_activation_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel()
    );
    return output;
}
"""

# Header declarations for the fused activation kernel
fused_activation_cpp_source = (
    "torch::Tensor fused_activation_cuda(torch::Tensor input);"
)

# Compile the fused activation kernel
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_activation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.fused_activation = fused_activation  # Custom fused activation module

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_activation.fused_activation_cuda(x)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]