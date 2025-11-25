import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused activation (HardSwish + ReLU) CUDA kernel
fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_activation_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        if (x <= 0.0f) {
            output[idx] = 0.0f;
        } else if (x <= 3.0f) {
            output[idx] = x * (x + 3.0f) / 6.0f;
        } else {
            output[idx] = x;
        }
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_activation_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size
    );

    return output;
}
"""

fused_activation_cpp_source = "torch::Tensor fused_activation_cuda(torch::Tensor input);"

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
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.conv(x)
        return self.fused_activation.fused_activation_cuda(x)

def get_inputs():
    batch_size = 128
    in_channels = 8
    height, width = 128, 128
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [8, 64, 3]