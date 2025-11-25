import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define fused Mish-Tanh CUDA kernel
fused_mish_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_mish_tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sp = logf(1.0f + expf(x)); // Softplus computation
        float mish = x * tanhf(sp);      // Mish activation
        output[idx] = tanhf(mish);       // Final tanh activation
    }
}

torch::Tensor fused_mish_tanh_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_mish_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size
    );

    return output;
}
"""

# Header for fused kernel
fused_mish_tanh_cpp_source = (
    "torch::Tensor fused_mish_tanh_cuda(torch::Tensor input);"
)

# Compile fused Mish-Tanh kernel
fused_mish_tanh = load_inline(
    name="fused_mish_tanh",
    cpp_sources=fused_mish_tanh_cpp_source,
    cuda_sources=fused_mish_tanh_source,
    functions=["fused_mish_tanh_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.fused_activation = fused_mish_tanh

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_activation.fused_mish_tanh_cuda(x)
        return x

# Required for compatibility with original setup
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 32, 64, 64
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]