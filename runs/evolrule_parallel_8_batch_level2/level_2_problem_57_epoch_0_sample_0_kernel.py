import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused ReLU + HardSwish CUDA kernel
fused_act_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_hardsigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float y = fmaxf(x, 0.0f); // ReLU
        float t = (y + 3.0f) / 6.0f;
        t = fminf(t, 1.0f); // Clamp at 1.0
        output[idx] = y * t;
    }
}

torch::Tensor fused_relu_hardsigmoid_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_relu_hardsigmoid_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

fused_act_cpp_header = """
torch::Tensor fused_relu_hardsigmoid_cuda(torch::Tensor input);
"""

# Compile the CUDA extension
fused_relu_hardsigmoid = load_inline(
    name="fused_relu_hardsigmoid",
    cpp_sources=fused_act_cpp_header,
    cuda_sources=fused_act_source,
    functions=["fused_relu_hardsigmoid_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_activation = fused_relu_hardsigmoid.fused_relu_hardsigmoid_cuda

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_activation(x)
        return x

# Helper functions for input generation
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]