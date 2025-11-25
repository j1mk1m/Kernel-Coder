import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused ReLU + bias addition CUDA kernel
fused_relu_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_add_bias_kernel(
    const float* input, const float* bias, float* output,
    int batch_size, int channels, int height, int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int n = idx / (channels * width * height);

    float val = input[idx];
    val = fmaxf(val, 0.f);  // ReLU
    val += bias[c];         // Add per-channel bias
    output[idx] = val;
}

torch::Tensor fused_relu_add_cuda(
    torch::Tensor input, torch::Tensor bias) {
    auto input_size = input.sizes();
    int batch_size = input_size[0];
    int channels = input_size[1];
    int height = input_size[2];
    int width = input_size[3];

    auto output = torch::empty_like(input);
    int total_elements = batch_size * channels * height * width;
    const int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_relu_add_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width
    );

    return output;
}
"""

fused_relu_add_cpp = """
torch::Tensor fused_relu_add_cuda(
    torch::Tensor input, torch::Tensor bias);
"""

# Compile the fused CUDA kernel
fused_relu_add = load_inline(
    name="fused_relu_add",
    cpp_sources=fused_relu_add_cpp,
    cuda_sources=fused_relu_add_source,
    functions=["fused_relu_add_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))  # Maintain same bias parameter
        self.fused_relu_add = fused_relu_add  # Inline CUDA kernel handle

    def forward(self, x):
        x = self.conv(x)
        return self.fused_relu_add.fused_relu_add_cuda(x, self.bias)