import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused ReLU + bias addition CUDA kernel
fused_relu_bias_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_bias_add(
    const float* input, const float* bias, float* output,
    int batch_size, int channels, int height, int width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;

    float val = input[idx];
    val = max(val, 0.0f);
    val += bias[c]; // bias is (out_channels, 1, 1), so bias[c] is the value for channel c
    output[idx] = val;
}

torch::Tensor fused_relu_bias_add_cuda(torch::Tensor input, torch::Tensor bias) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto output = torch::empty_like(input);

    int num_elements = batch_size * channels * height * width;
    const int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_relu_bias_add<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width
    );

    return output;
}
"""

fused_relu_bias_add_cpp_source = (
    "torch::Tensor fused_relu_bias_add_cuda(torch::Tensor input, torch::Tensor bias);"
)

# Compile the fused kernel
fused_relu_bias_add = load_inline(
    name="fused_relu_bias_add",
    cpp_sources=fused_relu_bias_add_cpp_source,
    cuda_sources=fused_relu_bias_add_source,
    functions=["fused_relu_bias_add_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_relu_bias_add = fused_relu_bias_add

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_relu_bias_add.fused_relu_bias_add_cuda(x, self.bias)
        return x

def get_inputs():
    batch_size = 128
    in_channels = 64
    height = width = 128
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [64, 128, 3, (128, 1, 1)]