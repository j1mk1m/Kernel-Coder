import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused ReLU and bias add CUDA kernel
fused_relu_bias_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_bias_add(
    const float* input, const float* bias, float* output,
    int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width) {
        return;
    }

    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int n = idx / (channels * width * height);

    float val = input[idx];
    val = fmaxf(val, 0.0f);  // ReLU
    val += bias[c];          // add per-channel bias

    output[idx] = val;
}

torch::Tensor fused_relu_bias_add_cuda(torch::Tensor input, torch::Tensor bias) {
    input = input.contiguous();
    bias = bias.contiguous();

    // Check dimensions
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    // Check bias shape
    auto bias_size = bias.sizes();
    TORCH_CHECK(bias_size[0] == channels && bias_size[1] == 1 && bias_size[2] == 1,
                "Bias tensor must be of shape (", channels, ", 1, 1)");

    auto output = torch::empty_like(input);

    int num_elements = batch_size * channels * height * width;
    const int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    fused_relu_bias_add<<<num_blocks, threads_per_block>>>(
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
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))  # shape (out_channels, 1, 1)
        self.fused_relu_bias_add = fused_relu_bias_add  # the loaded CUDA module

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_relu_bias_add.fused_relu_bias_add_cuda(x, self.bias)
        return x