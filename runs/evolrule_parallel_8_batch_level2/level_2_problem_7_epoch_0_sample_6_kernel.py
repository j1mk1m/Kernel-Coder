import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code for fused activations and bias addition
fused_activation_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void fused_activation_kernel(
    const float* input, 
    const float* bias,
    float* output,
    int batch_size,
    int out_channels,
    int depth,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth * height * width)
        return;

    int w = idx % width;
    int h = (idx / width) % height;
    int d = (idx / (width * height)) % depth;
    int c = (idx / (width * height * depth)) % out_channels;
    int n = idx / (width * height * depth * out_channels);

    float temp = input[idx];

    // Apply ReLU
    temp = fmaxf(temp, 0.0f);

    // Apply Leaky ReLU (redundant but required by original code)
    float leaky_relu_val = fmaxf(temp, 0.0f) + 0.01f * fminf(temp, 0.0f);
    temp = leaky_relu_val;

    // Apply GELU approximation
    float x = temp;
    float inner = sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x);
    float sigmoid_inner = 1.0f / (1.0f + expf(-inner));
    float gelu_val = x * sigmoid_inner;
    temp = gelu_val;

    // Apply Sigmoid
    float exp_neg_x = expf(-temp);
    float sigmoid_val = 1.0f / (1.0f + exp_neg_x);
    temp = sigmoid_val;

    // Add bias
    float bias_val = bias[c];
    temp += bias_val;

    output[idx] = temp;
}

torch::Tensor fused_activation_cuda(
    torch::Tensor input,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    int out_channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    int total_elements = batch_size * out_channels * depth * height * width;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    fused_activation_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, out_channels, depth, height, width
    );

    return output;
}
"""

fused_activation_cpp_source = """
torch::Tensor fused_activation_cuda(torch::Tensor input, torch::Tensor bias);
"""

# Load the fused activation kernel
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_activation_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_flags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_activation = fused_activation  # Access the compiled kernel

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_activation.fused_activation_cuda(x, self.bias)
        return x

# Ensure the get_inputs and get_init_inputs functions are as provided
batch_size = 64
in_channels = 8
out_channels = 32
depth, height, width = 32, 64, 64
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]