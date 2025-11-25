import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for activations and bias addition
fused_activation_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void fused_activations_kernel(
    const float* input, const float* bias, float* output,
    int batch_size, int out_channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth * height * width) return;

    // Compute indices
    int w = idx % width;
    int h = (idx / width) % height;
    int d = (idx / (width * height)) % depth;
    int c = (idx / (width * height * depth)) % out_channels;
    int b = idx / (width * height * depth * out_channels);

    float x = input[idx];

    // Apply ReLU
    x = fmaxf(x, 0.0f);

    // Apply LeakyReLU (slope 0.01)
    if (x < 0) {
        x = 0.01f * x;
    }

    // Apply GELU approximation
    const float sqrt_2_over_pi = 0.7978845608f;
    const float approximation_term = 0.044715f;
    float inner = sqrt_2_over_pi * (x + approximation_term * x * x * x);
    float tanh_val = tanhf(inner);
    x = 0.5f * x * (1.0f + tanh_val);

    // Apply Sigmoid
    x = 1.0f / (1.0f + expf(-x));

    // Add bias (shape: [out_channels, 1, 1, 1])
    x += bias[c];

    output[idx] = x;
}

torch::Tensor fused_activation_cuda(torch::Tensor input, torch::Tensor bias) {
    if (input.dim() != 5) {
        std::cerr << "Input must be a 5D tensor" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (bias.dim() != 4) {
        std::cerr << "Bias must be a 4D tensor" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (input.size(1) != bias.size(0)) {
        std::cerr << "Bias channels must match input's" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (bias.size(1) != 1 || bias.size(2) != 1 || bias.size(3) != 1) {
        std::cerr << "Bias must have shape (C, 1, 1, 1)" << std::endl;
        exit(EXIT_FAILURE);
    }

    int batch_size = input.size(0);
    int out_channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    int N = batch_size * out_channels * depth * height * width;

    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;

    auto output = torch::empty_like(input);

    fused_activations_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, out_channels, depth, height, width
    );

    return output;
}
"""

fused_activation_cpp_source = "torch::Tensor fused_activation_cuda(torch::Tensor input, torch::Tensor bias);"

# Compile the fused kernel
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_activation_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.conv(x)
        return self.fused_activation.fused_activation_cuda(x, self.bias)

# Environment setup remains the same as original
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