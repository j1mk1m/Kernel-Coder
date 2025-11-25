import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fusing Conv3D + ReLU + LeakyReLU + GELU + Sigmoid + Bias
# Note: Fusing all activations and bias into a single kernel is complex. Instead, we fuse the main operations where possible.

# First, create a fused kernel for ReLU + LeakyReLU + GELU + Sigmoid + Bias Addition
# However, combining all these may not be straightforward. Let's instead fuse the activation functions and bias addition.

# Define fused activation functions + bias addition kernel
# This kernel will apply all activations in sequence and add the bias in a single step to reduce memory copies and kernel launches.

fused_activations_with_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_activations_bias_kernel(
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
    if (idx >= batch_size * out_channels * depth * height * width) return;

    int channel = (idx / (depth * height * width)) % out_channels;

    // Compute activations in sequence
    float x = input[idx] + bias[channel];  // Add bias first (as per original code order)
    x = fmaxf(x, 0.0f);                   // ReLU
    x = fmaxf(x, 0.01f * x);              // LeakyReLU (negative slope 0.01)
    x = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * powf(x, 3)))); // GELU approximation
    x = 1.0f / (1.0f + expf(-x));         // Sigmoid

    output[idx] = x;
}

torch::Tensor fused_activations_with_bias_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    int batch_size,
    int out_channels,
    int depth,
    int height,
    int width
) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_activations_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        depth,
        height,
        width
    );

    return output;
}
"""

fused_activations_cpp_source = (
    "torch::Tensor fused_activations_with_bias_cuda("
    "torch::Tensor input, torch::Tensor bias, int batch_size, int out_channels, int depth, int height, int width);"
)

# Compile the fused activations kernel
fused_activations = load_inline(
    name="fused_activations_with_bias",
    cpp_sources=fused_activations_cpp_source,
    cuda_sources=fused_activations_source,
    functions=["fused_activations_with_bias_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_activations = fused_activations

    def forward(self, x):
        # Run convolution using PyTorch's optimized Conv3D
        x = self.conv(x)
        # Apply all fused activations and bias addition in a single kernel
        batch_size, out_channels, depth, height, width = x.shape
        x = self.fused_activations.fused_activations_with_bias_cuda(
            x,
            self.bias,
            batch_size,
            out_channels,
            depth,
            height,
 width
        )
        return x

# Keep the original functions as given
def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]