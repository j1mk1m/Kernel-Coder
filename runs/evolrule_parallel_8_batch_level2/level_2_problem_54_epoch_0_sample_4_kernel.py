import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for element-wise multiplication, LeakyReLU, and GELU
fused_kernel_source = """
#include <torch/extension.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void fused_operation_kernel(
    const float* input,
    const float* multiplier,
    float negative_slope,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width) return;

    // Compute 4D indices from linear index
    int n = idx / (channels * height * width);
    int remaining = idx % (channels * height * width);
    int c = remaining / (height * width);
    remaining = remaining % (height * width);
    int h = remaining / width;
    int w = remaining % width;

    // Get input value
    float val = input[idx];

    // Multiply by channel-wise multiplier
    val *= multiplier[c];

    // Apply LeakyReLU
    if (val < 0) val *= negative_slope;

    // Apply GELU approximation
    float x = val;
    float inner = sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x);
    float tanh_inner = tanhf(inner);
    val = 0.5f * x * (1.0f + tanh_inner);

    output[idx] = val;
}

torch::Tensor fused_operation_cuda(
    torch::Tensor input,
    torch::Tensor multiplier,
    float negative_slope
) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(multiplier.is_cuda(), "Multiplier must be on CUDA");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D tensor");
    TORCH_CHECK(multiplier.dim() == 3, "Multiplier must be 3D tensor");
    TORCH_CHECK(multiplier.size(1) == 1 && multiplier.size(2) == 1, "Multiplier must have shape (C,1,1)");
    TORCH_CHECK(input.size(1) == multiplier.size(0), "Channels of input and multiplier must match");

    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto output = torch::empty_like(input);

    const int total_elements = batch_size * channels * height * width;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    fused_operation_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        negative_slope,
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );

    return output;
}
"""

# Corresponding C++ header declarations
fused_kernel_cpp_source = (
    "torch::Tensor fused_operation_cuda(torch::Tensor input, torch::Tensor multiplier, float negative_slope);"
)

# Compile the fused CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_operation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()  # Required to access negative_slope
        self.fused_op = fused_op  # CUDA fused operation module

    def forward(self, x):
        x = self.conv(x)
        # Extract parameters for fused kernel
        negative_slope = self.leaky_relu.negative_slope
        multiplier = self.multiplier
        # Apply fused operation
        x = self.fused_op.fused_operation_cuda(x, multiplier, negative_slope)
        return x