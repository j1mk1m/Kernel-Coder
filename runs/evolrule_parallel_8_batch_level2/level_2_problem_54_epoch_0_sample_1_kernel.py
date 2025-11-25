import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, multiplies by a learnable scalar, applies LeakyReLU, and then GELU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape)) 
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        x = torch.nn.functional.gelu(x)
        return x

# Original get_inputs and get_init_inputs are already provided.

# Now, the optimized code starts here.

from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for fused operations (scaling, LeakyReLU, GELU approximation)
fused_ops_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void fused_operations_kernel(
    const float* x_data,
    const float* multiplier_data,
    float* out_data,
    int batch_size, int channels, int height, int width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width)
        return;

    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int b = idx / (width * height * channels);

    float x_val = x_data[b * channels * height * width + c * height * width + h * width + w];
    float multiplier_val = multiplier_data[c];

    float scaled = x_val * multiplier_val;
    float leaky = (scaled > 0.0f) ? scaled : 0.01f * scaled;

    // GELU approximation using the tanh-based formula for faster computation
    const float sqrt_2_over_pi = sqrt(2.0f / M_PI);
    float cubic_term = leaky * leaky * leaky;
    float inner = sqrt_2_over_pi * (leaky + 0.044715f * cubic_term);
    float tanh_val = tanhf(inner);
    float gelu = leaky * 0.5f * (1.0f + tanh_val);

    out_data[b * channels * height * width + c * height * width + h * width + w] = gelu;
}

torch::Tensor fused_operations_cuda(torch::Tensor x, torch::Tensor multiplier) {
    int batch_size = x.size(0);
    int channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);

    auto out = torch::empty_like(x);

    int num_elements = batch_size * channels * height * width;
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    fused_operations_kernel<<<blocks_per_grid, threads_per_block>>>(
        x.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size, channels, height, width
    );

    return out;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_operations_cuda(torch::Tensor x, torch::Tensor multiplier);"

# Compile the fused operations kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_operations_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape)) 

    def forward(self, x):
        x = self.conv(x)
        # Apply fused operations (scaling, LeakyReLU, and GELU approximation)
        x = fused_ops.fused_operations_cuda(x, self.multiplier)
        return x

# The get_inputs and get_init_inputs functions are already provided in the original code.