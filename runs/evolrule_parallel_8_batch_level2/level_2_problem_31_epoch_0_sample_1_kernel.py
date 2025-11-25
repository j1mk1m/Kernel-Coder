import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused element-wise kernel
fused_elementwise_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_elementwise_kernel(
    const float* input,
    const float* bias,
    float constant,
    float scaling_factor,
    int batch_size,
    int out_channels,
    int height,
    int width,
    float* output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * height * width) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % out_channels;
    int b = idx / (width * height * out_channels);

    float in_val = input[idx];
    float min_val = fminf(in_val, constant);
    float bias_val = bias[c]; // bias is (out_channels, 1, 1)
    float temp = min_val + bias_val;
    float out_val = temp * scaling_factor;

    output[idx] = out_val;
}

torch::Tensor fused_elementwise_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float constant,
    float scaling_factor
) {
    input = input.contiguous();
    bias = bias.contiguous();

    int batch_size = input.size(0);
    int out_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    // Check bias dimensions
    TORCH_CHECK(bias.sizes() == torch::IntArrayRef({out_channels, 1, 1}), 
                "Bias tensor must have shape (out_channels, 1, 1)");

    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    int elements = batch_size * out_channels * height * width;
    int blocks_per_grid = (elements + threads_per_block - 1) / threads_per_block;

    fused_elementwise_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        constant,
        scaling_factor,
        batch_size,
        out_channels,
        height,
        width,
        output.data_ptr<float>()
    );

    return output;
}
"""

fused_elementwise_cpp = """
torch::Tensor fused_elementwise_cuda(torch::Tensor input, torch::Tensor bias, float constant, float scaling_factor);
"""

# Load the fused element-wise CUDA extension
fused_elementwise = load_inline(
    name="fused_elementwise",
    cpp_sources=fused_elementwise_cpp,
    cuda_sources=fused_elementwise_source,
    functions=["fused_elementwise_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.constant_value = constant_value
        self.scaling_factor = scaling_factor
        self.fused_elementwise = fused_elementwise  # Reference to the CUDA module

    def forward(self, x):
        x = self.conv(x)
        # Apply fused operations using the custom CUDA kernel
        x = self.fused_elementwise.fused_elementwise_cuda(
            x, self.bias, self.constant_value, self.scaling_factor
        )
        return x