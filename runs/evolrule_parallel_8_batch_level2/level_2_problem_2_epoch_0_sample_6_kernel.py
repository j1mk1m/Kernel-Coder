import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_operations_kernel(
    const float* input, const float* bias, float scaling_factor,
    float* output, int batch_size, int channels, int height, int width) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width)
        return;

    // Compute the channel index
    int c = (idx / (height * width)) % channels;
    
    float val = input[idx] + bias[c];
    
    // First clamp between 0 and 1
    val = fmaxf(0.0f, val);
    val = fminf(1.0f, val);

    // Multiply by scaling factor
    val = val * scaling_factor;

    // Second clamp between 0 and 1
    val = fmaxf(0.0f, val);
    val = fminf(1.0f, val);

    // Divide by scaling factor
    val = val / scaling_factor;

    output[idx] = val;
}

torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor) {
    // Get the dimensions
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto bias_1d = bias.view({channels});  // Reshape bias to 1D

    auto output = torch::empty_like(input);

    const int num_elements = batch_size * channels * height * width;
    const int threads_per_block = 256;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    fused_operations_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        bias_1d.data_ptr<float>(),
        scaling_factor,
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width);

    cudaDeviceSynchronize();  // Ensure kernel finishes

    return output;
}
"""

fused_ops_header = (
    "torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor);"
)

# Compile the fused CUDA operations
fused_ops = load_inline(
    name="fused_ops",
    cuda_sources=fused_ops_source,
    cpp_sources=fused_ops_header,
    functions=["fused_operations_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_ops.fused_operations_cuda(x, self.bias, self.scaling_factor)
        return x

# The get_inputs() and get_init_inputs() remain as in the original code