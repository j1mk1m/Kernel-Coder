import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused kernel source code
fused_sub_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_sub_tanh_kernel(const float* input, const float* bias, float* output, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels * height * width) {
        int c = (idx / (height * width)) % channels;
        output[idx] = tanhf(input[idx] - bias[c]);
    }
}

torch::Tensor fused_sub_tanh_cuda(torch::Tensor input, torch::Tensor bias) {
    // Check inputs are on CUDA and contiguous
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(bias.is_cuda(), "Bias must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");

    // Get input dimensions
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    // Check bias dimensions
    TORCH_CHECK(bias.size(0) == channels, "Bias channels must match input");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1, "Bias must be (C,1,1)");

    // Reshape bias to 1D
    auto bias_1d = bias.view({channels});

    // Output tensor
    auto output = torch::empty_like(input);

    // Launch kernel
    const int block_size = 256;
    int total = batch_size * channels * height * width;
    int num_blocks = (total + block_size - 1) / block_size;

    fused_sub_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias_1d.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );

    return output;
}
"""

fused_sub_tanh_cpp_source = (
    "torch::Tensor fused_sub_tanh_cuda(torch::Tensor input, torch::Tensor bias);"
)

# Load the fused kernel
fused_sub_tanh = load_inline(
    name="fused_sub_tanh",
    cpp_sources=fused_sub_tanh_cpp_source,
    cuda_sources=fused_sub_tanh_source,
    functions=["fused_sub_tanh_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_sub_tanh = fused_sub_tanh  # Load the custom kernel

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_sub_tanh.fused_sub_tanh_cuda(x, self.bias)
        return x

# The get_inputs and get_init_inputs functions remain the same as in the original code
batch_size = 32
in_channels  = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]