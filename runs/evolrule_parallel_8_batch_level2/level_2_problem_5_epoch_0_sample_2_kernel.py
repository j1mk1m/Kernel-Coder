import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code for fused bias subtraction and tanh activation
fused_bias_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bias_tanh_kernel(
    const float* x_data,
    const float* bias_data,
    float* out_data,
    int batch_size,
    int channels,
    int height,
    int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * channels * height * width) return;

    int w = index % width;
    int h = (index / width) % height;
    int c = (index / (width * height)) % channels;
    int b = index / (width * height * channels);

    float val = x_data[index] - bias_data[c];
    out_data[index] = tanhf(val);
}

torch::Tensor fused_bias_tanh_cuda(torch::Tensor x, torch::Tensor bias) {
    AT_ASSERTM(x.device().is_cuda(), "x must be a CUDA tensor");
    AT_ASSERTM(bias.device().is_cuda(), "bias must be a CUDA tensor");
    AT_ASSERTM(x.size(1) == bias.size(0), "Channels of x and bias must match");

    auto output = torch::empty_like(x);

    int batch_size = x.size(0);
    int channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);

    int num_elements = batch_size * channels * height * width;

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_bias_tanh_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width
    );

    return output;
}
"""

fused_bias_tanh_cpp_source = (
    "torch::Tensor fused_bias_tanh_cuda(torch::Tensor x, torch::Tensor bias);"
)

# Load the fused CUDA kernel
fused_op = load_inline(
    name="fused_bias_tanh",
    cpp_sources=fused_bias_tanh_cpp_source,
    cuda_sources=fused_bias_tanh_source,
    functions=["fused_bias_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_bias_tanh_cuda(x, self.bias)
        return x

# Input and initialization functions
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