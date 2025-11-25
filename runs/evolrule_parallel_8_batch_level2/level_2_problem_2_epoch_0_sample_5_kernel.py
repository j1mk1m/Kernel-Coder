import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused element-wise CUDA kernel
fused_elementwise_cuda_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_elementwise_kernel(
    const float* input,
    const float* bias,
    float scaling_factor,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * channels * height * width)
        return;

    int w = index % width;
    int h = (index / width) % height;
    int c = (index / (width * height)) % channels;
    int n = index / (channels * width * height);

    float val = input[index] + bias[c];

    if (val < 0.0f) val = 0.0f;
    else if (val > 1.0f) val = 1.0f;

    val *= scaling_factor;

    if (val < 0.0f) val = 0.0f;
    else if (val > 1.0f) val = 1.0f;

    val /= scaling_factor;

    output[index] = val;
}

torch::Tensor fused_elementwise_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor
) {
    AT_ASSERT(input.dim() == 4);
    AT_ASSERT(bias.dim() == 3);
    AT_ASSERT(bias.size(0) == input.size(1));

    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int elements = batch_size * channels * height * width;
    const int blocks = (elements + threads - 1) / threads;

    fused_elementwise_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );

    return output;
}
"""

# Header for the C++ function
fused_elementwise_header = """
#include <torch/extension.h>

torch::Tensor fused_elementwise_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor
);
"""

# Compile the fused element-wise kernel
fused_elementwise = load_inline(
    name="fused_elementwise",
    cpp_sources=fused_elementwise_header,
    cuda_sources=fused_elementwise_cuda_src,
    functions=["fused_elementwise_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_elementwise = fused_elementwise  # Reference to the CUDA function

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_elementwise.fused_elementwise_cuda(x, self.bias, self.scaling_factor)
        return x

# The get_inputs and get_init_inputs functions remain unchanged from the original code
batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 128 
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]