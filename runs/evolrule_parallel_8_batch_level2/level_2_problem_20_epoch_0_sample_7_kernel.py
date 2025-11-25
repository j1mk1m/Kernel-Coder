import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused element-wise operations
fused_elementwise_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_elementwise_kernel(
    const float* conv_out,
    const float* bias,
    float* output,
    int batch_size,
    int out_channels,
    int depth,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth * height * width) {
        return;
    }

    float a = conv_out[idx];
    int c = (idx / (depth * height * width)) % out_channels;
    float bias_val = bias[c];

    float temp1 = a + bias_val;
    float temp2 = temp1 + a;
    float temp3 = temp2 * a;
    float result = temp3 + a;

    output[idx] = result;
}

torch::Tensor fused_elementwise_cuda(torch::Tensor conv_out, torch::Tensor bias) {
    conv_out = conv_out.contiguous();
    bias = bias.contiguous();

    int batch_size = conv_out.size(0);
    int out_channels = conv_out.size(1);
    int depth = conv_out.size(2);
    int height = conv_out.size(3);
    int width = conv_out.size(4);

    auto output = torch::empty_like(conv_out);

    int numel = batch_size * out_channels * depth * height * width;

    const int block_size = 256;
    int num_blocks = (numel + block_size - 1) / block_size;

    fused_elementwise_kernel<<<num_blocks, block_size>>>(
        conv_out.data_ptr<float>(),
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

fused_elementwise_cpp_source = "torch::Tensor fused_elementwise_cuda(torch::Tensor conv_out, torch::Tensor bias);"

# Compile the inline CUDA code for the fused element-wise operations
fused_elementwise = load_inline(
    name="fused_elementwise",
    cpp_sources=fused_elementwise_cpp_source,
    cuda_sources=fused_elementwise_source,
    functions=["fused_elementwise_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_elementwise = fused_elementwise

    def forward(self, x):
        conv_out = self.conv_transpose(x)
        x = self.fused_elementwise.fused_elementwise_cuda(conv_out, self.bias)
        return x

# The following functions are provided as per the original architecture
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]