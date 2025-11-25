import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for bias subtraction and tanh activation
fused_sub_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_sub_tanh_kernel(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int out_channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * height * width) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % out_channels;
    int b = idx / (out_channels * width * height);

    float bias_val = bias[c];
    float val = input[idx] - bias_val;
    output[idx] = tanh(val);
}

torch::Tensor fused_sub_tanh_cuda(torch::Tensor input, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto out_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    auto size = batch_size * out_channels * height * width;

    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_sub_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, out_channels, height, width
    );

    return output;
}
"""

cpp_source = """
extern "C" {
    torch::Tensor fused_sub_tanh_cuda(torch::Tensor input, torch::Tensor bias);
}
"""

# Compile the fused CUDA kernel
fused_sub_tanh = load_inline(
    name="fused_sub_tanh",
    cpp_sources=cpp_source,
    cuda_sources=fused_sub_tanh_source,
    functions=["fused_sub_tanh_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_sub_tanh = fused_sub_tanh

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()  # Ensure contiguous memory layout
        x = self.fused_sub_tanh.fused_sub_tanh_cuda(x, self.bias)
        return x

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]