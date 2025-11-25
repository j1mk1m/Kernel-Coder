import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, const float* bias, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size, int stride, int padding, int dilation) {
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    if (n >= batch_size || c >= out_channels) return;

    int h_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int w_out = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    int h_in_start = n * in_channels * height * width + c * height * width;
    int w_in_start = h_in_start;
    int h_out_start = n * out_channels * h_out * w_out + c * h_out * w_out;
    int w_out_start = h_out_start;

    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int w_in = blockIdx.x * blockDim.x + threadIdx.x;
            int h_in = blockIdx.y * blockDim.y + threadIdx.y;
            int w_out = blockIdx.x * blockDim.x + threadIdx.x;
            int h_out = blockIdx.y * blockDim.y + threadIdx.y;
            if (w_in >= width || h_in >= height) continue;

            int w_in_idx = w_in_start + h_in * width + w_in;
            int w_out_idx = w_out_start + h_out * w_out + w_out;
            output[w_out_idx] += input[w_in_idx] * weight[i * kernel_size + j];
        }
    }

    if (bias != nullptr) {
        output[h_out_start + w_out_start] += bias[c];
    }
}
"""

convolution_cpp_source = (
    "void convolution_kernel(const float* input, const float* weight, const float* bias, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size, int stride, int padding, int dilation);"
)

# Compile the inline CUDA code for convolution
convolution = load_inline(
    name="convolution",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["convolution_kernel"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        weight = self.conv.weight.data.cpu().numpy()
        bias = self.conv.bias.data.cpu().numpy()

        output = torch.zeros((batch_size, out_channels, height, width)).to(x.device)
        self.conv.convolution_kernel(x.contiguous().data_ptr(), weight, bias, output.data_ptr(), batch_size, in_channels, out_channels, height, width, kernel_size, 1, 1, 1)

        x = output
        x = x + self.bias
        x = x * self.scale
        x = torch.sigmoid(x)
        x = self.group_norm(x)
        return x