import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

convolution_min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_min_kernel(const float* input, const float* weight, float* output, int channels, int height, int width, int kernel_size, float constant_value) {
    int c = blockIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;

    if (c >= channels || h >= height || w >= width) {
        return;
    }

    int pad_h = kernel_size / 2;
    int pad_w = kernel_size / 2;

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = h + kh - pad_h;
            int iw = w + kw - pad_w;

            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                sum += input[(h * width + w) * channels + c] * weight[kh * kernel_size + kw];
            }
        }
    }

    output[h * width + w] = sum < constant_value ? sum : constant_value;
}
"""

convolution_min_cpp_source = (
    "void convolution_min_kernel(const float* input, const float* weight, float* output, int channels, int height, int width, int kernel_size, float constant_value);"
)

convolution_min = load_inline(
    name="convolution_min",
    cpp_sources=convolution_min_cpp_source,
    cuda_sources=convolution_min_source,
    functions=["convolution_min_kernel"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.convolution_min = convolution_min

    def forward(self, x):
        x = self.conv(x)
        x = self.convolution_min.convolution_min_kernel(x.data_ptr(), self.conv.weight.data_ptr(), x.data_ptr(), self.out_channels, x.size(2), x.size(3), self.kernel_size, self.constant_value)
        x = x + self.bias
        x = x * self.scaling_factor
        return x