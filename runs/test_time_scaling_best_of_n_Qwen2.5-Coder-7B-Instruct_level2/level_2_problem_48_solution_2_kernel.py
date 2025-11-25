import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
convolution_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_3d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth, int height, int width, int kernel_size) {
    int b = blockIdx.x / (height * width);
    int h = (blockIdx.x % (height * width)) / width;
    int w = blockIdx.x % width;
    int oc = blockIdx.y;

    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int id = max(0, b * depth + h * kernel_size + kd - kernel_size / 2);
                    int ih = max(0, b * height + h * kernel_size + kh - kernel_size / 2);
                    int iw = max(0, b * width + w * kernel_size + kw - kernel_size / 2);
                    sum += input[id * in_channels * depth * height * width + ic * depth * height * width + ih * height * width + iw * width + ic] *
                           weight[oc * in_channels * kernel_size * kernel_size * kernel_size + ic * kernel_size * kernel_size * kernel_size + kd * kernel_size * kernel_size + kh * kernel_size + kw];
                }
            }
        }
    }
    output[b * out_channels * depth * height * width + oc * depth * height * width + h * height * width + w * width] = sum;
}

torch::Tensor convolution_3d_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels, depth, height, width}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * height * width + block_size - 1) / block_size;

    convolution_3d_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth, height, width, kernel_size);

    return output;
}
"""

convolution_3d_cpp_source = (
    "torch::Tensor convolution_3d_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for 3D convolution
convolution_3d = load_inline(
    name="convolution_3d",
    cpp_sources=convolution_3d_cpp_source,
    cuda_sources=convolution_3d_source,
    functions=["convolution_3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = convolution_3d
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x, self.conv.weight)
        x = x * self.scaling_factor
        x = torch.tanh(x)
        x = x * self.bias
        x = torch.sigmoid(x)
        return x