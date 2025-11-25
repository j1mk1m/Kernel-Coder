import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c_out = blockIdx.y * blockDim.y + threadIdx.y;
    int h_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < batch_size && c_out < out_channels && h_out < height) {
        float sum = 0.0f;
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int i_h = h_out * stride + kh - pad;
                    int i_w = h_out * stride + kw - pad;
                    if (i_h >= 0 && i_h < height && i_w >= 0 && i_w < width) {
                        sum += input[n * in_channels * height * width + c_in * height * width + i_h * width + i_w] * weight[c_out * in_channels * kernel_size * kernel_size + c_in * kernel_size * kernel_size + kh * kernel_size + kw];
                    }
                }
            }
        }
        output[n * out_channels * height * width + c_out * height * width + h_out * width] = sum;
    }
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int stride, int pad) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height = input.size(2);
    auto width = input.size(3);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels, height, width});

    const int block_size = 16;
    const int num_blocks = (output.numel() + block_size - 1) / block_size;

    convolution_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width, kernel_size);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int stride, int pad);"
)

# Compile the inline CUDA code for convolution
convolution = load_inline(
    name="convolution",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.conv.convolution_cuda(x, self.weight, self.stride, self.pad)
        x = torch.multiply(torch.tanh(torch.nn.functional.softplus(x)), x)
        x = self.bn(x)
        return x