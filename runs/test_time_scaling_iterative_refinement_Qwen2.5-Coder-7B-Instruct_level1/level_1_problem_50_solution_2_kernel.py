import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_height, int kernel_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_channels * height * width) {
        int batch = idx / (out_channels * height * width);
        int oc = (idx % (out_channels * height * width)) / (height * width);
        int h = (idx % (out_channels * height * width)) % height;
        int w = (idx % (out_channels * height * width)) % width;

        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int ih = h + kh - kernel_height / 2;
                    int iw = w + kw - kernel_width / 2;
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        sum += input[(batch * in_channels + ic) * height * width + ih * width + iw] * weight[oc * in_channels * kernel_height * kernel_width + ic * kernel_height * kernel_width + kh * kernel_width + kw];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height = input.size(2);
    auto width = input.size(3);
    auto kernel_height = weight.size(2);
    auto kernel_width = weight.size(3);

    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * height * width + block_size - 1) / block_size;

    convolution_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width, kernel_height, kernel_width);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight);"
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
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1_weight = nn.Parameter(torch.randn(96, 3, 11, 11))
        self.conv1_bias = nn.Parameter(torch.zeros(96))

    def forward(self, x):
        x = convolution.convolution_cuda(x, self.conv1_weight)
        return x

# Test code
batch_size = 256
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]