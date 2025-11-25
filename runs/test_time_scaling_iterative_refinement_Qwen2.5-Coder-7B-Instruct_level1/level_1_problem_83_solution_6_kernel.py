import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for depthwise 2D convolution
depthwise_convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int height, int width, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * in_channels * height * width) {
        return;
    }

    int b = idx / (in_channels * height * width);
    int c = (idx / (height * width)) % in_channels;
    int h = (idx / width) % height;
    int w = idx % width;

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = h + kh;
            int iw = w + kw;
            if (ih >= height || iw >= width) {
                continue;
            }
            int input_idx = b * in_channels * height * width + c * height * width + ih * width + iw;
            int weight_idx = c * kernel_size * kernel_size + kh * kernel_size + kw;
            sum += input[input_idx] * weight[weight_idx];
        }
    }
    output[idx] = sum;
}

torch::Tensor depthwise_convolution_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (batch_size * in_channels * height * width + block_size - 1) / block_size;

    depthwise_convolution_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, height, width, kernel_size);

    return output;
}
"""

depthwise_convolution_cpp_source = (
    "torch::Tensor depthwise_convolution_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for depthwise 2D convolution
depthwise_convolution = load_inline(
    name="depthwise_convolution",
    cpp_sources=depthwise_convolution_cpp_source,
    cuda_sources=depthwise_convolution_source,
    functions=["depthwise_convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.depthwise_convolution = depthwise_convolution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthwise_convolution.depthwise_convolution_cuda(x, x[:, :, :kernel_size, :kernel_size])