import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused convolution and GELU
conv_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_gelu_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels) return;

    int oc = idx / batch_size;
    int ic = idx % batch_size;
    int h = ic / width;
    int w = ic % width;

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = h + kh;
            int iw = w + kw;
            if (ih >= height || iw >= width) continue;
            int i_idx = (ic * kernel_size + kh) * kernel_size + kw;
            int o_idx = (oc * height + ih) * width + iw;
            sum += input[i_idx] * weight[o_idx];
        }
    }

    output[idx] = sum * 0.79788456f + 0.04471505f * sum * sum * sum;
}

torch::Tensor conv_gelu_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height = input.size(2);
    auto width = input.size(3);

    auto output = torch::zeros({batch_size, out_channels}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels + block_size - 1) / block_size;

    conv_gelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width, kernel_size);

    return output;
}
"""

conv_gelu_cpp_source = (
    "torch::Tensor conv_gelu_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size);"
)

# Compile the inline CUDA code for fused convolution and GELU
conv_gelu = load_inline(
    name="conv_gelu",
    cpp_sources=conv_gelu_cpp_source,
    cuda_sources=conv_gelu_source,
    functions=["conv_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for global average pooling
avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool_kernel(const float* input, float* output, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels) return;

    int c = idx / batch_size;
    int b = idx % batch_size;

    float sum = 0.0f;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int i_idx = (b * channels + c) * height * width + h * width + w;
            sum += input[i_idx];
        }
    }

    output[idx] = sum / (height * width);
}

torch::Tensor avg_pool_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    auto output = torch::zeros({batch_size, channels}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels + block_size - 1) / block_size;

    avg_pool_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width);

    return output;
}
"""

avg_pool_cpp_source = (
    "torch::Tensor avg_pool_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for global average pooling
avg_pool = load_inline(
    name="avg_pool",
    cpp_sources=avg_pool_cpp_source,
    cuda_sources=avg_pool_source,
    functions=["avg_pool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv_gelu = conv_gelu
        self.avg_pool = avg_pool

    def forward(self, x):
        x = self.conv_gelu.conv_gelu_cuda(x, self.weight, kernel_size)
        x = self.avg_pool.avg_pool_cuda(x)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def initialize_weights(self, kernel_size):
        self.weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device='cuda')