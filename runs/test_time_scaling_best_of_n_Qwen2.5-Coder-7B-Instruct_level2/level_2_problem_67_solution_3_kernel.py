import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_channels * height * width) {
        int o_idx = idx / (height * width);
        int h_idx = (idx % (height * width)) / width;
        int w_idx = (idx % (height * width)) % width;
        float sum = 0.0f;
        for (int c_idx = 0; c_idx < in_channels; ++c_idx) {
            int i_idx = c_idx * height * width + (h_idx + kernel_size / 2) * width + (w_idx + kernel_size / 2);
            sum += input[i_idx] * weight[o_idx * in_channels + c_idx];
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
    auto kernel_size = weight.size(2);
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * height * width + block_size - 1) / block_size;

    convolution_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width, kernel_size);

    return output;
}
"""

# Define the custom CUDA kernel for GELU activation
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

# Define the custom CUDA kernel for adaptive average pooling
avg_pooling_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pooling_kernel(const float* input, float* output, int batch_size, int in_channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * in_channels) {
        int h_idx = idx % height;
        int w_idx = idx / height;
        int out_h_idx = h_idx / 2;
        int out_w_idx = w_idx / 2;
        int out_idx = out_h_idx * width / 2 + out_w_idx;
        atomicAdd(&output[out_idx], input[idx]);
        atomicAdd(&output[out_idx + 1], input[idx + 1]);
        atomicAdd(&output[out_idx + width / 2], input[idx + height * width]);
        atomicAdd(&output[out_idx + width / 2 + 1], input[idx + height * width + 1]);
    }
}

torch::Tensor avg_pooling_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto output = torch::zeros({batch_size, in_channels, height / 2, width / 2}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * in_channels + block_size - 1) / block_size;

    avg_pooling_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, height, width);

    return output;
}
"""

# Compile the inline CUDA code for convolution, GELU, and adaptive average pooling
convolution = load_inline(
    name="convolution",
    cpp_sources="",
    cuda_sources=convolution_source,
    functions=["convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

gelu = load_inline(
    name="gelu",
    cpp_sources="",
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

avg_pooling = load_inline(
    name="avg_pooling",
    cpp_sources="",
    cuda_sources=avg_pooling_source,
    functions=["avg_pooling_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.gelu = gelu
        self.avg_pooling = avg_pooling

    def forward(self, x):
        x = self.conv.convolution_cuda(x, self.weight)
        x = self.gelu.gelu_cuda(x)
        x = self.avg_pooling.avg_pooling_cuda(x)
        x = x.squeeze(-1).squeeze(-1)
        return x