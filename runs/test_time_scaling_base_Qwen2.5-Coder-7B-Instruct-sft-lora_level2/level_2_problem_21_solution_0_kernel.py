import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    int b = blockIdx.x / (output_width * output_height);
    int oh = blockIdx.x % (output_width * output_height) / output_width;
    int ow = blockIdx.x % (output_width * output_height) % output_width;
    int ic = blockIdx.y;
    int oc = threadIdx.x;

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = oh + kh;
            int iw = ow + kw;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                int in_idx = b * in_channels * height * width + ic * height * width + ih * width + iw;
                int wt_idx = oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw;
                sum += input[in_idx] * weight[wt_idx];
            }
        }
    }

    int out_idx = b * out_channels * output_height * output_width + oc * output_height * output_width + oh * output_width + ow;
    output[out_idx] = sum;
}

void convolution_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor output) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height = input.size(2);
    auto width = input.size(3);
    auto kernel_size = weight.size(2);

    int output_height = (height - kernel_size + 1) / 1;
    int output_width = (width - kernel_size + 1) / 1;

    const int block_size = 256;
    const int num_blocks = (out_channels * output_height * output_width) / block_size;

    convolution_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width, kernel_size);
}
"""

convolution_cpp_source = (
    "void convolution_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor output);"
)

# Compile the inline CUDA code for convolution
convolution = load_inline(
    name="convolution",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["convolution_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for bias addition
bias_addition_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bias_addition_kernel(const float* input, const float* bias, float* output, int batch_size, int out_channels, int height, int width) {
    int b = blockIdx.x / (height * width);
    int c = blockIdx.x % (height * width) / width;
    int h = blockIdx.x % (height * width) % width;
    int w = threadIdx.x;

    int in_idx = b * out_channels * height * width + c * height * width + h * width + w;
    int bias_idx = c;
    output[in_idx] = input[in_idx] + bias[bias_idx];
}

void bias_addition_forward_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor output) {
    auto batch_size = input.size(0);
    auto out_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * height * width) / block_size;

    bias_addition_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), batch_size, out_channels, height, width);
}
"""

bias_addition_cpp_source = (
    "void bias_addition_forward_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor output);"
)

# Compile the inline CUDA code for bias addition
bias_addition = load_inline(
    name="bias_addition",
    cpp_sources=bias_addition_cpp_source,
    cuda_sources=bias_addition_source,
    functions=["bias_addition_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for scaling
scaling_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scaling_kernel(const float* input, const float* scale, float* output, int batch_size, int out_channels, int height, int width) {
    int b = blockIdx.x / (height * width);
    int c = blockIdx.x % (height * width) / width;
    int h = blockIdx.x % (height * width) % width;
    int w = threadIdx.x;

    int in_idx = b * out_channels * height * width + c * height * width + h * width + w;
    int scale_idx = c;
    output[in_idx] = input[in_idx] * scale[scale_idx];
}

void scaling_forward_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor output) {
    auto batch_size = input.size(0);
    auto out_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * height * width) / block_size;

    scaling_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), scale.data_ptr<float>(), output.data_ptr<float>(), batch_size, out_channels, height, width);
}
"""

scaling_cpp_source = (
    "void scaling_forward_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor output);"
)

# Compile the inline CUDA code for scaling
scaling = load_inline(
    name="scaling",
    cpp_sources=scaling_cpp_source,
    cuda_sources=scaling_source,
    functions=["scaling_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for group normalization
group_normalization_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_normalization_kernel(const float* input, float* mean, float* var, float* output, int batch_size, int out_channels, int height, int width, int num_groups) {
    int g = blockIdx.x / (out_channels * height * width);
    int b = blockIdx.x % (out_channels * height * width) / (height * width);
    int oc = blockIdx.x % (out_channels * height * width) % (height * width);
    int h = blockIdx.y;
    int w = threadIdx.x;

    int in_idx = b * out_channels * height * width + oc * height * width + h * width + w;
    float value = input[in_idx];

    atomicAdd(&mean[g], value);
    atomicAdd(&var[g], value * value);
}

void group_normalization_forward_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor var, torch::Tensor output) {
    auto batch_size = input.size(0);
    auto out_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto num_groups = out_channels / num_groups;

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * height * width) / block_size;

    group_normalization_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), output.data_ptr<float>(), batch_size, out_channels, height, width, num_groups);
}
"""

group_normalization_cpp_source = (
    "void group_normalization_forward_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor var, torch::Tensor output);"
)

# Compile the inline CUDA code for group normalization
group_normalization = load_inline(
    name="group_normalization",
    cpp_sources=group_normalization_cpp_source,
    cuda_sources=group_normalization_source,
    functions=["group_normalization_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.bias = bias_addition
        self.scale = scaling
        self.group_norm = group_normalization

    def forward(self, x):
        x = self.conv.convolution_forward_cuda(x, self.weight, self.output)
        x = self.bias.bias_addition_forward_cuda(x, self.bias, self.output)
        x = self.scale.scaling_forward_cuda(x, self.scale, self.output)
        x = torch.sigmoid(x)
        x = self.group_norm.group_normalization_forward_cuda(x, self.mean, self.var, self.output)
        return x