import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_3d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth, int height, int width, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth * height * width) return;

    int b = idx / (out_channels * depth * height * width);
    int o = (idx / (depth * height * width)) % out_channels;
    int d = (idx / (height * width)) % depth;
    int h = (idx / width) % height;
    int w = idx % width;

    float sum = 0.0f;
    for (int c = 0; c < in_channels; ++c) {
        for (int kd = -kernel_size / 2; kd <= kernel_size / 2; ++kd) {
            for (int kh = -kernel_size / 2; kh <= kernel_size / 2; ++kh) {
                for (int kw = -kernel_size / 2; kw <= kernel_size / 2; ++kw) {
                    int ic = c;
                    int id = d + kd;
                    int ih = h + kh;
                    int iw = w + kw;

                    if (id >= 0 && id < depth && ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        sum += input[b * in_channels * depth * height * width + ic * depth * height * width + id * height * width + ih * width + iw] *
                               weight[o * in_channels * kernel_size * kernel_size * kernel_size + c * kernel_size * kernel_size * kernel_size + kd * kernel_size * kernel_size + kh * kernel_size + kw];
                    }
                }
            }
        }
    }

    output[idx] = sum;
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
    const int num_blocks = (batch_size * out_channels * depth * height * width + block_size - 1) / block_size;

    convolution_3d_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth, height, width, kernel_size);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_3d_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for 3D convolution
convolution = load_inline(
    name="convolution",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["convolution_3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for group normalization
group_normalization_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_normalization_kernel(const float* input, float* output, float* mean, float* var, int batch_size, int channels, int depth, int height, int width, int groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * depth * height * width) return;

    int b = idx / (channels * depth * height * width);
    int c = (idx / (depth * height * width)) % channels;
    int d = (idx / (height * width)) % depth;
    int h = (idx / width) % height;
    int w = idx % width;

    int g = c / (channels / groups);
    int gc = c % (channels / groups);

    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < groups; ++i) {
        sum += input[b * channels * depth * height * width + (g * (channels / groups) + i) * depth * height * width + d * height * width + h * width + w];
        sum_sq += input[b * channels * depth * height * width + (g * (channels / groups) + i) * depth * height * width + d * height * width + h * width + w] * input[b * channels * depth * height * width + (g * (channels / groups) + i) * depth * height * width + d * height * width + h * width + w];
    }

    mean[idx] = sum / (groups * depth * height * width);
    var[idx] = sum_sq / (groups * depth * height * width) - mean[idx] * mean[idx];

    output[idx] = (input[idx] - mean[idx]) / sqrt(var[idx] + 1e-5);
}

torch::Tensor group_normalization_cuda(torch::Tensor input, int groups) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);

    auto output = torch::zeros_like(input);
    auto mean = torch::zeros({batch_size, channels, depth, height, width});
    auto var = torch::zeros({batch_size, channels, depth, height, width});

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * depth * height * width + block_size - 1) / block_size;

    group_normalization_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), batch_size, channels, depth, height, width, groups);

    return output;
}
"""

group_normalization_cpp_source = (
    "torch::Tensor group_normalization_cuda(torch::Tensor input, int groups);"
)

# Compile the inline CUDA code for group normalization
group_normalization = load_inline(
    name="group_normalization",
    cpp_sources=group_normalization_cpp_source,
    cuda_sources=group_normalization_source,
    functions=["group_normalization_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for clamp operation
clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void clamp_kernel(const float* input, float* output, float min_value, float max_value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > max_value ? max_value : (input[idx] < min_value ? min_value : input[idx]);
    }
}

torch::Tensor clamp_cuda(torch::Tensor input, float min_value, float max_value) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    clamp_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), min_value, max_value, size);

    return output;
}
"""

clamp_cpp_source = (
    "torch::Tensor clamp_cuda(torch::Tensor input, float min_value, float max_value);"
)

# Compile the inline CUDA code for clamp operation
clamp = load_inline(
    name="clamp",
    cpp_sources=clamp_cpp_source,
    cuda_sources=clamp_source,
    functions=["clamp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.norm = group_normalization
        self.clamp = clamp
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv.convolution_3d_cuda(x, self.weight)
        x = self.norm.group_normalization_cuda(x, groups)
        x = self.clamp.clamp_cuda(x, min_value, max_value)
        x = self.dropout(x)
        return x