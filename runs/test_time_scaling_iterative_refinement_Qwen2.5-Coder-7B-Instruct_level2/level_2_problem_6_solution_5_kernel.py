import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth, int height, int width, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth * height * width) {
        return;
    }

    int b = idx / (out_channels * depth * height * width);
    int c_out = (idx / (depth * height * width)) % out_channels;
    int d = (idx / (height * width)) % depth;
    int h = (idx / width) % height;
    int w = idx % width;

    float sum = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int k_d = 0; k_d < kernel_size; ++k_d) {
            for (int k_h = 0; k_h < kernel_size; ++k_h) {
                for (int k_w = 0; k_w < kernel_size; ++k_w) {
                    int i_d = d + k_d - kernel_size / 2;
                    int i_h = h + k_h - kernel_size / 2;
                    int i_w = w + k_w - kernel_size / 2;
                    if (i_d >= 0 && i_d < depth && i_h >= 0 && i_h < height && i_w >= 0 && i_w < width) {
                        sum += input[b * in_channels * depth * height * width + c_in * depth * height * width + i_d * height * width + i_h * width + i_w] *
                               weight[c_out * in_channels * kernel_size * kernel_size * kernel_size + c_in * kernel_size * kernel_size * kernel_size + k_d * kernel_size * kernel_size + k_h * kernel_size + k_w];
                    }
                }
            }
        }
    }
    output[idx] = sum;
}

torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight) {
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

    conv3d_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth, height, width, kernel_size);

    return output;
}
"""

conv3d_cpp_source = (
    "torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for 3D convolution
conv3d = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int batch_size, int out_channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth * height * width) {
        return;
    }

    int b = idx / (out_channels * depth * height * width);
    int c_out = (idx / (depth * height * width)) % out_channels;
    int d = (idx / (height * width)) % depth;
    int h = (idx / width) % height;
    int w = idx % width;

    float max_val = -std::numeric_limits<float>::infinity();
    for (int c_in = 0; c_in < out_channels; ++c_in) {
        max_val = std::max(max_val, input[b * out_channels * depth * height * width + c_in * depth * height * width + d * height * width + h * width + w]);
    }

    float sum = 0.0f;
    for (int c_in = 0; c_in < out_channels; ++c_in) {
        sum += exp(input[b * out_channels * depth * height * width + c_in * depth * height * width + d * height * width + h * width + w] - max_val);
    }

    output[idx] = exp(input[b * out_channels * depth * height * width + c_out * depth * height * width + d * height * width + h * width + w] - max_val) / sum;
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto out_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);

    auto output = torch::zeros({batch_size, out_channels, depth, height, width}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * depth * height * width + block_size - 1) / block_size;

    softmax_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, out_channels, depth, height, width);

    return output;
}
"""

softmax_cpp_source = (
    "torch::Tensor softmax_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Softmax
softmax = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Max Pooling
max_pooling_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_pooling_kernel(const float* input, float* output, int batch_size, int in_channels, int depth, int height, int width, int pool_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * in_channels * depth * height * width) {
        return;
    }

    int b = idx / (in_channels * depth * height * width);
    int c_in = (idx / (depth * height * width)) % in_channels;
    int d = (idx / (height * width)) % depth;
    int h = (idx / width) % height;
    int w = idx % width;

    int max_idx = 0;
    float max_val = -std::numeric_limits<float>::infinity();
    for (int p_d = 0; p_d < pool_size; ++p_d) {
        for (int p_h = 0; p_h < pool_size; ++p_h) {
            for (int p_w = 0; p_w < pool_size; ++p_w) {
                int i_d = d + p_d;
                int i_h = h + p_h;
                int i_w = w + p_w;
                if (i_d >= 0 && i_d < depth && i_h >= 0 && i_h < height && i_w >= 0 && i_w < width) {
                    if (input[b * in_channels * depth * height * width + c_in * depth * height * width + i_d * height * width + i_h * width + i_w] > max_val) {
                        max_val = input[b * in_channels * depth * height * width + c_in * depth * height * width + i_d * height * width + i_h * width + i_w];
                        max_idx = p_d * pool_size * pool_size + p_h * pool_size + p_w;
                    }
                }
            }
        }
    }

    output[idx] = max_val;
}

torch::Tensor max_pooling_cuda(torch::Tensor input, int pool_size) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);

    auto output_depth = depth / pool_size;
    auto output_height = height / pool_size;
    auto output_width = width / pool_size;

    auto output = torch::zeros({batch_size, in_channels, output_depth, output_height, output_width}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * in_channels * output_depth * output_height * output_width + block_size - 1) / block_size;

    max_pooling_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, depth, height, width, pool_size);

    return output;
}
"""

max_pooling_cpp_source = (
    "torch::Tensor max_pooling_cuda(torch::Tensor input, int pool_size);"
)

# Compile the inline CUDA code for Max Pooling
max_pooling = load_inline(
    name="max_pooling",
    cpp_sources=max_pooling_cpp_source,
    cuda_sources=max_pooling_source,
    functions=["max_pooling_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = conv3d
        self.pool1 = max_pooling
        self.pool2 = max_pooling

    def forward(self, x):
        x = self.conv.conv3d_cuda(x, self.weight)
        x = self.softmax.softmax_cuda(x)
        x = self.pool1.max_pooling_cuda(x, pool_kernel_size)
        x = self.pool2.max_pooling_cuda(x, pool_kernel_size)
        return x


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]