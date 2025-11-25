import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D Transposed Convolution
transposed_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void transposed_conv_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth_in, int height_in, int width_in, int depth_out, int height_out, int width_out, int kernel_size, int stride, int padding) {
    int n = blockIdx.x / (height_out * width_out);
    int h = (blockIdx.x % (height_out * width_out)) / width_out;
    int w = blockIdx.x % width_out;
    int c = blockIdx.y;
    int d_out = blockIdx.z;

    float sum = 0.0f;
    for (int d_in = 0; d_in < depth_in; ++d_in) {
        for (int h_in = 0; h_in < height_in; ++h_in) {
            for (int w_in = 0; w_in < width_in; ++w_in) {
                int d_in_pad = d_in + padding;
                int h_in_pad = h_in + padding;
                int w_in_pad = w_in + padding;
                if (d_in_pad >= 0 && d_in_pad < depth_out && h_in_pad >= 0 && h_in_pad < height_out && w_in_pad >= 0 && w_in_pad < width_out) {
                    int i = n * in_channels * depth_in * height_in * width_in + d_in * height_in * width_in + h_in * width_in + w_in;
                    int j = c * kernel_size * kernel_size * kernel_size * in_channels * depth_in * height_in * width_in + d_in * kernel_size * kernel_size * kernel_size * in_channels * height_in * width_in + h_in * kernel_size * kernel_size * in_channels * height_in * width_in + w_in * kernel_size * kernel_size * in_channels * height_in * width_in + d_in_pad * kernel_size * kernel_size * in_channels * height_in * width_in + h_in_pad * kernel_size * kernel_size * in_channels * height_in * width_in + w_in_pad * kernel_size * kernel_size * in_channels * height_in * width_in;
                    sum += input[i] * weight[j];
                }
            }
        }
    }
    int o = n * out_channels * depth_out * height_out * width_out + d_out * height_out * width_out + h * width_out + w;
    output[o] = sum;
}

torch::Tensor transposed_conv_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(1);
    auto depth_in = input.size(2);
    auto height_in = input.size(3);
    auto width_in = input.size(4);
    auto depth_out = weight.size(2);
    auto height_out = weight.size(3);
    auto width_out = weight.size(4);
    auto kernel_size = weight.size(5);
    auto stride = 2;
    auto padding = 1;

    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * depth_out * height_out * width_out + block_size - 1) / block_size;

    transposed_conv_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth_in, height_in, width_in, depth_out, height_out, width_out, kernel_size, stride, padding);

    return output;
}
"""

transposed_conv_cpp_source = (
    "torch::Tensor transposed_conv_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for 3D Transposed Convolution
transposed_conv = load_inline(
    name="transposed_conv",
    cpp_sources=transposed_conv_cpp_source,
    cuda_sources=transposed_conv_source,
    functions=["transposed_conv_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for LogSumExp
logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void logsumexp_kernel(const float* input, float* output, int batch_size, int channels, int depth, int height, int width) {
    int n = blockIdx.x / (channels * depth * height * width);
    int c = (blockIdx.x % (channels * depth * height * width)) / (depth * height * width);
    int d = (blockIdx.x % (depth * height * width)) / (height * width);
    int h = (blockIdx.x % (height * width)) / width;
    int w = blockIdx.x % width;

    float max_val = -INFINITY;
    for (int i = 0; i < channels; ++i) {
        for (int j = 0; j < depth; ++j) {
            for (int k = 0; k < height; ++k) {
                for (int l = 0; l < width; ++l) {
                    int idx = n * channels * depth * height * width + i * depth * height * width + j * height * width + k * width + l;
                    max_val = fmax(max_val, input[idx]);
                }
            }
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < channels; ++i) {
        for (int j = 0; j < depth; ++j) {
            for (int k = 0; k < height; ++k) {
                for (int l = 0; l < width; ++l) {
                    int idx = n * channels * depth * height * width + i * depth * height * width + j * height * width + k * width + l;
                    sum += exp(input[idx] - max_val);
                }
            }
        }
    }

    int o = n * channels * depth * height * width + c * depth * height * width + d * height * width + h * width + w;
    output[o] = max_val + log(sum);
}

torch::Tensor logsumexp_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);

    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * depth * height * width + block_size - 1) / block_size;

    logsumexp_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, depth, height, width);

    return output;
}
"""

logsumexp_cpp_source = (
    "torch::Tensor logsumexp_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for LogSumExp
logsumexp = load_inline(
    name="logsumexp",
    cpp_sources=logsumexp_cpp_source,
    cuda_sources=logsumexp_source,
    functions=["logsumexp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for HardSwish
hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardswish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * (input[idx] > 0 ? (input[idx] < 6 ? 1 : 6) / 6 : 0);
    }
}

torch::Tensor hardswish_cuda(torch::Tensor input) {
    auto size = input.numel();

    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardswish_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

hardswish_cpp_source = (
    "torch::Tensor hardswish_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for HardSwish
hardswish = load_inline(
    name="hardswish",
    cpp_sources=hardswish_cpp_source,
    cuda_sources=hardswish_source,
    functions=["hardswish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Subtraction
subtraction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void subtraction_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

torch::Tensor subtraction_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    subtraction_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

subtraction_cpp_source = (
    "torch::Tensor subtraction_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for Subtraction
subtraction = load_inline(
    name="subtraction",
    cpp_sources=subtraction_cpp_source,
    cuda_sources=subtraction_source,
    functions=["subtraction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Clamp
clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void clamp_kernel(const float* input, float* output, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] < min_val ? min_val : (input[idx] > max_val ? max_val : input[idx]);
    }
}

torch::Tensor clamp_cuda(torch::Tensor input, float min_val, float max_val) {
    auto size = input.numel();

    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    clamp_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size, min_val, max_val);

    return output;
}
"""

clamp_cpp_source = (
    "torch::Tensor clamp_cuda(torch::Tensor input, float min_val, float max_val);"
)

# Compile the inline CUDA code for Clamp
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.transposed_conv = transposed_conv
        self.logsumexp = logsumexp
        self.hardswish = hardswish
        self.subtraction = subtraction
        self.clamp = clamp
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, x):
        x = self.transposed_conv.transposed_conv_cuda(x, self.weight)
        x = self.logsumexp.logsumexp_cuda(x)
        x = self.hardswish.hardswish_cuda(x)
        x = self.subtraction.subtraction_cuda(x, self.bias)
        x = self.clamp.clamp_cuda(x, min_val=-1, max_val=1)
        return x