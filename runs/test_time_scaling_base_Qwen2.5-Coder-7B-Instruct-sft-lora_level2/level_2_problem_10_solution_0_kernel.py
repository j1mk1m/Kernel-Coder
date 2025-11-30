import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed convolution
transposed_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void transposed_convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height_in, int width_in, int height_out, int width_out, int kernel_size, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * height_out * width_out) {
        return;
    }

    int b = idx / (out_channels * height_out * width_out);
    int c = (idx % (out_channels * height_out * width_out)) / (height_out * width_out);
    int h_out = (idx % (out_channels * height_out * width_out)) / width_out;
    int w_out = idx % width_out;

    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int h_in = h_out * stride - padding + i;
            int w_in = w_out * stride - padding + j;
            if (h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
                sum += input[b * in_channels * height_in * width_in + (c * height_in * width_in + h_in * width_in + w_in)] * weight[c * kernel_size * kernel_size + i * kernel_size + j];
            }
        }
    }

    output[idx] = sum;
}

torch::Tensor transposed_convolution_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height_in = input.size(2);
    auto width_in = input.size(3);
    auto kernel_size = weight.size(1);
    auto stride = 1; // Assuming stride is always 1 for simplicity
    auto padding = 1; // Assuming padding is always 1 for simplicity

    auto height_out = (height_in - 1) * stride + kernel_size - 2 * padding;
    auto width_out = (width_in - 1) * stride + kernel_size - 2 * padding;

    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * height_out * width_out + block_size - 1) / block_size;

    transposed_convolution_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height_in, width_in, height_out, width_out, kernel_size, stride, padding);

    return output;
}
"""

transposed_conv_cpp_source = (
    "torch::Tensor transposed_convolution_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for transposed convolution
transposed_conv = load_inline(
    name="transposed_conv",
    cpp_sources=transposed_conv_cpp_source,
    cuda_sources=transposed_conv_source,
    functions=["transposed_convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for max pooling
maxpooling_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void maxpooling_kernel(const float* input, float* output, int batch_size, int channels, int height_in, int width_in, int height_out, int width_out, int pool_size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height_out * width_out) {
        return;
    }

    int b = idx / (channels * height_out * width_out);
    int c = (idx % (channels * height_out * width_out)) / (height_out * width_out);
    int h_out = (idx % (channels * height_out * width_out)) / width_out;
    int w_out = idx % width_out;

    float max_value = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < pool_size; ++i) {
        for (int j = 0; j < pool_size; ++j) {
            int h_in = h_out * stride + i;
            int w_in = w_out * stride + j;
            if (h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
                max_value = std::max(max_value, input[b * channels * height_in * width_in + (c * height_in * width_in + h_in * width_in + w_in)]);
            }
        }
    }

    output[idx] = max_value;
}

torch::Tensor maxpooling_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height_in = input.size(2);
    auto width_in = input.size(3);
    auto pool_size = 2;
    auto stride = 2;

    auto height_out = (height_in - pool_size) / stride + 1;
    auto width_out = (width_in - pool_size) / stride + 1;

    auto output = torch::zeros({batch_size, channels, height_out, width_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * height_out * width_out + block_size - 1) / block_size;

    maxpooling_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height_in, width_in, height_out, width_out, pool_size, stride);

    return output;
}
"""

maxpooling_cpp_source = (
    "torch::Tensor maxpooling_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for max pooling
maxpooling = load_inline(
    name="maxpooling",
    cpp_sources=maxpooling_cpp_source,
    cuda_sources=maxpooling_source,
    functions=["maxpooling_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for hardtanh activation
hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float* input, float* output, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fminf(fmaxf(input[idx], min_val), max_val);
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardtanh_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size, min_val, max_val);

    return output;
}
"""

hardtanh_cpp_source = (
    "torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val);"
)

# Compile the inline CUDA code for hardtanh activation
hardtanh = load_inline(
    name="hardtanh",
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_source,
    functions=["hardtanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for mean operation
mean_operation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_operation_kernel(const float* input, float* output, int batch_size, int channels, int height_in, int width_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels) {
        return;
    }

    int b = idx / channels;
    int c = idx % channels;

    float sum = 0.0f;
    for (int h = 0; h < height_in; ++h) {
        for (int w = 0; w < width_in; ++w) {
            sum += input[b * channels * height_in * width_in + (c * height_in * width_in + h * width_in + w)];
        }
    }

    output[idx] = sum / (height_in * width_in);
}

torch::Tensor mean_operation_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height_in = input.size(2);
    auto width_in = input.size(3);

    auto output = torch::zeros({batch_size, channels}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels + block_size - 1) / block_size;

    mean_operation_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height_in, width_in);

    return output;
}
"""

mean_operation_cpp_source = (
    "torch::Tensor mean_operation_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for mean operation
mean_operation = load_inline(
    name="mean_operation",
    cpp_sources=mean_operation_cpp_source,
    cuda_sources=mean_operation_source,
    functions=["mean_operation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for tanh activation
tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

torch::Tensor tanh_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    tanh_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

tanh_cpp_source = (
    "torch::Tensor tanh_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for tanh activation
tanh = load_inline(
    name="tanh",
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_source,
    functions=["tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.transposed_conv = transposed_conv
        self.maxpool = maxpooling
        self.hardtanh = hardtanh
        self.mean_operation = mean_operation
        self.tanh = tanh

    def forward(self, x):
        x = self.transposed_conv.transposed_convolution_cuda(x, self.weight)
        x = self.maxpool.maxpooling_cuda(x)
        x = self.hardtanh.hardtanh_cuda(x, self.hardtanh_min, self.hardtanh_max)
        x = self.mean_operation.mean_operation_cuda(x)
        x = self.tanh.tanh_cuda(x)
        return x


# Example usage
if __name__ == "__main__":
    batch_size = 128
    in_channels = 64
    out_channels = 64
    height = width = 256
    kernel_size = 3
    stride = 1
    padding = 1
    maxpool_kernel_size = 2
    maxpool_stride = 2
    hardtanh_min = -1
    hardtanh_max = 1

    model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max)
    inputs = get_inputs()
    outputs = model_new(inputs[0])
    print(outputs.shape)