import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int input_height, int input_width, int input_channels, int kernel_size) {
    int batch_idx = blockIdx.y;
    int channel_idx = blockIdx.z;
    int output_height = gridDim.y;
    int output_width = gridDim.z;

    int output_idx = batch_idx * input_channels * output_height * output_width + channel_idx * output_height * output_width + blockIdx.x * blockDim.x + threadIdx.x;
    int input_idx = batch_idx * input_channels * input_height * input_width + channel_idx * input_height * input_width + blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            sum += input[input_idx + i * input_width + j] * weight[channel_idx * kernel_size * kernel_size + i * kernel_size + j];
        }
    }

    output[output_idx] = sum;
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size) {
    auto output_height = (input.size(2) - kernel_size + 1);
    auto output_width = (input.size(3) - kernel_size + 1);
    auto output = torch::zeros({input.size(0), weight.size(0), output_height, output_width}, input.options());

    const int block_size = 256;
    const int num_blocks = (output_height * output_width + block_size - 1) / block_size;

    convolution_kernel<<<dim3(num_blocks, output_height, output_width), dim3(block_size)>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), input.size(2), input.size(3), input.size(1), kernel_size);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size);"
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


# Define the custom CUDA kernel for subtraction
subtraction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void subtraction_kernel(const float* input, const float* subtract_value, float* output, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels * height * width) {
        output[idx] = input[idx] - subtract_value;
    }
}

torch::Tensor subtraction_cuda(torch::Tensor input, float subtract_value) {
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (input.numel() + block_size - 1) / block_size;

    subtraction_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), &subtract_value, output.data_ptr<float>(), input.size(0), input.size(1), input.size(2), input.size(3));

    return output;
}
"""

subtraction_cpp_source = (
    "torch::Tensor subtraction_cuda(torch::Tensor input, float subtract_value);"
)

# Compile the inline CUDA code for subtraction
subtraction = load_inline(
    name="subtraction",
    cpp_sources=subtraction_cpp_source,
    cuda_sources=subtraction_source,
    functions=["subtraction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for max pooling
max_pooling_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_pooling_kernel(const float* input, float* output, int input_height, int input_width, int pool_size) {
    int batch_idx = blockIdx.y;
    int channel_idx = blockIdx.z;
    int output_height = gridDim.y;
    int output_width = gridDim.z;

    int output_idx = batch_idx * input.size(1) * output_height * output_width + channel_idx * output_height * output_width + blockIdx.x * blockDim.x + threadIdx.x;
    int input_idx = batch_idx * input.size(1) * input_height * input_width + channel_idx * input_height * input_width + blockIdx.x * blockDim.x + threadIdx.x;

    int max_val = -999999999;
    for (int i = 0; i < pool_size; ++i) {
        for (int j = 0; j < pool_size; ++j) {
            int val = input[input_idx + i * input_width + j];
            if (val > max_val) {
                max_val = val;
            }
        }
    }

    output[output_idx] = max_val;
}

torch::Tensor max_pooling_cuda(torch::Tensor input, int pool_size) {
    auto output_height = (input.size(2) - pool_size + 1);
    auto output_width = (input.size(3) - pool_size + 1);
    auto output = torch::zeros({input.size(0), input.size(1), output_height, output_width}, input.options());

    const int block_size = 256;
    const int num_blocks = (output_height * output_width + block_size - 1) / block_size;

    max_pooling_kernel<<<dim3(num_blocks, output_height, output_width), dim3(block_size)>>>(input.data_ptr<float>(), output.data_ptr<float>(), input.size(2), input.size(3), pool_size);

    return output;
}
"""

max_pooling_cpp_source = (
    "torch::Tensor max_pooling_cuda(torch::Tensor input, int pool_size);"
)

# Compile the inline CUDA code for max pooling
max_pooling = load_inline(
    name="max_pooling",
    cpp_sources=max_pooling_cpp_source,
    cuda_sources=max_pooling_source,
    functions=["max_pooling_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for mish activation
mish_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mish_activation_kernel(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = input[idx] * tanh(log(1 + exp(input[idx])));
    }
}

torch::Tensor mish_activation_cuda(torch::Tensor input) {
    auto output = input.clone();

    const int block_size = 256;
    const int num_blocks = (input.numel() + block_size - 1) / block_size;

    mish_activation_kernel<<<num_blocks, block_size>>>(output.data_ptr<float>(), input.numel());

    return output;
}
"""

mish_activation_cpp_source = (
    "torch::Tensor mish_activation_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for mish activation
mish_activation = load_inline(
    name="mish_activation",
    cpp_sources=mish_activation_cpp_source,
    cuda_sources=mish_activation_source,
    functions=["mish_activation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.subtract_value = subtraction
        self.pool = max_pooling
        self.mish_activation = mish_activation

    def forward(self, x):
        x = self.conv.convolution_cuda(x, self.weight, kernel_size)
        x = self.subtract_value.subtraction_cuda(x, self.subtract_value)
        x = torch.nn.functional.hardswish(x)
        x = self.pool.max_pooling_cuda(x, pool_kernel_size)
        x = self.mish_activation.mish_activation_cuda(x)
        return x