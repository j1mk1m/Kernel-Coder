import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int input_height, int input_width, int output_height, int output_width, int channels, int kernel_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= output_height || col >= output_width) {
        return;
    }

    float sum = 0.0f;
    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int input_row = row + i - kernel_size / 2;
                int input_col = col + j - kernel_size / 2;
                if (input_row >= 0 && input_row < input_height && input_col >= 0 && input_col < input_width) {
                    sum += input[c * input_height * input_width + input_row * input_width + input_col] * weight[c * kernel_size * kernel_size + i * kernel_size + j];
                }
            }
        }
    }
    output[row * output_width + col] = sum;
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight) {
    int input_height = input.size(2);
    int input_width = input.size(3);
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    int channels = input.size(1);
    int kernel_size = 3;

    auto output = torch::zeros({channels, output_height, output_width}, input.options());

    const int block_size = 16;
    const int num_blocks_x = (output_width + block_size - 1) / block_size;
    const int num_blocks_y = (output_height + block_size - 1) / block_size;

    convolution_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), input_height, input_width, output_height, output_width, channels, kernel_size);

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


# Define the custom CUDA kernel for subtraction
subtraction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void subtraction_kernel(const float* input, const float* value, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] - value[0];
    }
}

torch::Tensor subtraction_cuda(torch::Tensor input, float value) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    subtraction_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), &value, output.data_ptr<float>(), size);

    return output;
}
"""

subtraction_cpp_source = (
    "torch::Tensor subtraction_cuda(torch::Tensor input, float value);"
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


# Define the custom CUDA kernel for tanh activation
tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanh(input[idx]);
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
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)

    def forward(self, x):
        x = self.conv.convolution_cuda(x, torch.randn(out_channels, kernel_size, kernel_size))
        x = self.subtract1_value.subtraction_cuda(x, self.subtract1_value)
        x = self.tanh_cuda.tanh_cuda(x)
        x = self.subtract2_value.subtraction_cuda(x, self.subtract2_value)
        x = self.avgpool(x)
        return x