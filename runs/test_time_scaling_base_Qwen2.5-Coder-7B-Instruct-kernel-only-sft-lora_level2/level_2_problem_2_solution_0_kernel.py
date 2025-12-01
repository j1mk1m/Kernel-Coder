import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed convolution
transposed_convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void transposed_convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height_in, int width_in, int height_out, int width_out, int kernel_size, int stride, int padding, int output_padding) {
    int b = blockIdx.y;
    int o = blockIdx.z;
    int h = blockIdx.x / width_out;
    int w = blockIdx.x % width_out;

    float sum = 0.0f;
    for (int c = 0; c < in_channels; ++c) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = h * stride - padding + kh;
                int iw = w * stride - padding + kw;
                if (ih >= 0 && ih < height_in && iw >= 0 && iw < width_in) {
                    sum += input[b * in_channels * height_in * width_in + c * height_in * width_in + ih * width_in + iw] * weight[o * in_channels * kernel_size * kernel_size + c * kernel_size * kernel_size + kh * kernel_size + kw];
                }
            }
        }
    }

    int oh = h * stride - padding + kernel_size - 1 + output_padding;
    int ow = w * stride - padding + kernel_size - 1 + output_padding;
    if (oh >= 0 && oh < height_out && ow >= 0 && ow < width_out) {
        output[b * out_channels * height_out * width_out + o * height_out * width_out + oh * width_out + ow] = sum;
    }
}

torch::Tensor transposed_convolution_cuda(torch::Tensor input, torch::Tensor weight, int batch_size, int in_channels, int out_channels, int height_in, int width_in, int height_out, int width_out, int kernel_size, int stride, int padding, int output_padding) {
    auto out = torch::zeros({batch_size, out_channels, height_out, width_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (height_out * width_out + block_size - 1) / block_size;

    dim3 grid(block_size * num_blocks, batch_size, out_channels);
    transposed_convolution_kernel<<<grid, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), batch_size, in_channels, out_channels, height_in, width_in, height_out, width_out, kernel_size, stride, padding, output_padding);

    return out;
}
"""

transposed_convolution_cpp_source = (
    "torch::Tensor transposed_convolution_cuda(torch::Tensor input, torch::Tensor weight, int batch_size, int in_channels, int out_channels, int height_in, int width_in, int height_out, int width_out, int kernel_size, int stride, int padding, int output_padding);"
)

# Compile the inline CUDA code for transposed convolution
transposed_convolution = load_inline(
    name="transposed_convolution",
    cpp_sources=transposed_convolution_cpp_source,
    cuda_sources=transposed_convolution_source,
    functions=["transposed_convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for bias addition
bias_addition_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bias_addition_kernel(const float* input, const float* bias, float* output, int batch_size, int out_channels, int height, int width) {
    int b = blockIdx.y;
    int o = blockIdx.z;
    int h = blockIdx.x / width;
    int w = blockIdx.x % width;

    output[b * out_channels * height * width + o * height * width + h * width + w] = input[b * out_channels * height * width + o * height * width + h * width + w] + bias[o];
}

torch::Tensor bias_addition_cuda(torch::Tensor input, torch::Tensor bias, int batch_size, int out_channels, int height, int width) {
    auto out = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (height * width + block_size - 1) / block_size;

    dim3 grid(block_size * num_blocks, batch_size, out_channels);
    bias_addition_kernel<<<grid, block_size>>>(input.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), batch_size, out_channels, height, width);

    return out;
}
"""

bias_addition_cpp_source = (
    "torch::Tensor bias_addition_cuda(torch::Tensor input, torch::Tensor bias, int batch_size, int out_channels, int height, int width);"
)

# Compile the inline CUDA code for bias addition
bias_addition = load_inline(
    name="bias_addition",
    cpp_sources=bias_addition_cpp_source,
    cuda_sources=bias_addition_source,
    functions=["bias_addition_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for division
division_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void division_kernel(const float* input, float* output, int batch_size, int out_channels, int height, int width, float divisor) {
    int b = blockIdx.y;
    int o = blockIdx.z;
    int h = blockIdx.x / width;
    int w = blockIdx.x % width;

    output[b * out_channels * height * width + o * height * width + h * width + w] = input[b * out_channels * height * width + o * height * width + h * width + w] / divisor;
}

torch::Tensor division_cuda(torch::Tensor input, float divisor, int batch_size, int out_channels, int height, int width) {
    auto out = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (height * width + block_size - 1) / block_size;

    dim3 grid(block_size * num_blocks, batch_size, out_channels);
    division_kernel<<<grid, block_size>>>(input.data_ptr<float>(), out.data_ptr<float>(), batch_size, out_channels, height, width, divisor);

    return out;
}
"""

division_cpp_source = (
    "torch::Tensor division_cuda(torch::Tensor input, float divisor, int batch_size, int out_channels, int height, int width);"
)

# Compile the inline CUDA code for division
division = load_inline(
    name="division",
    cpp_sources=division_cpp_source,
    cuda_sources=division_source,
    functions=["division_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.transposed_convolution = transposed_convolution
        self.bias_addition = bias_addition
        self.division = division
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.transposed_convolution.transposed_convolution_cuda(x, self.weight, batch_size, in_channels, out_channels, x.size(2), x.size(3), x.size(2) * 2, x.size(3) * 2, kernel_size, stride, padding, output_padding)
        x = self.bias_addition.bias_addition_cuda(x, self.bias, batch_size, out_channels, x.size(2), x.size(3))
        x = torch.clamp(x, min=0.0, max=1.0)
        x = x * self.scaling_factor
        x = torch.clamp(x, min=0.0, max=1.0)
        x = self.division.division_cuda(x, self.scaling_factor, batch_size, out_channels, x.size(2), x.size(3))
        return x