import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* input, const float* kernel, const float* bias, float* output,
    int batch_size, int in_channels, int input_height, int input_width,
    int kernel_height, int kernel_width,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int output_height, int output_width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size * in_channels * output_height * output_width)
        return;

    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int c = (idx / (output_width * output_height)) % in_channels;
    int n = idx / (output_width * output_height * in_channels);

    float sum = 0.0f;

    for (int kh = 0; kh < kernel_height; ++kh) {
        for (int kw = 0; kw < kernel_width; ++kw) {
            int h_in = h_out * stride_h - padding_h + kh * dilation_h;
            int w_in = w_out * stride_w - padding_w + kw * dilation_w;

            if (h_in >= 0 && h_in < input_height &&
                w_in >= 0 && w_in < input_width) {
                int input_offset = n * in_channels * input_height * input_width +
                                   c * input_height * input_width +
                                   h_in * input_width + w_in;
                int kernel_offset = c * kernel_height * kernel_width +
                                    kh * kernel_width + kw;
                sum += input[input_offset] * kernel[kernel_offset];
            }
        }
    }

    if (bias != nullptr)
        sum += bias[c];

    int output_offset = n * in_channels * output_height * output_width +
                        c * output_height * output_width +
                        h_out * output_width + w_out;
    output[output_offset] = sum;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input, torch::Tensor kernel, torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int kernel_height = kernel.size(2);
    int kernel_width = kernel.size(3);

    // Correct output dimensions calculation
    int numerator_h = input_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1;
    int output_height = (numerator_h / stride_h) + 1;
    int numerator_w = input_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1;
    int output_width = (numerator_w / stride_w) + 1;

    auto output = torch::empty({batch_size, in_channels, output_height, output_width}, input.options());

    int threads_per_block = 256;
    int total_elements = batch_size * in_channels * output_height * output_width;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    depthwise_conv2d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        (bias.defined()) ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, input_height, input_width,
        kernel_height, kernel_width,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        output_height, output_width);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}
"""

depthwise_conv_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input, torch::Tensor kernel, torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w);
"""

depthwise_conv = load_inline(
    name='depthwise_conv',
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=['depthwise_conv2d_cuda'],
    verbose=True,
    extra_cflags=['-std=c++14'],
    extra_cuda_cflags=['-std=c++14']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int,
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0,
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        assert groups == in_channels, "Depthwise convolution requires groups equal to in_channels"
        self.in_channels = in_channels
        self.out_channels = out_channels  # Must equal in_channels for depthwise but kept for interface
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.bias = bias

        # Initialize the kernel and bias parameters
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size_h, kernel_size_w))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_buffer('bias', None)  # Pass None to the kernel

    def forward(self, x):
        return depthwise_conv.depthwise_conv2d_cuda(
            x, self.weight, self.bias,
            self.stride_h, self.stride_w,
            self.padding_h, self.padding_w,
            self.dilation_h, self.dilation_w
        )