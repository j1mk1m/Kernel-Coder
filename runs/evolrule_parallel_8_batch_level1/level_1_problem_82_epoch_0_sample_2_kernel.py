import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_forward(
    const float* input,
    const float* kernel,
    float* output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int output_height,
    int output_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * in_channels * output_height * output_width) {
        return;
    }

    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int c = (idx / (output_height * output_width)) % in_channels;
    int n = idx / (in_channels * output_height * output_width);

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_in = h_out * stride + kh - padding;
            int w_in = w_out * stride + kw - padding;
            if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                const int input_offset = n * in_channels * input_height * input_width
                    + c * input_height * input_width
                    + h_in * input_width + w_in;
                const float input_val = input[input_offset];

                const int kernel_offset = c * kernel_size * kernel_size
                    + kh * kernel_size + kw;
                const float kernel_val = kernel[kernel_offset];

                sum += input_val * kernel_val;
            }
        }
    }

    const int output_offset = n * in_channels * output_height * output_width
        + c * output_height * output_width
        + h_out * output_width + w_out;
    output[output_offset] = sum;
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor kernel, int stride, int padding, int kernel_size) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::empty({batch_size, in_channels, output_height, output_width}, options);

    const int threads_per_block = 256;
    const int num_elements = batch_size * in_channels * output_height * output_width;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    depthwise_conv2d_forward<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        output_height,
        output_width
    );

    return output;
}
"""

depthwise_conv2d_cpp_source = "torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor kernel, int stride, int padding, int kernel_size);"

depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.bias = None

        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.depthwise_conv2d = depthwise_conv2d

    def forward(self, x):
        out = self.depthwise_conv2d.depthwise_conv2d_cuda(
            x, self.weight, self.stride, self.padding, self.kernel_size
        )
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        return out