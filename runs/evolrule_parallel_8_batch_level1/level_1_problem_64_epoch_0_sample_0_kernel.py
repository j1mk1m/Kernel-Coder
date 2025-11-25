import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void conv_transpose_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weights,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int out_channels, int kernel_size,
    int input_length, int output_length, int stride, int padding, int output_padding) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length)
        return;

    int n = idx / (out_channels * output_length);
    int rem = idx % (out_channels * output_length);
    int c_out = rem / output_length;
    int l_out = rem % output_length;

    scalar_t sum = 0.0;

    for (int k = 0; k < kernel_size; ++k) {
        int input_l = (l_out + padding - k + output_padding) / stride - padding;
        if (input_l < 0 || input_l >= input_length)
            continue;

        for (int c_in = 0; c_in < in_channels; ++c_in) {
            int weight_idx = c_in * out_channels * kernel_size + c_out * kernel_size + k;
            scalar_t w = weights[weight_idx];

            int input_offset = n * in_channels * input_length + c_in * input_length + input_l;
            scalar_t val = input[input_offset];

            sum += w * val;
        }
    }

    int output_offset = n * out_channels * output_length + c_out * output_length + l_out;
    output[output_offset] = sum;
}

torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weights,
                                 int stride, int padding, int output_padding) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int out_channels = weights.size(1);
    int kernel_size = weights.size(2);
    int output_length = (input_length - 1)*stride - 2*padding + kernel_size + output_padding;

    auto output = torch::empty({batch_size, out_channels, output_length}, 
                              input.options());

    const int threads = 256;
    const int elements = batch_size * out_channels * output_length;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose_cuda", ([&] {
        conv_transpose_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels, kernel_size,
            input_length, output_length, stride, padding, output_padding);
    }));

    return output;
}
"""

conv_transpose_header = """
#include <torch/extension.h>
torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weights,
                                 int stride, int padding, int output_padding);
"""

conv_transpose = load_inline(
    name="conv_transpose",
    cuda_headers=[conv_transpose_header],
    cuda_sources=[conv_transpose_source],
    functions=["conv_transpose_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        output = conv_transpose.conv_transpose_cuda(
            x, self.weight, self.stride, self.padding, self.output_padding)

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)

        return output