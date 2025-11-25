import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# CUDA kernel code
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * blockDim.x)

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int kernel_size,
    int stride, int padding, int output_padding, int groups,
    int output_height, int output_width, 
    const scalar_t* __restrict__ bias) {

    CUDA_KERNEL_LOOP(index, batch_size * out_channels * output_height * output_width) {
        int w_out = index % output_width;
        int h_out = (index / output_width) % output_height;
        int c_out = (index / (output_height * output_width)) % out_channels;
        int n = index / (out_channels * output_height * output_width);

        int group = c_out / (out_channels / groups);
        int c_out_in_group = c_out % (out_channels / groups);
        int in_channels_per_group = in_channels / groups;
        int c_in_start = group * in_channels_per_group;

        scalar_t sum = 0.0;

        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = (h_out + 2 * padding - kh) / stride;
                int w_in = (w_out + 2 * padding - kw) / stride;

                if (h_in < 0 || h_in >= input_height || w_in < 0 || w_in >= input_width) {
                    continue;
                }

                for (int c_in = 0; c_in < in_channels_per_group; ++c_in) {
                    int weight_offset = c_out_in_group * in_channels_per_group * kernel_size * kernel_size;
                    weight_offset += c_in * kernel_size * kernel_size;
                    weight_offset += kh * kernel_size + kw;

                    int input_offset = n * in_channels * input_height * input_width;
                    input_offset += (c_in_start + c_in) * input_height * input_width;
                    input_offset += h_in * input_width + w_in;

                    sum += weight[weight_offset] * input[input_offset];
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[c_out];
        }

        int output_offset = n * out_channels * output_height * output_width;
        output_offset += c_out * output_height * output_width;
        output_offset += h_out * output_width + w_out;
        output[output_offset] = sum;
    }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                   int stride, int padding, int output_padding, int groups) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 blocks((output.numel() + 1024 - 1) / 1024);
    dim3 threads(1024);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            input_height, input_width, kernel_size,
            stride, padding, output_padding, groups,
            output_height, output_width, 
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr
        );
    }));

    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                   int stride, int padding, int output_padding, int groups);
"""

# Compile the custom CUDA operator
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, output_padding: int = 0,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weights and bias similar to PyTorch's ConvTranspose2d
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, kernel_size, kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return conv_transpose2d.conv_transpose2d_cuda(
            x, self.weight, self.bias if self.bias is not None else torch.empty(0),
            self.stride, self.padding, self.output_padding, self.groups
        )