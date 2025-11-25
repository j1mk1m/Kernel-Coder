import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for transposed convolution
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int input_height,
    int input_width,
    int output_height,
    int output_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_height * output_width) {
        return;
    }

    int n = idx / (out_channels * output_height * output_width);
    int rem = idx % (out_channels * output_height * output_width);
    int c_out = rem / (output_height * output_width);
    rem %= (output_height * output_width);
    int h_out = rem / output_width;
    int w_out = rem % output_width;

    scalar_t sum = 0.0;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int kh_rot = kernel_size - 1 - kh;
            int kw_rot = kernel_size - 1 - kw;

            int h_in = (h_out + padding - kh_rot - output_padding) / stride;
            int w_in = (w_out + padding - kw_rot - output_padding) / stride;

            if (h_in < 0 || h_in >= input_height || w_in < 0 || w_in >= input_width) {
                continue;
            }

            for (int c_in = 0; c_in < in_channels; ++c_in) {
                int weight_offset = c_in * out_channels * kernel_size * kernel_size;
                weight_offset += c_out * kernel_size * kernel_size;
                weight_offset += kh * kernel_size + kw;

                int input_offset = n * in_channels * input_height * input_width;
                input_offset += c_in * input_height * input_width;
                input_offset += h_in * input_width + w_in;

                sum += weight[weight_offset] * input[input_offset];
            }
        }
    }

    int output_offset = n * out_channels * output_height * output_width;
    output_offset += c_out * output_height * output_width;
    output_offset += h_out * output_width + w_out;

    output[output_offset] = sum;
}

at::Tensor conv_transpose2d_cuda(at::Tensor input, at::Tensor weight, int stride, int padding, int output_padding, int kernel_size) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const auto output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = at::empty({batch_size, out_channels, output_height, output_width}, input.options());

    const int threads_per_block = 256;
    const int elements = output.numel();
    const int blocks_per_grid = (elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            input_height,
            input_width,
            output_height,
            output_width
        );
    }));

    return output;
}
"""

conv_transpose_cuda = load_inline(
    name="conv_transpose_cuda",
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weight parameter
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return conv_transpose_cuda.conv_transpose2d_cuda(
            x, self.weight, self.stride, self.padding, self.output_padding, self.kernel_size
        )