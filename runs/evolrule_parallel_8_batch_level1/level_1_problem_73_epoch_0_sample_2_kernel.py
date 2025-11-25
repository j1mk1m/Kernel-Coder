import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv3d_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv3d_transpose_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weights,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth, int input_height, int input_width,
    int kernel_depth, int kernel_height, int kernel_width,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
    int output_padding_depth, int output_padding_height, int output_padding_width,
    int groups,
    int output_depth, int output_height, int output_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_depth * output_height * output_width)
        return;

    int b = idx / (out_channels * output_depth * output_height * output_width);
    int remaining = idx % (out_channels * output_depth * output_height * output_width);
    int c_out = remaining / (output_depth * output_height * output_width);
    remaining %= (output_depth * output_height * output_width);
    int d_out = remaining / (output_height * output_width);
    remaining %= (output_height * output_width);
    int h_out = remaining / output_width;
    int w_out = remaining % output_width;

    int in_per_group = in_channels / groups;
    int out_per_group = out_channels / groups;
    int group = c_out / out_per_group;
    int c_out_in_group = c_out % out_per_group;

    scalar_t val = 0.0;

    for (int kd = 0; kd < kernel_depth; ++kd) {
        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                int d_in = (d_out + padding_depth - kd - output_padding_depth) / stride_depth;
                int h_in = (h_out + padding_height - kh - output_padding_height) / stride_height;
                int w_in = (w_out + padding_width - kw - output_padding_width) / stride_width;

                if (d_in < 0 || d_in >= input_depth ||
                    h_in < 0 || h_in >= input_height ||
                    w_in < 0 || w_in >= input_width) {
                    continue;
                }

                for (int c_in = 0; c_in < in_per_group; ++c_in) {
                    int c_in_total = group * in_per_group + c_in;

                    int weight_offset = group * out_per_group * in_per_group * kernel_depth * kernel_height * kernel_width
                        + c_out_in_group * in_per_group * kernel_depth * kernel_height * kernel_width
                        + c_in * kernel_depth * kernel_height * kernel_width
                        + kd * kernel_height * kernel_width
                        + kh * kernel_width
                        + kw;

                    scalar_t weight = weights[weight_offset];

                    int input_offset = b * in_channels * input_depth * input_height * input_width
                        + c_in_total * input_depth * input_height * input_width
                        + d_in * input_height * input_width
                        + h_in * input_width
                        + w_in;

                    val += input[input_offset] * weight;
                }
            }
        }
    }

    int output_offset = b * out_channels * output_depth * output_height * output_width
        + c_out * output_depth * output_height * output_width
        + d_out * output_height * output_width
        + h_out * output_width
        + w_out;

    output[output_offset] = val;
}

torch::Tensor conv3d_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
    int output_padding_depth, int output_padding_height, int output_padding_width,
    int groups
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_depth = input.size(2);
    auto input_height = input.size(3);
    auto input_width = input.size(4);

    auto out_channels = weights.size(0);
    auto kernel_depth = weights.size(2);
    auto kernel_height = weights.size(3);
    auto kernel_width = weights.size(4);

    auto output_depth = (input_depth - 1) * stride_depth + kernel_depth - 2 * padding_depth + output_padding_depth;
    auto output_height = (input_height - 1) * stride_height + kernel_height - 2 * padding_height + output_padding_height;
    auto output_width = (input_width - 1) * stride_width + kernel_width - 2 * padding_width + output_padding_width;

    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    int threads_per_block = 256;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_transpose_cuda", ([&] {
        conv3d_transpose_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_depth, input_height, input_width,
            kernel_depth, kernel_height, kernel_width,
            stride_depth, stride_height, stride_width,
            padding_depth, padding_height, padding_width,
            output_padding_depth, output_padding_height, output_padding_width,
            groups,
            output_depth, output_height, output_width
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv3d_transpose = load_inline(
    name="conv3d_transpose",
    cuda_sources=conv3d_transpose_source,
    functions=["conv3d_transpose_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size, kernel_size)
        self.stride = (stride, stride, stride)
        self.padding = (padding, padding, padding)
        self.output_padding = (output_padding, output_padding, output_padding)
        self.groups = groups
        self.bias = bias

        # Initialize weights similar to PyTorch's default
        self.weight = nn.Parameter(torch.empty(
            out_channels,
            in_channels // groups,
            kernel_size, kernel_size, kernel_size
        ))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Assign CUDA function
        self._conv3d_transpose_cuda = conv3d_transpose.conv3d_transpose_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        output_padding_d, output_padding_h, output_padding_w = self.output_padding

        return self._conv3d_transpose_cuda(
            x,
            self.weight,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w,
            self.groups
        )