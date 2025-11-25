import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int groups,
    int input_height,
    int input_width,
    int output_height,
    int output_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_height * output_width) {
        return;
    }

    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int c_out = (idx / (output_width * output_height)) % out_channels;
    int n = idx / (out_channels * output_height * output_width);

    scalar_t sum = 0;

    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group = in_channels / groups;
    int group = c_out / out_channels_per_group;
    int out_c_in_group = c_out % out_channels_per_group;
    int in_c_start = group * in_channels_per_group;
    int in_c_end = (group + 1) * in_channels_per_group;

    for (int in_c = in_c_start; in_c < in_c_end; ++in_c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = (h_out - kh + 2 * padding_h - output_padding_h) / stride_h;
                int w_in = (w_out - kw + 2 * padding_w - output_padding_w) / stride_w;

                if (h_in < 0 || h_in >= input_height || w_in < 0 || w_in >= input_width) {
                    continue;
                }

                int weight_offset = (group * out_channels_per_group + out_c_in_group) * in_channels_per_group * kernel_h * kernel_w
                    + (in_c - in_c_start) * kernel_h * kernel_w
                    + kh * kernel_w + kw;

                scalar_t w_val = weight[weight_offset];
                scalar_t in_val = input[ n * in_channels * input_height * input_width
                    + in_c * input_height * input_width
                    + h_in * input_width + w_in ];

                sum += in_val * w_val;
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    int output_offset = n * out_channels * output_height * output_width
        + c_out * output_height * output_width
        + h_out * output_width + w_out;

    output[output_offset] = sum;
}

at::Tensor conv_transpose2d_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int groups
) {
    AT_ASSERT(input.dim() == 4);
    AT_ASSERT(weight.dim() == 4);
    AT_ASSERT(input.type().scalarType() == weight.type().scalarType());
    if (bias.defined()) {
        AT_ASSERT(bias.dim() == 1);
        AT_ASSERT(bias.size(0) == weight.size(1)*groups);
    }

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(1)*groups;
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int output_height = (input_height - 1)*stride_h - 2*padding_h + kernel_h + output_padding_h;
    int output_width = (input_width -1)*stride_w - 2*padding_w + kernel_w + output_padding_w;

    at::Tensor output = at::empty({batch_size, out_channels, output_height, output_width}, input.options());

    const int threads_per_block = 256;
    const int total_elements = batch_size * out_channels * output_height * output_width;
    const int blocks_per_grid = (total_elements + threads_per_block -1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            bias.defined() ? bias.data<scalar_t>() : nullptr,
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_padding_h,
            output_padding_w,
            groups,
            input_height,
            input_width,
            output_height,
            output_width
        );
    }));

    return output;
}
"""

conv_transpose2d_h = """
#include <torch/extension.h>
at::Tensor conv_transpose2d_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int groups
);
"""

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_h,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias if bias else None

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kernel_size, kernel_size))
        if self.bias is not None:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose2d.conv_transpose2d_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.empty(0, device=x.device),
            self.stride, self.stride,
            self.padding, self.padding,
            self.output_padding, self.output_padding,
            self.groups
        )