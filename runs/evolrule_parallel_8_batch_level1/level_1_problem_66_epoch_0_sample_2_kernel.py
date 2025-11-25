import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

custom_conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void custom_conv3d(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth, int input_height, int input_width,
    int kernel_depth, int kernel_height, int kernel_width,
    int output_depth, int output_height, int output_width,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_depth * output_height * output_width) {
        return;
    }

    int n = idx / (out_channels * output_depth * output_height * output_width);
    int residual = idx % (out_channels * output_depth * output_height * output_width);
    int c_out = residual / (output_depth * output_height * output_width);
    residual %= (output_depth * output_height * output_width);
    int od = residual / (output_height * output_width);
    residual %= (output_height * output_width);
    int oh = residual / output_width;
    int ow = residual % output_width;

    scalar_t acc = 0.0;

    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    int group_id = c_out / out_channels_per_group;

    for (int c_in = group_id * in_channels_per_group; c_in < (group_id + 1)*in_channels_per_group; ++c_in) {
        int c_in_group = c_in - group_id * in_channels_per_group;
        for (int kd = 0; kd < kernel_depth; ++kd) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int id = od * stride_d - padding_d + dilation_d * kd;
                    int ih = oh * stride_h - padding_h + dilation_h * kh;
                    int iw = ow * stride_w - padding_w + dilation_w * kw;

                    if (id < 0 || id >= input_depth || ih < 0 || ih >= input_height || iw < 0 || iw >= input_width) {
                        continue;
                    }

                    int input_offset = n * in_channels * input_depth * input_height * input_width +
                                       c_in * input_depth * input_height * input_width +
                                       id * input_height * input_width +
                                       ih * input_width +
                                       iw;

                    int weight_offset = c_out * (in_channels_per_group * kernel_depth * kernel_height * kernel_width) +
                                        c_in_group * (kernel_depth * kernel_height * kernel_width) +
                                        kd * (kernel_height * kernel_width) +
                                        kh * kernel_width +
                                        kw;

                    acc += input[input_offset] * weight[weight_offset];
                }
            }
        }
    }

    if (bias != nullptr) {
        acc += bias[c_out];
    }

    int output_offset = n * out_channels * output_depth * output_height * output_width +
                        c_out * output_depth * output_height * output_width +
                        od * output_height * output_width +
                        oh * output_width +
                        ow;

    output[output_offset] = acc;
}

torch::Tensor custom_conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth, int input_height, int input_width,
    int kernel_depth, int kernel_height, int kernel_width,
    int output_depth, int output_height, int output_width,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "custom_conv3d", ([&] {
        custom_conv3d<scalar_t><<<grid_size, block_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_depth, input_height, input_width,
            kernel_depth, kernel_height, kernel_width,
            output_depth, output_height, output_width,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            dilation_d, dilation_h, dilation_w,
            groups
        );
    }));

    return output;
}
"""

custom_conv3d_cuda = load_inline(
    name='custom_conv3d',
    cpp_sources='',
    cuda_sources=custom_conv3d_source,
    functions=['custom_conv3d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        kernel_d, kernel_h, kernel_w = kernel_size
        weight_shape = (out_channels, in_channels // groups, kernel_d, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.randn(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        batch_size, in_channels, input_depth, input_height, input_width = x.size()
        assert in_channels == self.in_channels, "Input channels must match"

        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        dilation_d, dilation_h, dilation_w = self.dilation
        groups = self.groups
        bias = self.bias if self.bias is not None else torch.empty(0)

        output_depth = (input_depth + 2 * padding_d - dilation_d * (kernel_d - 1) - 1) // stride_d + 1
        output_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        output_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

        padding = (padding_w, padding_w, padding_h, padding_h, padding_d, padding_d)
        x_padded = F.pad(x, padding)

        padded_input_depth = input_depth + 2 * padding_d
        padded_input_height = input_height + 2 * padding_h
        padded_input_width = input_width + 2 * padding_w

        output = custom_conv3d_cuda.custom_conv3d_cuda(
            x_padded,
            self.weight,
            bias,
            batch_size,
            in_channels,
            self.out_channels,
            padded_input_depth, padded_input_height, padded_input_width,
            kernel_d, kernel_h, kernel_w,
            output_depth, output_height, output_width,
            stride_d, stride_h, stride_w,
            0, 0, 0,
            dilation_d, dilation_h, dilation_w,
            groups
        )

        return output