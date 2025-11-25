import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose3d_cuda_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_depth * output_height * output_width) return;

    int batch = idx / (out_channels * output_depth * output_height * output_width);
    int rem = idx % (out_channels * output_depth * output_height * output_width);
    int out_channel = rem / (output_depth * output_height * output_width);
    rem %= (output_depth * output_height * output_width);
    int d = rem / (output_height * output_width);
    int h = (rem % (output_height * output_width)) / output_width;
    int w = rem % output_width;

    int out_channels_per_group = out_channels / groups;
    int group = out_channel / out_channels_per_group;
    int out_channel_in_group = out_channel % out_channels_per_group;

    int in_channels_per_group = in_channels / groups;
    int in_channel_offset = group * in_channels_per_group;

    scalar_t sum = 0.0;

    for (int kd = 0; kd < kernel_d; ++kd) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int input_d = (d + padding_d - kd - output_padding_d) / stride_d;
                int input_h = (h + padding_h - kh - output_padding_h) / stride_h;
                int input_w = (w + padding_w - kw - output_padding_w) / stride_w;

                if (input_d < 0 || input_d >= input_depth || 
                    input_h < 0 || input_h >= input_height || 
                    input_w < 0 || input_w >= input_width) {
                    continue;
                }

                for (int in_c = 0; in_c < in_channels_per_group; ++in_c) {
                    int weight_offset = (group * out_channels_per_group + out_channel_in_group) * 
                        (in_channels_per_group * kernel_d * kernel_h * kernel_w) +
                        in_c * kernel_d * kernel_h * kernel_w +
                        kd * kernel_h * kernel_w + 
                        kh * kernel_w + 
                        kw;

                    scalar_t w_val = weight[weight_offset];

                    int in_offset = batch * in_channels * input_depth * input_height * input_width +
                        (in_channel_offset + in_c) * input_depth * input_height * input_width +
                        input_d * input_height * input_width +
                        input_h * input_width + 
                        input_w;

                    scalar_t in_val = input[in_offset];

                    sum += w_val * in_val;
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[out_channel];
    }

    int output_offset = batch * out_channels * output_depth * output_height * output_width +
                        out_channel * output_depth * output_height * output_width +
                        d * output_height * output_width +
                        h * output_width + 
                        w;

    output[output_offset] = sum;
}

at::Tensor conv_transpose3d_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);

    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int output_depth = (input_depth - 1) * stride_d + kernel_d - 2 * padding_d + output_padding_d;
    int output_height = (input_height - 1) * stride_h + kernel_h - 2 * padding_h + output_padding_h;
    int output_width = (input_width - 1) * stride_w + kernel_w - 2 * padding_w + output_padding_w;

    auto output = at::empty({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    int threads_per_block = 256;
    int blocks_per_grid = (output.numel() + threads_per_block - 1) / threads_per_block;

    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_cuda", ([&] {
        conv_transpose3d_cuda_kernel<scalar_t><<<blocks_per_grid, threads_per_block, 0, stream>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            bias.defined() ? bias.data<scalar_t>() : nullptr,
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_depth, input_height, input_width,
            output_depth, output_height, output_width,
            kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w,
            groups
        );
    }));

    return output;
}
"""

conv_transpose3d_cpp_source = """
at::Tensor conv_transpose3d_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups
);
"""

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding, 
            groups=groups, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose3d.weight
        bias = self.conv_transpose3d.bias if self.conv_transpose3d.bias is not None else None
        return conv_transpose3d.conv_transpose3d_cuda(
            x, weight, bias,
            self.conv_transpose3d.stride[0], self.conv_transpose3d.stride[1], self.conv_transpose3d.stride[2],
            self.conv_transpose3d.padding[0], self.conv_transpose3d.padding[1], self.conv_transpose3d.padding[2],
            self.conv_transpose3d.output_padding[0], self.conv_transpose3d.output_padding[1], self.conv_transpose3d.output_padding[2],
            self.conv_transpose3d.groups
        )