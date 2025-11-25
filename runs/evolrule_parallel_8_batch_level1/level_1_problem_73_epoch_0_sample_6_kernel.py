import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const scalar_t* input,
    const scalar_t* weights,
    scalar_t* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int depth_in,
    int height_in,
    int width_in,
    int depth_out,
    int height_out,
    int width_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth_out * height_out * width_out) return;

    int w_out = idx % width_out;
    int h_out = (idx / width_out) % height_out;
    int d_out = (idx / (width_out * height_out)) % depth_out;
    int c_out = (idx / (width_out * height_out * depth_out)) % out_channels;
    int n = idx / (out_channels * depth_out * height_out * width_out);

    scalar_t acc = 0.0;

    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group = in_channels / groups;
    int group = c_out / out_channels_per_group;
    int c_out_in_group = c_out % out_channels_per_group;

    for (int c_in_in_group = 0; c_in_in_group < in_channels_per_group; ++c_in_in_group) {
        int c_in = group * in_channels_per_group + c_in_in_group;

        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int d_in = (d_out + padding - kd - output_padding) / stride;
                    int h_in = (h_out + padding - kh - output_padding) / stride;
                    int w_in = (w_out + padding - kw - output_padding) / stride;

                    if (d_in < 0 || d_in >= depth_in || 
                        h_in < 0 || h_in >= height_in || 
                        w_in < 0 || w_in >= width_in) {
                        continue;
                    }

                    int weight_offset = 
                        (group * in_channels_per_group + c_in_in_group) * out_channels_per_group * kernel_size * kernel_size * kernel_size
                        + c_out_in_group * kernel_size * kernel_size * kernel_size
                        + kd * kernel_size * kernel_size
                        + kh * kernel_size
                        + kw;

                    scalar_t weight_val = weights[weight_offset];

                    int input_offset = 
                        n * in_channels * depth_in * height_in * width_in
                        + c_in * depth_in * height_in * width_in
                        + d_in * height_in * width_in
                        + h_in * width_in
                        + w_in;
                    scalar_t input_val = input[input_offset];

                    acc += input_val * weight_val;
                }
            }
        }
    }

    int output_offset = 
        n * out_channels * depth_out * height_out * width_out
        + c_out * depth_out * height_out * width_out
        + d_out * height_out * width_out
        + h_out * width_out
        + w_out;
    output[output_offset] = acc;
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input, 
                                   torch::Tensor weight,
                                   int stride,
                                   int padding,
                                   int output_padding,
                                   int groups) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth_in = input.size(2);
    int height_in = input.size(3);
    int width_in = input.size(4);

    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;
    int kernel_size = weight.size(2);

    int depth_out = (depth_in - 1) * stride + kernel_size + output_padding - 2 * padding;
    int height_out = (height_in - 1) * stride + kernel_size + output_padding - 2 * padding;
    int width_out = (width_in - 1) * stride + kernel_size + output_padding - 2 * padding;

    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    const int threads_per_block = 256;
    const int num_elements = batch_size * out_channels * depth_out * height_out * width_out;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose3d_cuda", ([&] {
        conv_transpose3d_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            depth_in,
            height_in,
            width_in,
            depth_out,
            height_out,
            width_out
        );
    }));

    return output;
}
"""

conv_transpose3d_cpp_source = (
    "torch::Tensor conv_transpose3d_cuda(torch::Tensor input, "
    "torch::Tensor weight, int stride, int padding, int output_padding, int groups);"
)

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.kernel_size = kernel_size
        self.bias = None

        # Initialize weights
        self.weight = nn.Parameter(torch.empty(
            (in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size)
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights and bias
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv_transpose3d.conv_transpose3d_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output