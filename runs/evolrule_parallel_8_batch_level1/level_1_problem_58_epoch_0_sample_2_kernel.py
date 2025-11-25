import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose3d_forward(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth_in, int height_in, int width_in,
    int depth_out, int height_out, int width_out,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups,
    bool has_bias,
    const scalar_t* __restrict__ bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size * out_channels * depth_out * height_out * width_out) {
        return;
    }

    int batch = idx / (out_channels * depth_out * height_out * width_out);
    int remaining = idx % (out_channels * depth_out * height_out * width_out);
    int oc = remaining / (depth_out * height_out * width_out);
    remaining %= (depth_out * height_out * width_out);
    int d_out = remaining / (height_out * width_out);
    remaining %= (height_out * width_out);
    int h_out = remaining / width_out;
    int w_out = remaining % width_out;

    scalar_t acc = 0.0;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int d_in = (d_out - kd + padding_d + output_padding_d) / stride_d;
                    int h_in = (h_out - kh + padding_h + output_padding_h) / stride_h;
                    int w_in = (w_out - kw + padding_w + output_padding_w) / stride_w;

                    if (d_in >= 0 && d_in < depth_in &&
                        h_in >= 0 && h_in < height_in &&
                        w_in >= 0 && w_in < width_in) {
                        int input_offset = batch * in_channels * depth_in * height_in * width_in +
                            ic * depth_in * height_in * width_in +
                            d_in * height_in * width_in +
                            h_in * width_in +
                            w_in;
                        scalar_t input_val = input[input_offset];

                        int weight_offset = ic * out_channels * kernel_d * kernel_h * kernel_w +
                            oc * kernel_d * kernel_h * kernel_w +
                            kd * kernel_h * kernel_w +
                            kh * kernel_w +
                            kw;
                        scalar_t weight_val = weight[weight_offset];

                        acc += input_val * weight_val;
                    }
                }
            }
        }
    }

    if (has_bias) {
        acc += bias[oc];
    }

    int output_offset = batch * out_channels * depth_out * height_out * width_out +
        oc * depth_out * height_out * width_out +
        d_out * height_out * width_out +
        h_out * width_out +
        w_out;
    output[output_offset] = acc;
}

std::tuple<torch::Tensor> conv_transpose3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups,
    bool has_bias
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth_in = input.size(2);
    auto height_in = input.size(3);
    auto width_in = input.size(4);

    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);

    auto depth_out = (depth_in - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
    auto height_out = (height_in - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    auto width_out = (width_in - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = torch::zeros({batch_size, weight.size(1), depth_out, height_out, width_out}, input.options());

    auto num_elements = output.numel();
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_forward", ([&] {
        conv_transpose3d_forward<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            weight.size(1),
            depth_in, height_in, width_in,
            depth_out, height_out, width_out,
            kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w,
            groups,
            has_bias,
            bias.data_ptr<scalar_t>()
        );
    }));

    return std::make_tuple(output);
}
"""

conv_transpose3d_header = """
#include <torch/extension.h>

std::tuple<torch::Tensor> conv_transpose3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups,
    bool has_bias
);
"""

conv_transpose3d_cuda = load_inline(
    name="conv_transpose3d_cuda",
    cpp_sources=conv_transpose3d_header,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, 
                 bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.has_bias = bias

        kernel_d, kernel_h, kernel_w = kernel_size
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_d, kernel_h, kernel_w))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        op_d, op_h, op_w = self.output_padding

        output = conv_transpose3d_cuda.conv_transpose3d_forward_cuda(
            x,
            self.weight,
            self.bias if self.has_bias else x.new_zeros(0),
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            op_d, op_h, op_w,
            self.groups,
            self.has_bias
        )[0]

        return output