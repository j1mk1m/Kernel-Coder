import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from torch.nn.parameter import Parameter
import math

def _pair(x):
    return (x, x) if isinstance(x, int) else x

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        output_padding = _pair(output_padding)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        self.weight = Parameter(torch.empty(
            in_channels,
            out_channels // groups,
            kernel_size[0],
            kernel_size[1]
        ))
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Load the custom CUDA kernel
        self.conv_transpose2d = self._load_custom_convtranspose()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def _load_custom_convtranspose(self):
        conv_transpose_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void conv_transpose2d_kernel(
            const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
            const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
            const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits> bias,
            torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
            const int in_channels,
            const int out_channels,
            const int kernel_h,
            const int kernel_w,
            const int stride_h,
            const int stride_w,
            const int padding_h,
            const int padding_w,
            const int output_padding_h,
            const int output_padding_w,
            const int groups
        ) {
            int batch_size = input.size(0);
            int input_h = input.size(2);
            int input_w = input.size(3);
            int output_h = output.size(2);
            int output_w = output.size(3);

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * out_channels * output_h * output_w) return;

            int b = idx / (out_channels * output_h * output_w);
            int rem = idx % (out_channels * output_h * output_w);
            int c_out = rem / (output_h * output_w);
            rem %= (output_h * output_w);
            int y_out = rem / output_w;
            int x_out = rem % output_w;

            scalar_t sum = 0.0;

            for (int g = 0; g < groups; ++g) {
                int out_per_group = out_channels / groups;
                if (c_out < g * out_per_group || c_out >= (g + 1) * out_per_group)
                    continue;

                int in_per_group = in_channels / groups;
                int in_start = g * in_per_group;

                for (int in_c = in_start; in_c < in_start + in_per_group; ++in_c) {
                    for (int ky = 0; ky < kernel_h; ++ky) {
                        for (int kx = 0; kx < kernel_w; ++kx) {
                            int y_in = (y_out + padding_h - ky + output_padding_h) / stride_h;
                            int x_in = (x_out + padding_w - kx + output_padding_w) / stride_w;

                            if (y_in < 0 || y_in >= input_h || x_in < 0 || x_in >= input_w)
                                continue;

                            int out_c_in_group = c_out - g * out_per_group;
                            scalar_t w = weight[in_c][out_c_in_group][ky][kx];
                            sum += input[b][in_c][y_in][x_in] * w;
                        }
                    }
                }
            }

            if (bias.size(0) > 0)
                sum += bias[c_out];

            output[b][c_out][y_out][x_out] = sum;
        }

        extern "C" {

        torch::Tensor custom_conv_transpose2d(
            torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor bias,
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
            int groups
        ) {
            int batch_size = input.size(0);
            int input_h = input.size(2);
            int input_w = input.size(3);

            int output_h = (input_h - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
            int output_w = (input_w - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

            auto output_options = input.options();
            auto output = torch::zeros({batch_size, out_channels, output_h, output_w}, output_options);

            int total_threads = batch_size * out_channels * output_h * output_w;
            int threads_per_block = 256;
            int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d", ([&] {
                conv_transpose2d_kernel<scalar_t><<<blocks, threads_per_block>>>(
                    input.packed_accessor<scalar_t,4>(),
                    weight.packed_accessor<scalar_t,4>(),
                    (bias.defined() ? bias.packed_accessor<scalar_t,1>() : torch::empty(0).packed_accessor<scalar_t,1>()),
                    output.packed_accessor<scalar_t,4>(),
                    in_channels, out_channels, kernel_h, kernel_w,
                    stride_h, stride_w,
                    padding_h, padding_w,
                    output_padding_h, output_padding_w,
                    groups
                );
            }));

            return output;
        }

        }
        """

        conv_transpose_mod = load_inline(
            name="custom_conv_transpose",
            cuda_sources=conv_transpose_source,
            functions=["custom_conv_transpose2d"],
            verbose=True
        )

        return conv_transpose_mod.custom_conv_transpose2d

    def forward(self, x):
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        output_padding_h, output_padding_w = self.output_padding
        groups = self.groups
        bias = self.bias if self.bias is not None else torch.empty(0, device=x.device)

        return self.conv_transpose2d(
            x,
            self.weight,
            bias,
            self.in_channels,
            self.out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_padding_h,
            output_padding_w,
            groups
        )

def get_inputs():
    batch_size = 8
    in_channels = 64
    height = 1024
    width = 1024
    x = torch.rand(batch_size, in_channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [64, 64, 3, 1, 0, 0, 1, False]