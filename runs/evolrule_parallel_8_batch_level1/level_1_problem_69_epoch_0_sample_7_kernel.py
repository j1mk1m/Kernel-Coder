import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose2d
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void custom_conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int output_h,
    int output_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_h * output_w) return;

    int n = idx / (out_channels * output_h * output_w);
    int rem = idx % (out_channels * output_h * output_w);
    int c_out = rem / (output_h * output_w);
    rem %= (output_h * output_w);
    int y_out = rem / output_w;
    int x_out = rem % output_w;

    scalar_t total = 0.0;
    for (int ky = 0; ky < kernel_h; ++ky) {
        for (int kx = 0; kx < kernel_w; ++kx) {
            int y_in = y_out - ky;
            int x_in = x_out - kx;
            if (y_in < 0 || y_in >= input_h || x_in < 0 || x_in >= input_w) continue;

            for (int c_in = 0; c_in < in_channels; ++c_in) {
                int input_offset = n * in_channels * input_h * input_w +
                                   c_in * input_h * input_w +
                                   y_in * input_w +
                                   x_in;
                scalar_t input_val = input[input_offset];

                int weight_offset = c_in * out_channels * kernel_h * kernel_w +
                                    c_out * kernel_h * kernel_w +
                                    ky * kernel_w +
                                    kx;
                scalar_t w = weight[weight_offset];

                total += input_val * w;
            }
        }
    }
    output[idx] = total;
}

std::tuple<torch::Tensor> custom_conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int output_h,
    int output_w
) {
    auto output_size = {batch_size, out_channels, output_h, output_w};
    auto output = torch::empty(output_size, input.options());

    int total_elements = batch_size * out_channels * output_h * output_w;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "custom_conv_transpose2d_forward", ([&] {
        custom_conv_transpose2d_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            output_h,
            output_w
        );
    }));

    return std::make_tuple(output);
}
"""

# Compile the CUDA kernel
conv_transpose = load_inline(
    name="custom_conv_transpose2d",
    cpp_sources=conv_transpose_source,
    functions=["custom_conv_transpose2d_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1), padding: tuple = (0, 0),
                 output_padding: tuple = (0, 0), dilation: tuple = (1, 1),
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        weight_shape = (in_channels, out_channels // groups, kernel_size[0], kernel_size[1])
        self.weight = nn.Parameter(torch.randn(weight_shape))

        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, input_h, input_w = x.size()
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        output_padding_h, output_padding_w = self.output_padding
        dilation_h, dilation_w = self.dilation

        # Compute output dimensions
        output_h = (input_h - 1) * stride_h - 2 * padding_h + \
                   dilation_h * (kernel_h - 1) + output_padding_h + 1
        output_w = (input_w - 1) * stride_w - 2 * padding_w + \
                   dilation_w * (kernel_w - 1) + output_padding_w + 1

        # Call CUDA kernel
        output = conv_transpose.custom_conv_transpose2d_forward(
            x,
            self.weight,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            output_h,
            output_w
        )[0]

        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output