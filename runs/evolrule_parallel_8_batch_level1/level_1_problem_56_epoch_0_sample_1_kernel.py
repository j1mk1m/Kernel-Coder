import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void conv2d_forward_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int output_height,
    int output_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size * out_channels * output_height * output_width)
        return;

    // Compute indices
    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int c_out = (idx / (output_width * output_height)) % out_channels;
    int n = idx / (out_channels * output_height * output_width);

    scalar_t sum = 0;

    // Determine group
    int groups_out = out_channels / groups;
    int group_id = c_out / groups_out;
    int c_out_in_group = c_out % groups_out;

    int in_channels_per_group = in_channels / groups;
    for (int c_in_group = 0; c_in_group < in_channels_per_group; ++c_in_group) {
        int c_in = group_id * in_channels_per_group + c_in_group;

        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride_h - padding_h + kh * dilation_h;
                int w_in = w_out * stride_w - padding_w + kw * dilation_w;

                if (h_in >= 0 && h_in < input_height &&
                    w_in >= 0 && w_in < input_width) {

                    int weight_offset = (c_out_in_group * in_channels_per_group + c_in_group) * kernel_h * kernel_w +
                                        kh * kernel_w + kw;

                    int input_offset = n * in_channels * input_height * input_width +
                                       c_in * input_height * input_width +
                                       h_in * input_width + w_in;

                    sum += input[input_offset] * weight[weight_offset];
                }
            }
        }
    }

    int output_offset = n * out_channels * output_height * output_width +
                        c_out * output_height * output_width +
                        h_out * output_width + w_out;
    output[output_offset] = sum;
}

template <typename scalar_t>
at::Tensor conv2d_forward_cuda(
    at::Tensor input,
    at::Tensor weight,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int output_height = (input_height + 2 * padding_h - (kernel_h - 1)*dilation_h) / stride_h + 1;
    int output_width = (input_width + 2 * padding_w - (kernel_w - 1)*dilation_w) / stride_w + 1;

    auto output = at::empty({batch_size, out_channels, output_height, output_width}, input.options());

    int total_threads = batch_size * out_channels * output_height * output_width;
    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    conv2d_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups,
        output_height, output_width
    );

    return output;
}

at::Tensor conv2d_forward(
    at::Tensor input,
    at::Tensor weight,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    auto options = input.type();
    at::Tensor output;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_forward_cuda", ([&] {
        output = conv2d_forward_cuda<scalar_t>(
            input,
            weight,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups
        );
    }));
    return output;
}
"""

conv2d_cpp_source = """
at::Tensor conv2d_forward(
    at::Tensor input,
    at::Tensor weight,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
);
"""

conv2d = load_inline(
    name="conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weight parameter
        self.weight = nn.Parameter(torch.empty(
            out_channels,
            in_channels // groups,
            kernel_size[0],
            kernel_size[1]
        ))
        # Initialize weight using kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # Extract parameters
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation
        groups = self.groups

        return conv2d.conv2d_forward(
            x,
            self.weight,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups
        )

# Test code parameters (from original)
batch_size = 8
in_channels = 64
out_channels = 128
kernel_size = (5, 7)
height = 512
width = 256

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]