import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose3d_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w,
    int output_depth,
    int output_height,
    int output_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size * out_channels * output_depth * output_height * output_width) {
        return;
    }

    int batch = idx / (out_channels * output_depth * output_height * output_width);
    int remaining = idx % (out_channels * output_depth * output_height * output_width);
    int out_ch = remaining / (output_depth * output_height * output_width);
    remaining %= (output_depth * output_height * output_width);
    int d_out = remaining / (output_height * output_width);
    int h_out = (remaining % (output_height * output_width)) / output_width;
    int w_out = (remaining % (output_height * output_width)) % output_width;

    scalar_t sum = 0.0;

    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int k_d = 0; k_d < kernel_depth; ++k_d) {
            for (int k_h = 0; k_h < kernel_height; ++k_h) {
                for (int k_w = 0; k_w < kernel_width; ++k_w) {
                    int d_in = (d_out + padding_d - k_d - output_padding_d) / stride_d;
                    int h_in = (h_out + padding_h - k_h - output_padding_h) / stride_h;
                    int w_in = (w_out + padding_w - k_w - output_padding_w) / stride_w;

                    if (d_in >= 0 && d_in < input_depth &&
                        h_in >= 0 && h_in < input_height &&
                        w_in >= 0 && w_in < input_width) {
                        int input_offset = batch * in_channels * input_depth * input_height * input_width
                                        + in_ch * input_depth * input_height * input_width
                                        + d_in * input_height * input_width
                                        + h_in * input_width
                                        + w_in;
                        int weight_offset = out_ch * in_channels * kernel_depth * kernel_height * kernel_width
                                        + in_ch * kernel_depth * kernel_height * kernel_width
                                        + k_d * kernel_height * kernel_width
                                        + k_h * kernel_width
                                        + k_w;
                        sum += input[input_offset] * weight[weight_offset];
                    }
                }
            }
        }
    }

    int output_offset = batch * out_channels * output_depth * output_height * output_width
                     + out_ch * output_depth * output_height * output_width
                     + d_out * output_height * output_width
                     + h_out * output_width
                     + w_out;
    output[output_offset] = sum;
}

torch::Tensor conv_transpose3d_forward(torch::Tensor input,
                                      torch::Tensor weight,
                                      int stride_d, int stride_h, int stride_w,
                                      int padding_d, int padding_h, int padding_w,
                                      int output_padding_d, int output_padding_h, int output_padding_w) {
    auto device = input.device();
    TORCH_CHECK(weight.device() == device, "Input and weight must be on the same device");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);

    int out_channels = weight.size(0);
    int kernel_depth = weight.size(2);
    int kernel_height = weight.size(3);
    int kernel_width = weight.size(4);

    int output_depth = (input_depth - 1) * stride_d - 2 * padding_d + kernel_depth + output_padding_d;
    int output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h;
    int output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w;

    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width},
                              input.options());

    int num_elements = batch_size * out_channels * output_depth * output_height * output_width;

    const int threads_per_block = 256;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_forward", ([&] {
        conv_transpose3d_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, in_channels, out_channels,
            input_depth, input_height, input_width,
            kernel_depth, kernel_height, kernel_width,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w,
            output_depth, output_height, output_width
        );
    }));

    return output;
}
"""

conv_transpose3d_cpp_source = (
    "torch::Tensor conv_transpose3d_forward(torch::Tensor input, torch::Tensor weight, int stride_d, int stride_h, int stride_w, int padding_d, int padding_h, int padding_w, int output_padding_d, int output_padding_h, int output_padding_w);"
)

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_forward"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        self.stride = (stride, stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding, padding) if isinstance(padding, int) else padding
        self.output_padding = (output_padding, output_padding, output_padding) if isinstance(output_padding, int) else output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weight
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        # Load CUDA extension
        self.cuda_conv_transpose3d = conv_transpose3d

    def forward(self, x):
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        output_padding_d, output_padding_h, output_padding_w = self.output_padding

        output = self.cuda_conv_transpose3d.conv_transpose3d_forward(
            x,
            self.weight,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w
        )

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)

        return output