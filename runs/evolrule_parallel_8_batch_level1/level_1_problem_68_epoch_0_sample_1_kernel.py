import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

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
    int depth_in,
    int height_in,
    int width_in,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_depth,
    int stride_height,
    int stride_width,
    int padding_depth,
    int padding_height,
    int padding_width,
    int output_padding_depth,
    int output_padding_height,
    int output_padding_width,
    int groups) {

    int depth_out = (depth_in - 1) * stride_depth - 2 * padding_depth + kernel_depth + output_padding_depth;
    int height_out = (height_in - 1) * stride_height - 2 * padding_height + kernel_height + output_padding_height;
    int width_out = (width_in - 1) * stride_width - 2 * padding_width + kernel_width + output_padding_width;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int total_idx = idx; total_idx < batch_size * out_channels * depth_out * height_out * width_out; total_idx += blockDim.x * gridDim.x) {
        int w_out = total_idx % width_out;
        int h_out = (total_idx / width_out) % height_out;
        int d_out = (total_idx / (width_out * height_out)) % depth_out;
        int o_c = (total_idx / (width_out * height_out * depth_out)) % out_channels;
        int n = total_idx / (out_channels * depth_out * height_out * width_out);

        scalar_t acc = 0.0;

        for (int i_c = 0; i_c < in_channels; ++i_c) {
            for (int k_d = 0; k_d < kernel_depth; ++k_d) {
                for (int k_h = 0; k_h < kernel_height; ++k_h) {
                    for (int k_w = 0; k_w < kernel_width; ++k_w) {
                        int d_in = (d_out - k_d + padding_depth) / stride_depth - output_padding_depth;
                        int h_in = (h_out - k_h + padding_height) / stride_height - output_padding_height;
                        int w_in = (w_out - k_w + padding_width) / stride_width - output_padding_width;

                        if (d_in >= 0 && d_in < depth_in &&
                            h_in >= 0 && h_in < height_in &&
                            w_in >= 0 && w_in < width_in) {
                            int weight_offset = i_c * out_channels * kernel_depth * kernel_height * kernel_width +
                                                o_c * kernel_depth * kernel_height * kernel_width +
                                                k_d * kernel_height * kernel_width +
                                                k_h * kernel_width +
                                                k_w;

                            int input_offset = n * in_channels * depth_in * height_in * width_in +
                                               i_c * depth_in * height_in * width_in +
                                               d_in * height_in * width_in +
                                               h_in * width_in +
                                               w_in;

                            acc += input[input_offset] * weight[weight_offset];
                        }
                    }
                }
            }
        }

        int output_offset = n * out_channels * depth_out * height_out * width_out +
                           o_c * depth_out * height_out * width_out +
                           d_out * height_out * width_out +
                           h_out * width_out +
                           w_out;

        output[output_offset] = acc;
    }
}

std::tuple<torch::Tensor> conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_depth,
    int stride_height,
    int stride_width,
    int padding_depth,
    int padding_height,
    int padding_width,
    int output_padding_depth,
    int output_padding_height,
    int output_padding_width,
    int groups) {

    auto output_device = input.device();
    TORCH_CHECK(weight.device() == output_device, "Input and weight must be on the same device");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth_in = input.size(2);
    int height_in = input.size(3);
    int width_in = input.size(4);

    int in_channels_weight = weight.size(0);
    int out_channels_per_group = weight.size(1);
    int kernel_depth = weight.size(2);
    int kernel_height = weight.size(3);
    int kernel_width = weight.size(4);

    int out_channels = out_channels_per_group * groups;

    TORCH_CHECK(in_channels == in_channels_weight * groups, "Input channels must match weight dimensions");

    int depth_out = (depth_in - 1) * stride_depth - 2 * padding_depth + kernel_depth + output_padding_depth;
    int height_out = (height_in - 1) * stride_height - 2 * padding_height + kernel_height + output_padding_height;
    int width_out = (width_in - 1) * stride_width - 2 * padding_width + kernel_width + output_padding_width;

    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    int threads_per_block = 256;
    int blocks_per_grid = (output.numel() + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose3d_forward", ([&] {
        conv_transpose3d_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            depth_in,
            height_in,
            width_in,
            kernel_depth,
            kernel_height,
            kernel_width,
            stride_depth,
            stride_height,
            stride_width,
            padding_depth,
            padding_height,
            padding_width,
            output_padding_depth,
            output_padding_height,
            output_padding_width,
            groups
        );
    }));

    return std::make_tuple(output);
}
"""

conv_transpose3d_cpp_source = """
#include <torch/extension.h>

std::tuple<torch::Tensor> conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_depth,
    int stride_height,
    int stride_width,
    int padding_depth,
    int padding_height,
    int padding_width,
    int output_padding_depth,
    int output_padding_height,
    int output_padding_width,
    int groups);
"""

module = load_inline(
    name="conv_transpose3d",
    cpp_sources=[conv_transpose3d_cpp_source],
    cuda_sources=[conv_transpose3d_source],
    functions=["conv_transpose3d_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), output_padding=(0, 0, 0), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        kernel_depth, kernel_width, kernel_height = kernel_size
        assert kernel_width == kernel_height, "kernel_width must equal kernel_height"

        # Initialize weight and bias similar to PyTorch's ConvTranspose3d
        self.weight = nn.Parameter(torch.empty(
            in_channels,
            out_channels // groups,
            kernel_depth,
            kernel_height,
            kernel_width
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Unpack parameters
        stride_depth, stride_width, stride_height = self.stride
        padding_depth, padding_width, padding_height = self.padding
        output_padding_depth, output_padding_width, output_padding_height = self.output_padding

        # Call the custom CUDA kernel
        output = module.conv_transpose3d_forward(
            x,
            self.weight,
            stride_depth,
            stride_height,
            stride_width,
            padding_depth,
            padding_height,
            padding_width,
            output_padding_depth,
            output_padding_height,
            output_padding_width,
            self.groups
        )[0]

        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)

        return output