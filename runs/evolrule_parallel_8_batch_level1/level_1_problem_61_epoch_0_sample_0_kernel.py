import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the CUDA kernel for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
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
    idx /= width_out;
    int h_out = idx % height_out;
    idx /= height_out;
    int d_out = idx % depth_out;
    idx /= depth_out;
    int c_out = idx % out_channels;
    idx /= out_channels;
    int n = idx;

    scalar_t val = 0.0;

    // Compute group and within-group indices
    int output_channels_per_group = out_channels / groups;
    int g = c_out / output_channels_per_group;
    int c_out_in_group = c_out % output_channels_per_group;

    int input_channels_per_group = in_channels / groups;
    int c_in_base = g * input_channels_per_group;

    for (int local_c_in = 0; local_c_in < input_channels_per_group; ++local_c_in) {
        int c_in = c_in_base + local_c_in;

        for (int k_d = 0; k_d < kernel_size; ++k_d) {
            for (int k_h = 0; k_h < kernel_size; ++k_h) {
                for (int k_w = 0; k_w < kernel_size; ++k_w) {
                    int d_in = (d_out + 2 * padding - k_d) / stride - output_padding;
                    int h_in = (h_out + 2 * padding - k_h) / stride - output_padding;
                    int w_in = (w_out + 2 * padding - k_w) / stride - output_padding;

                    if (d_in < 0 || d_in >= depth_in || h_in < 0 || h_in >= height_in || w_in < 0 || w_in >= width_in) {
                        continue;
                    }

                    // Compute weight index
                    int weight_offset = c_in * output_channels_per_group * kernel_size * kernel_size * kernel_size
                                       + c_out_in_group * kernel_size * kernel_size * kernel_size
                                       + k_d * kernel_size * kernel_size
                                       + k_h * kernel_size
                                       + k_w;

                    val += input[ n * in_channels * depth_in * height_in * width_in
                                + c_in * depth_in * height_in * width_in
                                + d_in * height_in * width_in
                                + h_in * width_in
                                + w_in ]
                        * weight[weight_offset];
                }
            }
        }
    }

    // Add bias
    if (bias != nullptr) {
        val += bias[c_out];
    }

    // Write output
    output[ n * out_channels * depth_out * height_out * width_out
           + c_out * depth_out * height_out * width_out
           + d_out * height_out * width_out
           + h_out * width_out
           + w_out ] = val;
}

// Define the launcher function
std::tuple<torch::Tensor> conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int depth_in = input.size(2);
    const int height_in = input.size(3);
    const int width_in = input.size(4);

    const int kernel_size = weight.size(2);
    const int out_channels = weight.size(1) * groups;

    const int depth_out = (depth_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int height_out = (height_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int width_out = (width_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::empty({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    const int threads_per_block = 256;
    const int num_elements = output.numel();
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose3d_forward", ([&] {
        conv_transpose3d_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
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

    return std::make_tuple(output);
}
"""

cpp_source = """
#include <torch/extension.h>
"""

# Compile the CUDA code
conv_transpose3d_cuda = load_inline(
    name="conv_transpose3d_cuda",
    cpp_sources=cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Initialize weight
        self.weight = nn.Parameter(torch.empty(
            (in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size)
        ))

        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters like PyTorch's ConvTranspose3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        bias_tensor = self.bias if self.bias is not None else torch.empty(0, device=x.device)
        output_tuple = conv_transpose3d_cuda.conv_transpose3d_forward(
            x,
            self.weight,
            bias_tensor,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )
        return output_tuple[0]