import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth_in,
    int height_in,
    int width_in,
    int depth_out,
    int height_out,
    int width_out,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth_out * height_out * width_out)
        return;

    int w_out = idx % width_out;
    int h_out = (idx / width_out) % height_out;
    int d_out = (idx / (width_out * height_out)) % depth_out;
    int c_out = (idx / (width_out * height_out * depth_out)) % out_channels;
    int n = idx / (out_channels * depth_out * height_out * width_out);

    float output_val = 0.0f;

    // Determine group
    int out_per_group = out_channels / groups;
    int group = c_out / out_per_group;
    int in_per_group = in_channels / groups;
    int in_channel_offset = group * in_per_group;
    int out_channel_offset = group * out_per_group;
    int c_out_in_group = c_out - out_channel_offset;

    for (int kd = 0; kd < kernel_d; ++kd) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Compute input indices
                int d_in = (d_out - kd - output_padding_d + 2 * padding_d) / stride_d;
                int h_in = (h_out - kh - output_padding_h + 2 * padding_h) / stride_h;
                int w_in = (w_out - kw - output_padding_w + 2 * padding_w) / stride_w;

                if (d_in < 0 || d_in >= depth_in ||
                    h_in < 0 || h_in >= height_in ||
                    w_in < 0 || w_in >= width_in) {
                    continue;
                }

                for (int c_in_group = 0; c_in_group < in_per_group; ++c_in_group) {
                    int c_in = in_channel_offset + c_in_group;

                    // Compute weight index
                    int weight_offset = (out_channel_offset + c_out_in_group) * in_per_group * kernel_d * kernel_h * kernel_w
                                      + c_in_group * kernel_d * kernel_h * kernel_w
                                      + kd * kernel_h * kernel_w
                                      + kh * kernel_w
                                      + kw;
                    float w = weight[weight_offset];

                    // Compute input offset
                    int input_offset = n * in_channels * depth_in * height_in * width_in
                                    + c_in * depth_in * height_in * width_in
                                    + d_in * height_in * width_in
                                    + h_in * width_in
                                    + w_in;
                    output_val += w * input[input_offset];
                }
            }
        }
    }

    if (bias != nullptr) {
        output_val += bias[c_out];
    }

    // Compute output offset
    int output_offset = n * out_channels * depth_out * height_out * width_out
                     + c_out * depth_out * height_out * width_out
                     + d_out * height_out * width_out
                     + h_out * width_out
                     + w_out;
    output[output_offset] = output_val;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::tuple<int, int, int> stride,
    std::tuple<int, int, int> padding,
    std::tuple<int, int, int> output_padding,
    int groups
) {
    int stride_d = std::get<0>(stride);
    int stride_h = std::get<1>(stride);
    int stride_w = std::get<2>(stride);

    int padding_d = std::get<0>(padding);
    int padding_h = std::get<1>(padding);
    int padding_w = std::get<2>(padding);

    int output_padding_d = std::get<0>(output_padding);
    int output_padding_h = std::get<1>(output_padding);
    int output_padding_w = std::get<2>(output_padding);

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth_in = input.size(2);
    int height_in = input.size(3);
    int width_in = input.size(4);

    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    // Compute output dimensions
    int depth_out = (depth_in - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
    int height_out = (height_in - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int width_out = (width_in - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = torch::empty({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    int total_elements = batch_size * out_channels * depth_out * height_out * width_out;
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    conv_transpose3d_forward<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        depth_in,
        height_in,
        width_in,
        depth_out,
        height_out,
        width_out,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        output_padding_d,
        output_padding_h,
        output_padding_w,
        groups
    );

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose3d_cpp = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::tuple<int, int, int> stride,
    std::tuple<int, int, int> padding,
    std::tuple<int, int, int> output_padding,
    int groups
);
"""

conv_transpose3d_cuda = load_inline(
    name='conv_transpose3d_cuda',
    cpp_sources=conv_transpose3d_cpp,
    cuda_sources=conv_transpose3d_source,
    functions=['conv_transpose3d_cuda'],
    verbose=True
)

import math

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        kernel_d, kernel_h, kernel_w = kernel_size
        weight_shape = (out_channels, in_channels // groups, kernel_d, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights and bias like PyTorch's ConvTranspose3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return conv_transpose3d_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            self.bias if self.bias is not None else torch.empty(0),
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )