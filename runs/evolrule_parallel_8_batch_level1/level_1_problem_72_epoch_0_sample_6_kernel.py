import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
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
    int groups) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size * out_channels * depth_out * height_out * width_out)
        return;

    int w_out = idx % width_out;
    int h_out = (idx / width_out) % height_out;
    int d_out = (idx / (width_out * height_out)) % depth_out;
    int oc = (idx / (width_out * height_out * depth_out)) % out_channels;
    int b = idx / (out_channels * depth_out * height_out * width_out);

    int group_out_channels = out_channels / groups;
    int group = oc / group_out_channels;
    int group_out_channel = oc % group_out_channels;

    int in_channels_per_group = in_channels / groups;
    int start_in_channel = group * in_channels_per_group;
    int end_in_channel = start_in_channel + in_channels_per_group;

    float output_val = 0.0;

    for (int ic = start_in_channel; ic < end_in_channel; ic++) {
        for (int kd = 0; kd < kernel_d; kd++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int input_d_in = (d_out - kd + padding_d - output_padding_d) / stride_d;
                    int input_h_in = (h_out - kh + padding_h - output_padding_h) / stride_h;
                    int input_w_in = (w_out - kw + padding_w - output_padding_w) / stride_w;

                    if (input_d_in < 0 || input_d_in >= depth_in) continue;
                    if (input_h_in < 0 || input_h_in >= height_in) continue;
                    if (input_w_in < 0 || input_w_in >= width_in) continue;

                    int ic_in_group = ic - start_in_channel;
                    int first_dim = group * group_out_channels + group_out_channel;
                    int weight_offset = first_dim * (in_channels_per_group * kernel_d * kernel_h * kernel_w) 
                                      + ic_in_group * (kernel_d * kernel_h * kernel_w)
                                      + kd * (kernel_h * kernel_w)
                                      + kh * kernel_w
                                      + kw;

                    float w_val = weight[weight_offset];

                    int input_offset = b * in_channels * depth_in * height_in * width_in
                                      + ic * depth_in * height_in * width_in
                                      + input_d_in * height_in * width_in
                                      + input_h_in * width_in
                                      + input_w_in;

                    float in_val = input[input_offset];

                    output_val += w_val * in_val;
                }
            }
        }
    }

    int output_offset = b * out_channels * depth_out * height_out * width_out
                       + oc * depth_out * height_out * width_out
                       + d_out * height_out * width_out
                       + h_out * width_out
                       + w_out;

    output[output_offset] = output_val;
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, 
                                   int stride_d, int stride_h, int stride_w, 
                                   int padding_d, int padding_h, int padding_w,
                                   int output_padding_d, int output_padding_h, int output_padding_w,
                                   int groups) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth_in = input.size(2);
    int height_in = input.size(3);
    int width_in = input.size(4);

    int out_channels = weight.size(0);
    int in_channels_per_group = weight.size(1);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int depth_out = (depth_in - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
    int height_out = (height_in - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int width_out = (width_in - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = torch::empty({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    int num_output_elements = batch_size * out_channels * depth_out * height_out * width_out;
    const int threads_per_block = 256;
    const int num_blocks = (num_output_elements + threads_per_block - 1) / threads_per_block;

    conv_transpose3d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
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

    return output;
}
"""

conv_transpose3d_header = """
#include <torch/extension.h>
torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, 
                                   int stride_d, int stride_h, int stride_w, 
                                   int padding_d, int padding_h, int padding_w,
                                   int output_padding_d, int output_padding_h, int output_padding_w,
                                   int groups);
"""

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_header,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight using the same method as PyTorch's ConvTranspose3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv_transpose3d.conv_transpose3d_cuda(
            x,
            self.weight,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.output_padding[0], self.output_padding[1], self.output_padding[2],
            self.groups
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output