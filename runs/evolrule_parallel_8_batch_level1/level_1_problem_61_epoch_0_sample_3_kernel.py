import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv_transpose3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define GET_INPUT_INDEX(b, ic, di, hi, wi, in_channels, input_depth, input_height, input_width) \\
    ((b) * (in_channels) * (input_depth) * (input_height) * (input_width) + \\
    (ic) * (input_depth) * (input_height) * (input_width) + \\
    (di) * (input_height) * (input_width) + \\
    (hi) * (input_width) + (wi))

#define GET_WEIGHT_INDEX(ic, oc, kd, kh, kw, out_channels_per_group, kernel_depth, kernel_height, kernel_width, groups) \\
    ((ic) * (out_channels_per_group) * (kernel_depth) * (kernel_height) * (kernel_width) + \\
    (oc) * (kernel_depth) * (kernel_height) * (kernel_width) + \\
    (kd) * (kernel_height) * (kernel_width) + \\
    (kh) * (kernel_width) + (kw))

#define GET_OUTPUT_INDEX(b, oc, d, h, w, out_channels, out_depth, out_height, out_width) \\
    ((b) * (out_channels) * (out_depth) * (out_height) * (out_width) + \\
    (oc) * (out_depth) * (out_height) * (out_width) + \\
    (d) * (out_height) * (out_width) + \\
    (h) * (out_width) + (w))

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int out_channels_per_group,
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
    int groups,
    int out_depth,
    int out_height,
    int out_width
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * out_channels * out_depth * out_height * out_width)
        return;

    int w = tid % out_width;
    int h = (tid / out_width) % out_height;
    int d = (tid / (out_width * out_height)) % out_depth;
    int oc = (tid / (out_width * out_height * out_depth)) % out_channels;
    int b = tid / (out_channels * out_depth * out_height * out_width);

    oc /= out_channels_per_group;  // Handle groups
    int group = (tid / (out_channels * out_depth * out_height * out_width)) % groups;
    oc += group * out_channels_per_group;

    float sum = 0.0;

    for (int ic = 0; ic < in_channels / groups; ic++) {
        for (int kd = 0; kd < kernel_depth; kd++) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int di = (d - kd + padding_d) / stride_d - output_padding_d;
                    int hi = (h - kh + padding_h) / stride_h - output_padding_h;
                    int wi = (w - kw + padding_w) / stride_w - output_padding_w;

                    if (di < 0 || di >= input_depth || hi < 0 || hi >= input_height || wi < 0 || wi >= input_width)
                        continue;

                    int in_channel_group = ic + (group * in_channels / groups);
                    int weight_index = GET_WEIGHT_INDEX(
                        in_channel_group,
                        oc % out_channels_per_group,
                        kd, kh, kw,
                        out_channels_per_group,
                        kernel_depth, kernel_height, kernel_width,
                        groups
                    );

                    float w_val = weight[weight_index];
                    float in_val = input[GET_INPUT_INDEX(
                        b, in_channel_group, di, hi, wi,
                        in_channels, input_depth, input_height, input_width
                    )];
                    sum += w_val * in_val;
                }
            }
        }
    }

    if (bias) sum += bias[oc];

    output[GET_OUTPUT_INDEX(b, oc, d, h, w, out_channels, out_depth, out_height, out_width)] = sum;
}

torch::Tensor conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
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
    auto input_size = input.sizes();
    int batch_size = input_size[0];
    int in_channels = input_size[1];
    int input_depth = input_size[2];
    int input_height = input_size[3];
    int input_width = input_size[4];

    int kernel_depth = weight.size(2);
    int kernel_height = weight.size(3);
    int kernel_width = weight.size(4);

    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;

    int out_depth = (input_depth - 1) * stride_d + kernel_depth - 2 * padding_d + output_padding_d;
    int out_height = (input_height - 1) * stride_h + kernel_height - 2 * padding_h + output_padding_h;
    int out_width = (input_width - 1) * stride_w + kernel_width - 2 * padding_w + output_padding_w;

    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    const int threads_per_block = 256;
    const int num_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    conv_transpose3d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        out_channels_per_group,
        input_depth,
        input_height,
        input_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        groups,
        out_depth, out_height, out_width
    );

    return output;
}
"""

conv_transpose3d_cuda_header = """
torch::Tensor conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
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
);
"""

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cuda_header,
    cuda_sources=conv_transpose3d_cuda_source,
    functions=["conv_transpose3d_forward"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size, kernel_size)
        self.stride = (stride, stride, stride)
        self.padding = (padding, padding, padding)
        self.output_padding = (output_padding, output_padding, output_padding)
        self.groups = groups
        self.bias = bias

        # Initialize weights and bias
        kernel_shape = (in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(kernel_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.cuda_conv = conv_transpose3d

    def forward(self, x):
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        output_padding_d, output_padding_h, output_padding_w = self.output_padding

        bias_tensor = self.bias if self.bias is not None else torch.empty(0)

        return self.cuda_conv.conv_transpose3d_forward(
            x,
            self.weight,
            bias_tensor,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w,
            self.groups
        )