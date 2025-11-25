import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Custom CUDA kernel for forward pass of transposed convolution
conv_transpose2d_forward_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_forward_kernel(
    const float* input_data,
    const float* weight_data,
    const float* bias_data,
    float* output_data,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    bool has_bias
) {
    int output_h = (input_h - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1;
    int output_w = (input_w - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1;
    int output_size = batch_size * out_channels * output_h * output_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;

    int ow = idx % output_w;
    int oh = (idx / output_w) % output_h;
    int oc = (idx / (output_h * output_w)) % out_channels;
    int b = idx / (out_channels * output_h * output_w);

    int out_channels_per_group = out_channels / groups;
    int group = oc / out_channels_per_group;
    int oc_in_group = oc % out_channels_per_group;
    int in_channels_per_group = in_channels / groups;

    float output_val = 0.0f;

    for (int in_ch_group = 0; in_ch_group < in_channels_per_group; ++in_ch_group) {
        int in_ch_global = group * in_channels_per_group + in_ch_group;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = oh * stride_h - padding_h + kh * dilation_h;
                int iw = ow * stride_w - padding_w + kw * dilation_w;
                if (ih < 0 || ih >= input_h || iw < 0 || iw >= input_w) continue;

                int input_offset = b * in_channels * input_h * input_w;
                int input_idx = input_offset + in_ch_global * input_h * input_w + ih * input_w + iw;
                float input_val = input_data[input_idx];

                int weight_offset = in_ch_global * (out_channels_per_group * kernel_h * kernel_w) 
                                   + oc_in_group * (kernel_h * kernel_w) 
                                   + kh * kernel_w + kw;
                float weight_val = weight_data[weight_offset];

                output_val += input_val * weight_val;
            }
        }
    }

    if (has_bias) {
        output_val += bias_data[oc];
    }

    int output_offset = b * out_channels * output_h * output_w;
    int output_idx = output_offset + oc * output_h * output_w + oh * output_w + ow;
    output_data[output_idx] = output_val;
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    bool has_bias
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(1) * groups;
    int input_h = input.size(2);
    int input_w = input.size(3);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int output_h = (input_h - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1;
    int output_w = (input_w - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    dim3 block(256);
    dim3 grid((output.numel() + block.x - 1) / block.x);

    conv_transpose2d_forward_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_h,
        input_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        has_bias
    );

    return output;
}
"""

conv_transpose2d_forward_cpp_source = (
    "torch::Tensor conv_transpose2d_forward("
    "torch::Tensor input, "
    "torch::Tensor weight, "
    "torch::Tensor bias, "
    "int stride_h, int stride_w, "
    "int padding_h, int padding_w, "
    "int dilation_h, int dilation_w, "
    "int groups, bool has_bias);"
)

# Compile the CUDA code
conv_transpose2d_forward = load_inline(
    name="conv_transpose2d_forward",
    cpp_sources=conv_transpose2d_forward_cpp_source,
    cuda_sources=conv_transpose2d_forward_source,
    functions=["conv_transpose2d_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), 
                 padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights and bias similar to nn.ConvTranspose2d
        weight_shape = (in_channels, out_channels // groups, kernel_size[0], kernel_size[1])
        self.weight = nn.Parameter(torch.empty(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation
        groups = self.groups
        has_bias = self.bias is not None

        # Call the custom CUDA forward function
        output = conv_transpose2d_forward(
            x,
            self.weight,
            self.bias if has_bias else torch.empty(0),
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
            has_bias
        )

        return output