import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_height * output_width)
        return;

    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int c_out = (idx / (output_height * output_width)) % out_channels;
    int n = idx / (out_channels * output_height * output_width);

    int out_per_group = out_channels / groups;
    int group_id = c_out / out_per_group;
    int local_c_out = c_out % out_per_group;

    int in_channels_per_group = in_channels / groups;
    int start_in_channel = group_id * in_channels_per_group;

    float sum = 0.0f;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h_ker = kh * dilation_h;
            int w_ker = kw * dilation_w;

            int h_in = (h_out - h_ker + 2 * padding_h) / stride_h;
            int w_in = (w_out - w_ker + 2 * padding_w) / stride_w;

            if (h_in < 0 || h_in >= input_height || w_in < 0 || w_in >= input_width)
                continue;

            for (int c_in = 0; c_in < in_channels_per_group; ++c_in) {
                int global_c_in = start_in_channel + c_in;

                int weight_offset = global_c_in * out_per_group * kernel_h * kernel_w +
                                    local_c_out * kernel_h * kernel_w +
                                    kh * kernel_w + kw;

                float w_val = weight[weight_offset];

                int input_offset = n * in_channels * input_height * input_width +
                                   global_c_in * input_height * input_width +
                                   h_in * input_width + w_in;

                sum += w_val * input[input_offset];
            }
        }
    }

    if (has_bias) {
        sum += bias[c_out];
    }

    int output_offset = n * out_channels * output_height * output_width +
                        c_out * output_height * output_width +
                        h_out * output_width + w_out;

    output[output_offset] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    bool has_bias
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = weight.size(1) * groups;

    int output_height = (input_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1;
    int output_width = (input_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1;

    auto options = input.options();
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, options);

    int num_elements = output.numel();
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    conv_transpose2d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
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

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias

        weight_shape = (in_channels, out_channels // groups, kernel_size[0], kernel_size[1])
        self.weight = nn.Parameter(torch.empty(weight_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation
        groups = self.groups

        bias_tensor = self.bias if self.has_bias else torch.empty(0, device=x.device)
        output = conv_transpose2d.conv_transpose2d_cuda(
            x,
            self.weight,
            bias_tensor,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
            self.has_bias
        )
        return output