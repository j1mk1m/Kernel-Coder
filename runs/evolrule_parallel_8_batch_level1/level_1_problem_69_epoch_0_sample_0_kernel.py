import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_forward(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height_in,
    int width_in,
    int height_out,
    int width_out,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int bias_flag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * height_out * width_out) return;

    int w_out = idx % width_out;
    int h_out = (idx / width_out) % height_out;
    int c_out = (idx / (width_out * height_out)) % out_channels;
    int n = idx / (width_out * height_out * out_channels);

    float sum = 0.0;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = (h_out + padding_h - kh * dilation_h - output_padding_h) / stride_h;
                int w_in = (w_out + padding_w - kw * dilation_w - output_padding_w) / stride_w;

                if (h_in < 0 || h_in >= height_in || w_in < 0 || w_in >= width_in) {
                    continue;
                }

                int group = c_in / (in_channels / groups);
                int c_in_group = c_in % (in_channels / groups);
                int c_out_group = c_out % (out_channels / groups);

                int kernel_offset = group * (in_channels/groups) * (out_channels/groups) * kernel_h * kernel_w
                                   + c_in_group * (out_channels/groups) * kernel_h * kernel_w
                                   + c_out_group * kernel_h * kernel_w
                                   + kh * kernel_w + kw;
                float kernel_val = kernel[kernel_offset];

                int input_offset = n * in_channels * height_in * width_in
                                  + c_in * height_in * width_in
                                  + h_in * width_in + w_in;
                float input_val = input[input_offset];

                sum += input_val * kernel_val;
            }
        }
    }

    if (bias_flag) {
        sum += bias[c_out];
    }

    int output_offset = n * out_channels * height_out * width_out
                      + c_out * height_out * width_out
                      + h_out * width_out + w_out;
    output[output_offset] = sum;
}

torch::Tensor conv_transpose2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int bias_flag
) {
    auto input_size = input.sizes();
    auto weight_size = weight.sizes();

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height_in = input.size(2);
    int width_in = input.size(3);
    int kernel_h = weight_size[2];
    int kernel_w = weight_size[3];
    int out_channels = weight_size[1] * groups;

    int height_out = (height_in - 1)*stride_h - 2*padding_h + dilation_h*(kernel_h-1) + output_padding_h + 1;
    int width_out = (width_in -1)*stride_w - 2*padding_w + dilation_w*(kernel_w-1) + output_padding_w + 1;

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::empty({batch_size, out_channels, height_out, width_out}, output_options);

    int threads_per_block = 256;
    int num_elements = batch_size * out_channels * height_out * width_out;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    dim3 threads(threads_per_block);
    dim3 blocks(blocks_per_grid);

    conv_transpose2d_forward<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height_in,
        width_in,
        height_out,
        width_out,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        dilation_h,
        dilation_w,
        groups,
        bias_flag
    );

    return output;
}
"""

conv_transpose2d_cuda = load_inline(
    name="conv_transpose2d_cuda",
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1), padding: tuple = (0, 0),
                 output_padding: tuple = (0, 0), dilation: tuple = (1, 1),
                 groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        kernel_h, kernel_w = kernel_size
        self.weight = nn.Parameter(torch.empty(
            (in_channels, out_channels // groups, kernel_h, kernel_w)
        ))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        output_padding_h, output_padding_w = self.output_padding
        dilation_h, dilation_w = self.dilation
        bias_flag = 1 if self.bias_param is not None else 0

        return conv_transpose2d_cuda.conv_transpose2d_forward_cuda(
            x,
            self.weight,
            self.bias_param if self.bias_param is not None else torch.empty(0),
            stride_h, stride_w,
            padding_h, padding_w,
            output_padding_h, output_padding_w,
            dilation_h, dilation_w,
            self.groups,
            bias_flag
        )