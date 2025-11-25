import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv3d_source = """
#include <torch/extension.h>

__global__ void conv3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int input_depth, int input_height, int input_width,
    int out_channels,
    int kernel_d, int kernel_h, int kernel_w,
    int stride,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int depth_out, int height_out, int width_out,
    int groups,
    int has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size * out_channels * depth_out * height_out * width_out)
        return;

    int w_out = idx % width_out;
    int h_out = (idx / width_out) % height_out;
    int d_out = (idx / (width_out * height_out)) % depth_out;
    int c_out = (idx / (width_out * height_out * depth_out)) % out_channels;
    int n = idx / (width_out * height_out * depth_out * out_channels);

    float acc = 0.0;

    int out_per_group = out_channels / groups;
    int g_out = c_out / out_per_group;
    int in_per_group = in_channels / groups;
    int c_in_start = g_out * in_per_group;
    int c_in_end = (g_out + 1) * in_per_group;

    for (int c_in = c_in_start; c_in < c_in_end; c_in++) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int d = d_out * stride + kd * dilation_d - padding_d;
                    int h = h_out * stride + kh * dilation_h - padding_h;
                    int w = w_out * stride + kw * dilation_w - padding_w;

                    if (d < 0 || d >= input_depth || h < 0 || h >= input_height || w < 0 || w >= input_width)
                        continue;

                    int input_offset = n * in_channels * input_depth * input_height * input_width +
                        c_in * input_depth * input_height * input_width +
                        d * input_height * input_width +
                        h * input_width +
                        w;

                    float input_val = input[input_offset];

                    int c_in_group = c_in - c_in_start;
                    int weight_offset = c_out * in_per_group * kernel_d * kernel_h * kernel_w +
                        c_in_group * kernel_d * kernel_h * kernel_w +
                        kd * kernel_h * kernel_w +
                        kh * kernel_w +
                        kw;

                    float weight_val = weight[weight_offset];

                    acc += input_val * weight_val;
                }
            }
        }
    }

    if (has_bias) {
        acc += bias[c_out];
    }

    int output_offset = n * out_channels * depth_out * height_out * width_out +
        c_out * depth_out * height_out * width_out +
        d_out * height_out * width_out +
        h_out * width_out +
        w_out;

    output[output_offset] = acc;
}

torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups,
    int has_bias
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);

    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int depth_out = (input_depth + 2 * padding_d - dilation_d * (kernel_d - 1) - 1) / stride + 1;
    int height_out = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int width_out = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out},
                              input.options());

    int threads_per_block = 256;
    int total_elements = batch_size * out_channels * depth_out * height_out * width_out;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    conv3d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_depth, input_height, input_width,
        out_channels,
        kernel_d, kernel_h, kernel_w,
        stride,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w,
        depth_out, height_out, width_out,
        groups,
        has_bias
    );

    cudaDeviceSynchronize();
    return output;
}
"""

conv3d_cpp_source = """
#include <torch/extension.h>
#include <tuple>

torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups,
    int has_bias
);
"""

conv3d = load_inline(
    name="conv3d_cuda",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_cuda"],
    verbose=True,
)

def _conv3d_cuda(input, weight, bias, stride, padding_d, padding_h, padding_w, dilation_d, dilation_h, dilation_w, groups, has_bias):
    return conv3d.conv3d_cuda(
        input.contiguous(),
        weight.contiguous(),
        bias.contiguous() if has_bias else torch.empty(0),
        stride,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w,
        groups,
        has_bias
    )

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        self.bias_param = nn.Parameter(torch.empty(out_channels)) if bias else None

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x):
        if isinstance(self.padding, int):
            padding = (self.padding,) * 3
        else:
            padding = self.padding

        if isinstance(self.dilation, int):
            dilation = (self.dilation,) * 3
        else:
            dilation = self.dilation

        padding_d, padding_h, padding_w = padding
        dilation_d, dilation_h, dilation_w = dilation

        return _conv3d_cuda(
            x,
            self.weight,
            self.bias_param,
            self.stride,
            padding_d, padding_h, padding_w,
            dilation_d, dilation_h, dilation_w,
            self.groups,
            1 if self.bias else 0
        )