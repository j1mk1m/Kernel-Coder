import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

extern "C" {
__global__ void conv2d_cuda_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int output_height,
    int output_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_height * output_width)
        return;

    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int group_id = (idx / (output_height * output_width)) % groups;
    int c_out_group = (idx / (output_height * output_width * groups)) % (out_channels / groups);
    int n = idx / (output_height * output_width * groups * (out_channels / groups));

    int c_out = group_id * (out_channels / groups) + c_out_group;
    int in_channels_per_group = in_channels / groups;
    int in_start_channel = group_id * in_channels_per_group;

    float sum = 0.0f;
    if (bias != nullptr) {
        sum += bias[c_out];
    }

    for (int c_in = 0; c_in < in_channels_per_group; ++c_in) {
        int c_in_full = in_start_channel + c_in;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int input_h = h_out * stride_h + kh * dilation_h - padding_h;
                int input_w = w_out * stride_w + kw * dilation_w - padding_w;
                if (input_h < 0 || input_h >= input_height || 
                    input_w < 0 || input_w >= input_width) {
                    continue;
                }

                int input_offset = 
                    n * in_channels * input_height * input_width +
                    c_in_full * input_height * input_width +
                    input_h * input_width +
                    input_w;
                float in_val = input[input_offset];

                int weight_offset = 
                    c_out * in_channels_per_group * kernel_h * kernel_w +
                    c_in * kernel_h * kernel_w +
                    kh * kernel_w + kw;
                float weight_val = weight[weight_offset];

                sum += in_val * weight_val;
            }
        }
    }

    output[idx] = sum;
}

void conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int output_height,
    int output_width
) {
    int threads_per_block = 256;
    int num_elements = batch_size * out_channels * output_height * output_width;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    conv2d_cuda_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        output_height,
        output_width
    );
}
}
"""

conv2d_forward = load_inline(
    name='conv2d_cuda',
    cuda_sources=[cuda_source],
    functions=['conv2d_forward'],
    verbose=True,
)

def conv2d_cuda(input, weight, bias, stride, padding, dilation, groups):
    batch_size, in_channels, input_height, input_width = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation

    effective_kernel_h = dilation_h * (kernel_h - 1) + 1
    effective_kernel_w = dilation_w * (kernel_w - 1) + 1
    output_height = (input_height + 2 * padding_h - effective_kernel_h) // stride_h + 1
    output_width = (input_width + 2 * padding_w - effective_kernel_w) // stride_w + 1

    output = torch.empty((batch_size, out_channels, output_height, output_width),
                        dtype=input.dtype, device=input.device)

    conv2d_forward(
        input.contiguous(),
        weight.contiguous(),
        bias if bias is not None else torch.empty(0),
        output,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        output_height,
        output_width,
    )

    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"

        self.weight = nn.Parameter(torch.empty(
            out_channels,
            in_channels // groups,
            kernel_size[0],
            kernel_size[1]
        ))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x):
        return conv2d_cuda(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )