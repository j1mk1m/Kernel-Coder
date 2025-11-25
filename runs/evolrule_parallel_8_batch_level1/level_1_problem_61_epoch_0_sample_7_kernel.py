import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose3d_forward(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int output_padding_d,
    const int output_padding_h,
    const int output_padding_w,
    const int groups,
    const int depth_in,
    const int height_in,
    const int width_in,
    const int depth_out,
    const int height_out,
    const int width_out
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * out_channels * depth_out * height_out * width_out) return;

    int batch = index / (out_channels * depth_out * height_out * width_out);
    int remainder = index % (out_channels * depth_out * height_out * width_out);
    int out_channel = remainder / (depth_out * height_out * width_out);
    remainder %= (depth_out * height_out * width_out);
    int out_d = remainder / (height_out * width_out);
    remainder %= (height_out * width_out);
    int out_h = remainder / width_out;
    int out_w = remainder % width_out;

    int group = out_channel / (out_channels / groups);
    int out_channel_in_group = out_channel % (out_channels / groups);

    scalar_t acc = 0.0;

    int in_channels_per_group = in_channels / groups;
    for (int in_channel = 0; in_channel < in_channels_per_group; ++in_channel) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int in_d = (out_d + padding_d - kd) / stride_d - output_padding_d;
                    int in_h = (out_h + padding_h - kh) / stride_h - output_padding_h;
                    int in_w = (out_w + padding_w - kw) / stride_w - output_padding_w;

                    if (in_d < 0 || in_d >= depth_in) continue;
                    if (in_h < 0 || in_h >= height_in) continue;
                    if (in_w < 0 || in_w >= width_in) continue;

                    int weight_offset = group * in_channels_per_group * (out_channels / groups) * kernel_d * kernel_h * kernel_w
                        + in_channel * (out_channels / groups) * kernel_d * kernel_h * kernel_w
                        + out_channel_in_group * kernel_d * kernel_h * kernel_w
                        + kd * kernel_h * kernel_w + kh * kernel_w + kw;

                    int input_offset = batch * in_channels * depth_in * height_in * width_in
                        + (group * in_channels_per_group + in_channel) * depth_in * height_in * width_in
                        + in_d * height_in * width_in + in_h * width_in + in_w;

                    acc += input[input_offset] * weight[weight_offset];
                }
            }
        }
    }

    int output_offset = batch * out_channels * depth_out * height_out * width_out
        + out_channel * depth_out * height_out * width_out
        + out_d * height_out * width_out + out_h * width_out + out_w;

    output[output_offset] = acc;
}

at::Tensor conv_transpose3d_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int output_padding_d,
    const int output_padding_h,
    const int output_padding_w,
    const int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth_in = input.size(2);
    int height_in = input.size(3);
    int width_in = input.size(4);

    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;

    int depth_out = (depth_in - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
    int height_out = (height_in - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int width_out = (width_in - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = at::empty({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    int threads = 256;
    int elements = batch_size * out_channels * depth_out * height_out * width_out;
    int blocks = (elements + threads - 1) / threads;

    conv_transpose3d_forward<float><<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        groups,
        depth_in, height_in, width_in,
        depth_out, height_out, width_out
    );

    return output;
}

template <typename scalar_t>
__global__ void conv_transpose3d_backward_weight(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_weight,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int output_padding_d,
    const int output_padding_h,
    const int output_padding_w,
    const int groups,
    const int depth_in,
    const int height_in,
    const int width_in,
    const int depth_out,
    const int height_out,
    const int width_out
) {
    // Implementation of backward kernel for weight gradients
    // This is a placeholder and would require similar logic to forward kernel
}

at::Tensor conv_transpose3d_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int output_padding_d,
    const int output_padding_h,
    const int output_padding_w,
    const int groups
) {
    // Implementation for backward pass, including gradients for input and weight
    // This requires separate kernels for d_input and d_weight
    return at::zeros_like(input); // Placeholder
}
"""

conv_transpose3d_cuda = load_inline(
    name="conv_transpose3d_cuda",
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_forward_cuda", "conv_transpose3d_backward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size, kernel_size)
        self.stride = (stride, stride, stride)
        self.padding = (padding, padding, padding)
        self.output_padding = (output_padding, output_padding, output_padding)
        self.groups = groups

        # Initialize weight
        self.weight = nn.Parameter(torch.empty(
            in_channels,
            out_channels // groups,
            kernel_size,
            kernel_size,
            kernel_size
        ))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        output_padding_d, output_padding_h, output_padding_w = self.output_padding

        output = conv_transpose3d_cuda.conv_transpose3d_forward_cuda(
            x,
            self.weight,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w,
            self.groups
        )

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)

        return output

    def backward(self, grad_output):
        # This is a placeholder; actual backward would call the CUDA backward function
        pass