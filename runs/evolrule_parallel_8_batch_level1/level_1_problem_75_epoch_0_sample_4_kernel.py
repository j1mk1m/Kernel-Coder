import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# CUDA kernel code for transposed convolution
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_cuda_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int input_h,
    int input_w,
    int output_h,
    int output_w) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_h * output_w) return;

    int ow = idx % output_w;
    int rem = idx / output_w;
    int oh = rem % output_h;
    rem /= output_h;
    int oc = rem % out_channels;
    int b = rem / out_channels;

    int out_per_group = out_channels / groups;
    int g = oc / out_per_group;
    int oc_in_group = oc % out_per_group;

    int in_start = g * (in_channels / groups);
    int in_end = in_start + (in_channels / groups);

    float total = 0.0f;

    for (int ic = in_start; ic < in_end; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int numerator_h = oh + padding_h - kh * dilation_h;
                if (numerator_h % stride_h != 0) continue;
                int h_in = numerator_h / stride_h;
                if (h_in < 0 || h_in >= input_h) continue;

                int numerator_w = ow + padding_w - kw * dilation_w;
                if (numerator_w % stride_w != 0) continue;
                int w_in = numerator_w / stride_w;
                if (w_in < 0 || w_in >= input_w) continue;

                // Compute weight index
                int weight_offset = ic * (out_per_group * kernel_h * kernel_w) 
                    + oc_in_group * (kernel_h * kernel_w) 
                    + kh * kernel_w + kw;
                float w_val = weight[weight_offset];

                // Compute input index
                int input_offset = b * in_channels * input_h * input_w 
                    + ic * input_h * input_w 
                    + h_in * input_w + w_in;
                float in_val = input[input_offset];

                total += w_val * in_val;
            }
        }
    }

    if (bias) {
        total += bias[oc];
    }

    // Compute output index
    int output_offset = b * out_channels * output_h * output_w 
        + oc * output_h * output_w 
        + oh * output_w + ow;
    output[output_offset] = total;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);

    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int out_channels = weight.size(1) * groups;

    // Compute output dimensions
    int output_h = (input_h - 1) * stride_h - 2 * padding_h 
        + dilation_h * (kernel_h - 1) + 1;
    int output_w = (input_w - 1) * stride_w - 2 * padding_w 
        + dilation_w * (kernel_w - 1) + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, 
                              input.options());

    const int threads_per_block = 256;
    const int num_elements = batch_size * out_channels * output_h * output_w;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    conv_transpose2d_cuda_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        (bias.defined()) ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        input_h,
        input_w,
        output_h,
        output_w
    );

    cudaDeviceSynchronize();

    return output;
}
"""

# Compile the CUDA kernel
conv_transpose = load_inline(
    name="conv_transpose",
    cpp_sources="",
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        kernel_h, kernel_w = kernel_size
        weight_shape = (in_channels, out_channels // groups, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose.conv_transpose2d_cuda(
            x, 
            self.weight, 
            self.bias if self.bias is not None else torch.Tensor([]),
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups
        )