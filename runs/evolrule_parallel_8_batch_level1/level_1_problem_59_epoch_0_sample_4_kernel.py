import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv3d_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void conv3d_cuda_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    int output_depth,
    int output_height,
    int output_width) {

    CUDA_KERNEL_LOOP(index, batch_size * out_channels * output_depth * output_height * output_width) {
        int w_out = index % output_width;
        int h_out = (index / output_width) % output_height;
        int d_out = (index / (output_width * output_height)) % output_depth;
        int c_out = (index / (output_width * output_height * output_depth)) % out_channels;
        int b = index / (out_channels * output_depth * output_height * output_width);

        int out_channels_per_group = out_channels / groups;
        int group = c_out / out_channels_per_group;
        int c_out_in_group = c_out % out_channels_per_group;

        int in_channels_per_group = in_channels / groups;
        int c_in_start = group * in_channels_per_group;

        float sum = 0.0f;

        for (int c_in = 0; c_in < in_channels_per_group; ++c_in) {
            int c_in_total = c_in_start + c_in;

            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int h_in = h_out * stride - padding + kh * dilation;
                    int w_in = w_out * stride - padding + kw * dilation;
                    int d_in = d_out * stride - padding + 0 * dilation;

                    if (h_in < 0 || h_in >= input_height) continue;
                    if (w_in < 0 || w_in >= input_width) continue;
                    if (d_in < 0 || d_in >= input_depth) continue;

                    int input_offset = b * in_channels * input_depth * input_height * input_width +
                                      c_in_total * input_depth * input_height * input_width +
                                      d_in * input_height * input_width +
                                      h_in * input_width +
                                      w_in;

                    float input_val = input[input_offset];

                    int weight_offset = c_out * in_channels_per_group * kernel_size * kernel_size +
                                       c_in * kernel_size * kernel_size +
                                       kh * kernel_size +
                                       kw;

                    float weight_val = weight[weight_offset];

                    sum += input_val * weight_val;
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[c_out];
        }

        int output_offset = b * out_channels * output_depth * output_height * output_width +
                           c_out * output_depth * output_height * output_width +
                           d_out * output_height * output_width +
                           h_out * output_width +
                           w_out;

        output[output_offset] = sum;
    }
}
"""

conv3d_forward = load_inline(
    name="conv3d_forward",
    cuda_sources=conv3d_kernel_source,
    functions=["conv3d_cuda_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, 1, kernel_size, kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters (like PyTorch's default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Compute output dimensions
        input_depth = x.size(2)
        input_height = x.size(3)
        input_width = x.size(4)

        # Compute output depth, height, width
        kernel_size_d = 1  # kernel depth is fixed at 1
        output_depth = (input_depth + 2 * self.padding - self.dilation * (kernel_size_d - 1)) // self.stride + 1
        output_height = (input_height + 2 * self.padding - self.dilation * (self.kernel_size - 1)) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.dilation * (self.kernel_size - 1)) // self.stride + 1

        # Create output tensor
        output_size = (x.size(0), self.out_channels, output_depth, output_height, output_width)
        output = torch.empty(output_size, device=x.device, dtype=x.dtype)

        # Calculate total elements for kernel loop
        total_elements = x.size(0) * self.out_channels * output_depth * output_height * output_width

        # Launch kernel
        block_size = 256
        grid_size = (total_elements + block_size - 1) // block_size

        conv3d_forward(
            input=x.contiguous(),
            weight=self.weight.contiguous(),
            bias=self.bias.contiguous() if self.bias is not None else None,
            output=output,
            batch_size=x.size(0),
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_depth=input_depth,
            input_height=input_height,
            input_width=input_width,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            output_depth=output_depth,
            output_height=output_height,
            output_width=output_width,
            grid=(grid_size, 1, 1),
            block=(block_size, 1, 1)
        )

        return output