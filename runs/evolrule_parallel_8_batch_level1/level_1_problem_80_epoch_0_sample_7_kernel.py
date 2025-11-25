import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the CUDA kernel for Conv2d with kernel_size (5, 9)
conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int stride,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int output_h,
    int output_w
) {
    int h_out = blockIdx.x * blockDim.x + threadIdx.x;
    int w_out = blockIdx.y * blockDim.y + threadIdx.y;
    if (h_out >= output_h || w_out >= output_w) return;

    int b = blockIdx.z % batch;
    int c_out = blockIdx.z / batch;
    if (c_out >= out_channels || b >= batch) return;

    float sum = 0.0f;

    int h_in_base = padding_h + h_out * stride;
    int w_in_base = padding_w + w_out * stride;
    int weight_base = c_out * in_channels * 5 * 9;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        int input_base = b * in_channels * input_h * input_w +
                         c_in * input_h * input_w;
        for (int kh = 0; kh < 5; ++kh) {
            int h_in = h_in_base + kh * dilation_h;
            if (h_in < 0 || h_in >= input_h) continue;
            for (int kw = 0; kw < 9; ++kw) {
                int w_in = w_in_base + kw * dilation_w;
                if (w_in < 0 || w_in >= input_w) continue;

                int input_offset = input_base + h_in * input_w + w_in;
                float input_val = input[input_offset];

                int weight_offset = weight_base +
                                    c_in * 5 * 9 +
                                    kh * 9 +
                                    kw;
                sum += input_val * weight[weight_offset];
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    int output_offset = b * out_channels * output_h * output_w +
                        c_out * output_h * output_w +
                        h_out * output_w +
                        w_out;
    output[output_offset] = sum;
}
"""

# Compile the CUDA kernel
conv2d = load_inline(
    name="conv2d",
    cuda_sources=conv2d_source,
    functions=["conv2d_kernel"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias

        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters as in PyTorch's Conv2d
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Compute output dimensions
        batch, _, input_h, input_w = x.size()
        kernel_h, kernel_w = self.kernel_size
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation
        stride = self.stride

        output_h = (input_h + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) // stride + 1
        output_w = (input_w + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) // stride + 1

        # Convert inputs and parameters to CUDA
        x = x.cuda()
        weight = self.weight.cuda()
        bias = self.bias.cuda() if self.bias is not None else None

        # Prepare output tensor
        output_data = torch.empty((batch, self.out_channels, output_h, output_w), device='cuda')

        # Determine grid and block dimensions
        block_dim = (32, 32, 1)  # Threads per block (h, w, batch+channel)
        grid_dim = (
            (output_h + block_dim[0] - 1) // block_dim[0],
            (output_w + block_dim[1] - 1) // block_dim[1],
            batch * self.out_channels  # blockIdx.z encodes batch and channel
        )

        # Launch the kernel
        conv2d.conv2d_kernel[grid_dim, block_dim](
            x.contiguous().data_ptr(),
            weight.contiguous().data_ptr(),
            bias.contiguous().data_ptr() if bias is not None else 0,
            output_data.data_ptr(),
            batch,
            self.in_channels,
            self.out_channels,
            input_h,
            input_w,
            stride,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            output_h,
            output_w
        )

        return output_data