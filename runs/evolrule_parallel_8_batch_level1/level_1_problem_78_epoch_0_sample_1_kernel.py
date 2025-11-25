import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

conv_transpose_source = """
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
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int output_height,
    int output_width,
    bool has_bias) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * out_channels * output_height * output_width) return;

    // Compute indices
    int n = index / (out_channels * output_height * output_width);
    int rem = index % (out_channels * output_height * output_width);
    int c_out = rem / (output_height * output_width);
    rem %= (output_height * output_width);
    int h_out = rem / output_width;
    int w_out = rem % output_width;

    float sum = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Compute input indices
                int input_h = (h_out + padding_h - kh - 1 + output_padding_h) / stride_h;
                int input_w = (w_out + padding_w - kw - 1 + output_padding_w) / stride_w;

                if (input_h < 0 || input_h >= input_height || input_w < 0 || input_w >= input_width) {
                    continue;  // Skip out of bounds
                }

                // Get weight index
                int weight_offset = c_in * out_channels * kernel_h * kernel_w +
                                    c_out * kernel_h * kernel_w +
                                    kh * kernel_w + kw;
                float w_val = weight[weight_offset];

                // Get input value
                int input_offset = n * in_channels * input_height * input_width +
                                   c_in * input_height * input_width +
                                   input_h * input_width + input_w;
                float in_val = input[input_offset];

                sum += in_val * w_val;
            }
        }
    }

    if (has_bias) {
        sum += bias[c_out];
    }

    // Write output
    int output_offset = n * out_channels * output_height * output_width +
                        c_out * output_height * output_width +
                        h_out * output_width + w_out;
    output[output_offset] = sum;
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                   int stride_h, int stride_w, int padding_h, int padding_w,
                                   int output_padding_h, int output_padding_w) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = weight.size(1);  // weight shape: [in_channels, out_channels, kh, kw]

    // Calculate output dimensions
    int output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    int total_elements = batch_size * out_channels * output_height * output_width;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    conv_transpose2d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        output_padding_h, output_padding_w,
        output_height, output_width,
        bias.defined()
    );

    cudaDeviceSynchronize();  // Ensure completion

    return output;
}
"""

conv_transpose_header = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                   int stride_h, int stride_w, int padding_h, int padding_w,
                                   int output_padding_h, int output_padding_w);
"""

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.output_padding = (0, 0)  # Default no output padding

        # Initialize parameters
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Parameter initialization (same as PyTorch's default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Load the custom CUDA kernel
        self.conv_transpose_cuda = load_inline(
            name="conv_transpose_cuda",
            cpp_sources=conv_transpose_header,
            cuda_sources=conv_transpose_source,
            functions=["conv_transpose2d_cuda"],
            verbose=False,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"]
        )

    def forward(self, x):
        # Extract parameters
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        output_padding_h, output_padding_w = self.output_padding

        # Prepare bias tensor (empty if not present)
        bias_tensor = self.bias if self.bias is not None else torch.empty(0)

        # Execute the custom CUDA kernel
        return self.conv_transpose_cuda.conv_transpose2d_cuda(
            x,
            self.weight,
            bias_tensor,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_padding_h,
            output_padding_w
        )