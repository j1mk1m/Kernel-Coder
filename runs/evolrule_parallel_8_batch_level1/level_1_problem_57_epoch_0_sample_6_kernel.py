import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_forward(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int input_height,
    int input_width,
    int output_height,
    int output_width
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * out_channels * output_height * output_width) {
        return;
    }

    int batch = index / (out_channels * output_height * output_width);
    int remaining = index % (out_channels * output_height * output_width);
    int c_out = remaining / (output_height * output_width);
    remaining = remaining % (output_height * output_width);
    int h_out = remaining / output_width;
    int w_out = remaining % output_width;

    float acc = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = (h_out + padding - kh) / stride;
                int w_in = (w_out + padding - kw) / stride;

                if (h_in < 0 || h_in >= input_height || w_in < 0 || w_in >= input_width) {
                    continue;
                }

                // Input index
                int input_offset = batch * in_channels * input_height * input_width
                    + c_in * input_height * input_width
                    + h_in * input_width + w_in;

                float input_val = input[input_offset];

                // Weight index
                int weight_offset = c_in * out_channels * kernel_size * kernel_size
                    + c_out * kernel_size * kernel_size
                    + kh * kernel_size + kw;

                float weight_val = weight[weight_offset];

                acc += input_val * weight_val;
            }
        }
    }

    // Output index
    int output_offset = batch * out_channels * output_height * output_width
        + c_out * output_height * output_width
        + h_out * output_width + w_out;

    output[output_offset] = acc;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int input_height,
    int input_width,
    int output_height,
    int output_width
) {
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, 
                              torch::device(input.device()).dtype(torch::kFloat32));

    int threads_per_block = 256;
    int blocks_per_grid = (output.numel() + threads_per_block - 1) / threads_per_block;

    conv_transpose2d_forward<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        input_height,
        input_width,
        output_height,
        output_width
    );

    return output;
}
"""

conv_transpose2d_cpp_source = """
extern "C" {
    torch::Tensor conv_transpose2d_cuda(
        torch::Tensor input,
        torch::Tensor weight,
        int batch_size,
        int in_channels,
        int out_channels,
        int kernel_size,
        int stride,
        int padding,
        int output_padding,
        int input_height,
        int input_width,
        int output_height,
        int output_width
    );
}
"""

conv_transpose2d_cuda = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weight
        self.weight = nn.Parameter(
            torch.empty(
                in_channels, out_channels // groups, kernel_size, kernel_size
            )
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias_param = nn.Parameter(
                torch.empty(out_channels)
            )
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)
        else:
            self.bias_param = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, input_height, input_width = x.size()
        output_height = (input_height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        output_width = (input_width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        output = conv_transpose2d_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            batch_size,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            input_height,
            input_width,
            output_height,
            output_width
        )

        if self.bias and self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1, 1)

        return output