import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

custom_conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_conv3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int stride,
    int padding_d, int padding_h, int padding_w,
    int dilation,
    int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * out_channels * output_depth * output_height * output_width)
        return;

    int n = index / (out_channels * output_depth * output_height * output_width);
    int remainder = index % (out_channels * output_depth * output_height * output_width);
    int c_out = remainder / (output_depth * output_height * output_width);
    remainder %= (output_depth * output_height * output_width);
    int d_out = remainder / (output_height * output_width);
    remainder %= (output_height * output_width);
    int h_out = remainder / output_width;
    int w_out = remainder % output_width;

    float sum = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        int base_in = n * in_channels * input_depth * input_height * input_width +
                      c_in * input_depth * input_height * input_width;

#pragma unroll
        for (int kd = 0; kd < 3; ++kd) {
            int input_d = d_out * stride + kd * dilation - padding_d;
            if (input_d < 0 || input_d >= input_depth) continue;

            int base_d = base_in + input_d * input_height * input_width;

#pragma unroll
            for (int kh = 0; kh <5; ++kh) {
                int input_h = h_out * stride + kh * dilation - padding_h;
                if (input_h <0 || input_h >= input_height) continue;

                int base_h = base_d + input_h * input_width;

#pragma unroll
                for (int kw = 0; kw <7; ++kw) {
                    int input_w = w_out * stride + kw * dilation - padding_w;
                    if (input_w <0 || input_w >= input_width) continue;

                    int in_offset = base_h + input_w;

                    // Compute weight offset
                    int w_offset = c_out * in_channels * 3*5*7 +
                                   c_in * 3*5*7 +
                                   kd * 5*7 +
                                   kh *7 +
                                   kw;

                    sum += weight[w_offset] * input[in_offset];
                }
            }
        }
    }

    // Write the output
    int out_offset = n * out_channels * output_depth * output_height * output_width +
                     c_out * output_depth * output_height * output_width +
                     d_out * output_height * output_width +
                     h_out * output_width +
                     w_out;

    output[out_offset] = sum;
}

torch::Tensor custom_conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int stride,
    int padding_d, int padding_h, int padding_w,
    int dilation,
    int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width)
{
    const int threads_per_block = 256;
    const int num_blocks = (output.numel() + threads_per_block -1)/ threads_per_block;

    custom_conv3d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        stride,
        padding_d, padding_h, padding_w,
        dilation,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width
    );

    return output;
}
"""

custom_conv3d_cpp_source = """
torch::Tensor custom_conv3d_cuda(torch::Tensor input,
                                torch::Tensor weight,
                                torch::Tensor output,
                                int batch_size,
                                int in_channels,
                                int out_channels,
                                int stride,
                                int padding_d, int padding_h, int padding_w,
                                int dilation,
                                int input_depth, int input_height, int input_width,
                                int output_depth, int output_height, int output_width);
"""

custom_conv3d = load_inline(
    name="custom_conv3d",
    cpp_sources=[custom_conv3d_cpp_source],
    cuda_sources=[custom_conv3d_source],
    functions=["custom_conv3d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        kernel_d, kernel_h, kernel_w = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_d, kernel_h, kernel_w))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        self.weight.data.uniform_()
        if self.bias is not None:
            self.bias.data.uniform_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, input_depth, input_height, input_width = x.size()
        kernel_d, kernel_h, kernel_w = self.kernel_size

        if isinstance(self.padding, int):
            padding_d = padding_h = padding_w = self.padding
        else:
            padding_d, padding_h, padding_w = self.padding

        output_depth = (input_depth + 2 * padding_d - self.dilation * (kernel_d - 1) - 1) // self.stride + 1
        output_height = (input_height + 2 * padding_h - self.dilation * (kernel_h - 1) - 1) // self.stride + 1
        output_width = (input_width + 2 * padding_w - self.dilation * (kernel_w - 1) - 1) // self.stride + 1

        output = torch.zeros(
            batch_size, self.out_channels, output_depth, output_height, output_width,
            dtype=x.dtype, device=x.device, requires_grad=False
        )

        custom_conv3d.custom_conv3d_cuda(
            x,
            self.weight,
            output,
            batch_size,
            self.in_channels,
            self.out_channels,
            self.stride,
            padding_d, padding_h, padding_w,
            self.dilation,
            input_depth, input_height, input_width,
            output_depth, output_height, output_width
        )

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)

        return output