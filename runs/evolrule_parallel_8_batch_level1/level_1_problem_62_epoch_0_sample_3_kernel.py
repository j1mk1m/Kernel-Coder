import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

custom_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_conv2d_forward(
    const float* input, const float* weight, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_h, int kernel_w,
    int padding_h, int padding_w,
    int stride, int dilation_h, int dilation_w,
    int output_height, int output_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_height * output_width)
        return;

    int n = idx / (out_channels * output_height * output_width);
    int remaining = idx % (out_channels * output_height * output_width);
    int c_out = remaining / (output_height * output_width);
    int hw = remaining % (output_height * output_width);
    int h_out = hw / output_width;
    int w_out = hw % output_width;

    float sum = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride + kh * dilation_h - padding_h;
                int w_in = w_out * stride + kw * dilation_w - padding_w;

                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    int weight_offset = c_out * in_channels * kernel_h * kernel_w
                        + c_in * kernel_h * kernel_w
                        + kh * kernel_w + kw;
                    float w_val = weight[weight_offset];

                    int input_offset = n * in_channels * input_height * input_width
                        + c_in * input_height * input_width
                        + h_in * input_width + w_in;
                    float in_val = input[input_offset];

                    sum += in_val * w_val;
                }
            }
        }
    }

    int output_offset = n * out_channels * output_height * output_width
        + c_out * output_height * output_width
        + h_out * output_width + w_out;
    output[output_offset] = sum;
}

torch::Tensor custom_conv2d_cuda(
    torch::Tensor input, torch::Tensor weight,
    int kernel_h, int kernel_w,
    int stride, int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int output_height, int output_width
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    const int total_elements = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    custom_conv2d_forward<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_h, kernel_w,
        padding_h, padding_w,
        stride, dilation_h, dilation_w,
        output_height, output_width
    );

    return output;
}
"""

custom_conv2d_cpp_source = """
torch::Tensor custom_conv2d_cuda(
    torch::Tensor input, torch::Tensor weight,
    int kernel_h, int kernel_w,
    int stride, int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int output_height, int output_width
);
"""

custom_conv2d = load_inline(
    name="custom_conv2d",
    cpp_sources=custom_conv2d_cpp_source,
    cuda_sources=custom_conv2d_source,
    functions=["custom_conv2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
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

        # Initialize weights
        self.weight = nn.Parameter(torch.empty(
            out_channels,
            in_channels // groups,
            kernel_size[0],
            kernel_size[1]
        ))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Bias term
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

        self.custom_conv = custom_conv2d

    def forward(self, x):
        kernel_h, kernel_w = self.kernel_size
        stride = self.stride
        padding = self.padding
        dilation = self.dilation

        if isinstance(padding, tuple):
            padding_h, padding_w = padding
        else:
            padding_h = padding_w = padding

        if isinstance(dilation, tuple):
            dilation_h, dilation_w = dilation
        else:
            dilation_h = dilation_w = dilation

        input_height = x.size(2)
        input_width = x.size(3)

        numerator_h = input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1
        output_height = (numerator_h // stride) + 1

        numerator_w = input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1
        output_width = (numerator_w // stride) + 1

        output = self.custom_conv.custom_conv2d_cuda(
            x.contiguous(), self.weight.contiguous(),
            kernel_h, kernel_w,
            stride, padding_h, padding_w,
            dilation_h, dilation_w,
            output_height, output_width
        )

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        return output