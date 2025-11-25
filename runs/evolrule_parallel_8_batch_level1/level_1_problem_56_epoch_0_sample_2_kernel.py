import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D convolution with 5x7 kernel
conv2d_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_conv2d_forward_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int output_height, int output_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_height * output_width)
        return;

    int n = idx / (out_channels * output_height * output_width);
    int c_out = (idx / (output_height * output_width)) % out_channels;
    int h_out = (idx / output_width) % output_height;
    int w_out = idx % output_width;

    float acc = 0.0f;

    // Handle groups
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    int group = c_out / out_channels_per_group;
    int c_in_start = group * in_channels_per_group;

    // Iterate over input channels within the group
    for (int c_in_in_group = 0; c_in_in_group < in_channels_per_group; ++c_in_in_group) {
        int c_in = c_in_start + c_in_in_group;

        // Iterate over kernel elements (hardcoded 5x7)
        for (int kh = 0; kh < 5; ++kh) {
            for (int kw = 0; kw < 7; ++kw) {
                int h = h_out * stride_h - padding_h + kh * dilation_h;
                int w = w_out * stride_w - padding_w + kw * dilation_w;

                if (h < 0 || h >= input_height || w < 0 || w >= input_width)
                    continue;

                // Input index calculation (NCHW layout)
                int in_offset = n * in_channels * input_height * input_width;
                in_offset += c_in * input_height * input_width;
                in_offset += h * input_width + w;
                float in_val = input[in_offset];

                // Weight index calculation (hardcoded 5x7 kernel)
                int weight_offset = (group * out_channels_per_group + (c_out % out_channels_per_group)) * 
                                    in_channels_per_group * 5 * 7;
                weight_offset += c_in_in_group * 5 * 7;
                weight_offset += kh * 7 + kw;
                float w_val = weight[weight_offset];

                acc += in_val * w_val;
            }
        }
    }

    // Add bias
    if (bias != nullptr)
        acc += bias[c_out];

    // Output index calculation
    int out_offset = n * out_channels * output_height * output_width;
    out_offset += c_out * output_height * output_width;
    out_offset += h_out * output_width + w_out;
    output[out_offset] = acc;
}

torch::Tensor custom_conv2d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride_h, int stride_w, int padding_h, int padding_w,
    int dilation_h, int dilation_w, int groups
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int kernel_h = 5, kernel_w = 7; // Fixed kernel size for optimization

    // Compute output dimensions
    const int output_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int output_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    // Launch kernel
    const int num_elements = batch_size * out_channels * output_height * output_width;
    const int threads_per_block = 256;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    custom_conv2d_forward_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups,
        output_height, output_width
    );

    return output;
}
"""

conv2d_cpp_source = (
    "torch::Tensor custom_conv2d_forward("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, "
    "int stride_h, int stride_w, int padding_h, int padding_w, "
    "int dilation_h, int dilation_w, int groups);"
)

# Compile the inline CUDA code
custom_conv = load_inline(
    name="custom_conv",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_kernel_source,
    functions=["custom_conv2d_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1), padding: tuple = (0, 0),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_h, self.kernel_w = kernel_size

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Weight initialization (same as PyTorch's default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation

        return custom_conv.custom_conv2d_forward(
            x, self.weight, self.bias if self.bias is not None else torch.empty(0),
            stride_h, stride_w, padding_h, padding_w,
            dilation_h, dilation_w, self.groups
        )