import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# CUDA kernel code
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

__global__ void conv_transpose2d_kernel(
    const float* input_data,
    const float* weight_data,
    float* output_data,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool has_bias,
    const float* bias_data
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_height * output_width) return;

    int b = idx / (out_channels * output_height * output_width);
    int remaining = idx % (out_channels * output_height * output_width);
    int oc = remaining / (output_height * output_width);
    remaining %= (output_height * output_width);
    int y_out = remaining / output_width;
    int x_out = remaining % output_width;

    float output_val = 0.0f;
    if (has_bias) {
        output_val += bias_data[oc];
    }

    for (int ic = 0; ic < in_channels; ic++) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int ky_effective = kernel_size - 1 - ky;
                int y_in = (y_out + padding - dilation * ky_effective) / stride;
                int kx_effective = kernel_size - 1 - kx;
                int x_in = (x_out + padding - dilation * kx_effective) / stride;

                if (y_in < 0 || y_in >= input_height || x_in < 0 || x_in >= input_width) {
                    continue;
                }

                // Input access
                int input_offset = b * in_channels * input_height * input_width +
                    ic * input_height * input_width +
                    y_in * input_width + x_in;
                float input_val = input_data[input_offset];

                // Weight access
                int weight_offset = ic * out_channels * kernel_size * kernel_size +
                    oc * kernel_size * kernel_size +
                    ky * kernel_size + kx;
                float weight_val = weight_data[weight_offset];

                output_val += input_val * weight_val;
            }
        }
    }

    // Output storage
    int output_offset = b * out_channels * output_height * output_width +
        oc * output_height * output_width +
        y_out * output_width + x_out;
    output_data[output_offset] = output_val;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation,
    int kernel_size,
    bool has_bias,
    torch::Tensor bias
) {
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    if (has_bias) {
        CHECK_CUDA(bias);
    }

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    // Compute output dimensions
    int output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::empty({batch_size, weight.size(1), output_height, output_width}, input.options());

    const int threads_per_block = 256;
    int num_elements = batch_size * weight.size(1) * output_height * output_width;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    conv_transpose2d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        weight.size(1),
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        dilation,
        has_bias,
        has_bias ? bias.data_ptr<float>() : nullptr
    );

    cudaDeviceSynchronize();
    return output;
}
"""

# C++ header declarations
conv_transpose_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation,
    int kernel_size,
    bool has_bias,
    torch::Tensor bias
);
"""

# Load the CUDA extension
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # Initialize weights and bias with same initialization as PyTorch's ConvTranspose2d
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        self.bias_param = nn.Parameter(torch.empty(out_channels)) if bias else None

        # Weight initialization (kaiming_uniform)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x):
        has_bias = self.bias_param is not None
        return conv_transpose2d.conv_transpose2d_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.kernel_size,
            has_bias,
            self.bias_param if has_bias else torch.empty(0, device=x.device)
        )