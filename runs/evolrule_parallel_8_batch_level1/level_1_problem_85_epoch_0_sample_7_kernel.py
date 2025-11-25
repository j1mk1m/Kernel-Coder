import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# CUDA kernel code
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N,
    int C,
    int H,
    int W,
    int kernel_size_h,
    int kernel_size_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int H_out,
    int W_out,
    bool has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H_out * W_out) return;

    int n = idx / (C * H_out * W_out);
    int remainder = idx % (C * H_out * W_out);
    int c = remainder / (H_out * W_out);
    int rem_xy = remainder % (H_out * W_out);
    int y_out = rem_xy / W_out;
    int x_out = rem_xy % W_out;

    float acc = 0.0f;

    int y_start = y_out * stride_h - padding_h;
    int x_start = x_out * stride_w - padding_w;

    for (int ky = 0; ky < kernel_size_h; ++ky) {
        for (int kx = 0; kx < kernel_size_w; ++kx) {
            int y_in_padded = y_start + dilation_h * ky;
            int x_in_padded = x_start + dilation_w * kx;

            if (y_in_padded < 0 || y_in_padded >= (H + 2 * padding_h))
                continue;
            if (x_in_padded < 0 || x_in_padded >= (W + 2 * padding_w))
                continue;

            int y_in = y_in_padded - padding_h;
            int x_in = x_in_padded - padding_w;

            int input_offset = n * C * H * W + c * H * W + y_in * W + x_in;
            float input_val = input[input_offset];

            int weight_offset = c * kernel_size_h * kernel_size_w + ky * kernel_size_w + kx;
            float weight_val = weight[weight_offset];

            acc += input_val * weight_val;
        }
    }

    if (has_bias) {
        acc += bias[c];
    }

    int output_offset = n * C * H_out * W_out + c * H_out * W_out + y_out * W_out + x_out;
    output[output_offset] = acc;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w
) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int kernel_size_h = weight.size(1);
    int kernel_size_w = weight.size(2);

    // Compute output dimensions
    int H_out = (H + 2 * padding_h - dilation_h * (kernel_size_h - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * padding_w - dilation_w * (kernel_size_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C, H_out, W_out}, input.options());

    bool has_bias = bias.defined();

    int threads_per_block = 256;
    int total_elements = N * C * H_out * W_out;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    depthwise_conv2d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C, H, W,
        kernel_size_h, kernel_size_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        H_out, W_out,
        has_bias
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

# Compile the CUDA code
depthwise_conv2d_cuda = load_inline(
    name="depthwise_conv2d_cuda",
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        assert groups == in_channels, "Groups must be equal to in_channels for depthwise convolution"
        self.in_channels = in_channels
        self.out_channels = out_channels  # Must equal in_channels for depthwise, but kept per original API
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.bias = bias

        # Initialize parameters
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights and biases similarly to PyTorch's Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Reshape weight to (C, kH, kW) by squeezing the second dimension
        weight_reshaped = self.weight.squeeze(1)
        bias = self.bias if self.bias is not None else None

        # Call the custom CUDA kernel
        output = depthwise_conv2d_cuda.depthwise_conv2d_cuda(
            x,
            weight_reshaped,
            bias,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w
        )

        return output