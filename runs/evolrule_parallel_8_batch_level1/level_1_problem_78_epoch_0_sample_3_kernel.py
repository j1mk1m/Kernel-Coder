import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source code for optimized transposed convolution
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int padding_h,
    int padding_w,
    int stride_h,
    int stride_w,
    int output_height,
    int output_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_height * output_width) {
        return;
    }

    int n = idx / (out_channels * output_height * output_width);
    int remainder = idx % (out_channels * output_height * output_width);
    int c_out = remainder / (output_height * output_width);
    remainder = remainder % (output_height * output_width);
    int y_out = remainder / output_width;
    int x_out = remainder % output_width;

    float acc = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                int y_in = y_out - kh + padding_h;
                int x_in = x_out - kw + padding_w;

                if (y_in >= 0 && y_in < input_height && x_in >= 0 && x_in < input_width) {
                    int w_idx = c_out * in_channels * kernel_height * kernel_width +
                                c_in * kernel_height * kernel_width +
                                kh * kernel_width + kw;

                    int in_idx = n * in_channels * input_height * input_width +
                                 c_in * input_height * input_width +
                                 y_in * input_width + x_in;

                    acc += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    int out_idx = n * out_channels * output_height * output_width +
                  c_out * output_height * output_width +
                  y_out * output_width + x_out;

    output[out_idx] = acc;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_height,
    int kernel_width,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);

    // Compute output dimensions using PyTorch's formula
    int output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_height;
    int output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_width;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    // Launch kernel
    int total_elements = batch_size * out_channels * output_height * output_width;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    conv_transpose2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        padding_h,
        padding_w,
        stride_h,
        stride_w,
        output_height,
        output_width
    );

    cudaDeviceSynchronize();
    return output;
}
"""

# Compile the CUDA code
conv_transpose2d_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_height, int kernel_width, int stride_h, int stride_w, int padding_h, int padding_w);"
)

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        # Initialize weights similarly to PyTorch's default (kaiming_uniform_)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose2d.conv_transpose2d_cuda(
            x,
            self.weight,
            kernel_height=self.kernel_size[0],
            kernel_width=self.kernel_size[1],
            stride_h=self.stride[0],
            stride_w=self.stride[1],
            padding_h=self.padding[0],
            padding_w=self.padding[1]
        )