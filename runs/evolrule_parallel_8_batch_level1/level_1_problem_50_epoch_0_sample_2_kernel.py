import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

custom_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void custom_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int padding,
    int stride,
    int output_height,
    int output_width
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int block_z = blockIdx.z;

    int n = block_z / out_channels;
    int f = block_z % out_channels;

    int y = block_y * BLOCK_SIZE_Y + ty;
    int x = block_x * BLOCK_SIZE_X + tx;

    if (y >= output_height || x >= output_width || n >= batch_size || f >= out_channels) {
        return;
    }

    float sum = bias[f];
    for (int c = 0; c < in_channels; ++c) {
        for (int ky = 0; ky < kernel_h; ++ky) {
            for (int kx = 0; kx < kernel_w; ++kx) {
                int input_y = y * stride + ky - padding;
                int input_x = x * stride + kx - padding;

                if (input_y >= 0 && input_y < input_height && input_x >= 0 && input_x < input_width) {
                    int in_offset = n * in_channels * input_height * input_width +
                                    c * input_height * input_width +
                                    input_y * input_width + input_x;

                    int wt_offset = f * in_channels * kernel_h * kernel_w +
                                    c * kernel_h * kernel_w +
                                    ky * kernel_w + kx;

                    sum += input[in_offset] * weight[wt_offset];
                }
            }
        }
    }

    int out_offset = n * out_channels * output_height * output_width +
                     f * output_height * output_width +
                     y * output_width + x;

    output[out_offset] = sum;
}

torch::Tensor custom_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int padding,
    int stride,
    int kernel_h,
    int kernel_w,
    int out_channels,
    int output_height,
    int output_width
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * out_channels
    );

    custom_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_height,
        input_width,
        out_channels,
        kernel_h,
        kernel_w,
        padding,
        stride,
        output_height,
        output_width
    );

    cudaDeviceSynchronize();
    return output;
}
"""

custom_conv_cpp_source = (
    "torch::Tensor custom_conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int padding, int stride, int kernel_h, int kernel_w, int out_channels, int output_height, int output_width);"
)

custom_conv = load_inline(
    name="custom_conv",
    cpp_sources=custom_conv_cpp_source,
    cuda_sources=custom_conv_source,
    functions=["custom_conv2d"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=96,
            kernel_size=11,
            stride=4,
            padding=2,
            bias=True
        )
        self.custom_conv = custom_conv

    def forward(self, x):
        weight = self.conv1.weight
        bias = self.conv1.bias
        padding = self.conv1.padding[0]
        stride = self.conv1.stride[0]
        kernel_h = self.conv1.kernel_size[0]
        kernel_w = self.conv1.kernel_size[1]
        out_channels = self.conv1.out_channels
        output_height = (x.size(2) + 2*padding - kernel_h) // stride + 1
        output_width = (x.size(3) + 2*padding - kernel_w) // stride + 1

        return self.custom_conv.custom_conv2d(
            x, weight, bias,
            padding, stride,
            kernel_h, kernel_w,
            out_channels,
            output_height, output_width
        )