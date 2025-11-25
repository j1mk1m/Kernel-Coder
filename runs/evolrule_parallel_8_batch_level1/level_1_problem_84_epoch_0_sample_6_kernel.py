import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* input, const float* weight, float* output,
    int batch_size, int in_channels, int height_in, int width_in,
    int kernel_size, int height_out, int width_out) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= batch_size * in_channels * height_out * width_out) return;

    int w = tid % width_out;
    int h = (tid / width_out) % height_out;
    int c = (tid / (width_out * height_out)) % in_channels;
    int n = tid / (width_out * height_out * in_channels);

    float sum = 0.0;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = h + kh;
            int iw = w + kw;
            if (ih < height_in && iw < width_in) {
                int input_offset = n * in_channels * height_in * width_in
                    + c * height_in * width_in
                    + ih * width_in
                    + iw;
                int weight_offset = c * kernel_size * kernel_size
                    + kh * kernel_size
                    + kw;
                sum += input[input_offset] * weight[weight_offset];
            }
        }
    }

    int output_offset = n * in_channels * height_out * width_out
        + c * height_out * width_out
        + h * width_out
        + w;
    output[output_offset] = sum;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input, torch::Tensor weight) {

    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height_in = input.size(2);
    auto width_in = input.size(3);

    auto kernel_size = weight.size(2);
    auto height_out = height_in - kernel_size + 1;
    auto width_out = width_in - kernel_size + 1;

    auto output = torch::zeros({batch_size, in_channels, height_out, width_out}, 
        torch::device(input.device()).dtype(input.dtype()));

    const int threads_per_block = 256;
    const int elements = batch_size * in_channels * height_out * width_out;
    const int blocks_per_grid = (elements + threads_per_block - 1) / threads_per_block;

    depthwise_conv2d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, height_in, width_in,
        kernel_size, height_out, width_out
    );

    return output;
}
"""

depthwise_conv2d_cpp_source = (
    "torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight);"
)

depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        self.depthwise_conv2d = depthwise_conv2d
        # Initialize weight parameter with same shape as PyTorch's Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, 1, kernel_size, kernel_size))
        # Initialize weights using PyTorch's default initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthwise_conv2d.depthwise_conv2d_cuda(x, self.weight)