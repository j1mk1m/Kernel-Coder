import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the CUDA kernel source code
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_forward(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height_in,
    int width_in,
    int kernel_size,
    int stride,
    int padding,
    int height_out,
    int width_out
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * out_channels * height_out * width_out) {
        return;
    }

    int w_out = index % width_out;
    int h_out = (index / width_out) % height_out;
    int k = (index / (height_out * width_out)) % out_channels;
    int b = index / (out_channels * height_out * width_out);

    int c = k % in_channels;

    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int h_in = h_out * stride + i - padding;
            int w_in = w_out * stride + j - padding;
            if (h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
                int input_offset = b * in_channels * height_in * width_in +
                                   c * height_in * width_in +
                                   h_in * width_in + w_in;
                int weight_offset = k * kernel_size * kernel_size +
                                    i * kernel_size + j;
                sum += input[input_offset] * weight[weight_offset];
            }
        }
    }

    int output_offset = b * out_channels * height_out * width_out +
                        k * height_out * width_out +
                        h_out * width_out + w_out;
    output[output_offset] = sum;
}

torch::Tensor depthwise_conv2d_forward_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding) {
    input = input.contiguous();
    weight = weight.contiguous();

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int height_in = input.size(2);
    int width_in = input.size(3);

    int height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
    int width_out = (width_in + 2 * padding - kernel_size) / stride + 1;

    torch::Tensor output = torch::empty({batch_size, out_channels, height_out, width_out},
                                       torch::dtype(input.dtype()).device(input.device()));

    int num_elements = batch_size * out_channels * height_out * width_out;
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    depthwise_conv2d_forward<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height_in,
        width_in,
        kernel_size,
        stride,
        padding,
        height_out,
        width_out
    );

    cudaDeviceSynchronize();
    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_forward_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding);
"""

# Compile the CUDA kernel
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weight parameter
        self.weight = nn.Parameter(torch.empty(out_channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        output = depthwise_conv2d.depthwise_conv2d_forward_cuda(x, self.weight, self.stride, self.padding)
        if self.bias is not None:
            output += self.bias.view(1, self.bias.size(0), 1, 1)
        return output