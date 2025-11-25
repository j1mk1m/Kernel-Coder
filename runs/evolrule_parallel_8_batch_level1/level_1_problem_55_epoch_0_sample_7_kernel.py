import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Conv2D
conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

template <typename scalar_t>
__global__ void conv2d_kernel(const torch::PackedTensorAccessor<scalar_t,4> input,
                             const torch::PackedTensorAccessor<scalar_t,4> weight,
                             torch::PackedTensorAccessor<scalar_t,4> output,
                             int kernel_size, int stride, int padding, int dilation) {
    const int B = blockIdx.z;
    const int C_out = blockIdx.y;
    const int Y = blockIdx.x * blockDim.y + threadIdx.y;
    const int X = threadIdx.x;

    if (Y >= output.size(2) || X >= output.size(3)) {
        return;
    }

    scalar_t sum = 0;
    for (int i = 0; i < weight.size(1); ++i) {  // in_channels
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                // Compute input coordinates
                int in_y = Y * stride + ky * dilation - padding;
                int in_x = X * stride + kx * dilation - padding;

                // Check if in bounds
                if (in_y >= 0 && in_y < input.size(2) && in_x >= 0 && in_x < input.size(3)) {
                    sum += input[B][i][in_y][in_x] * weight[C_out][i][ky][kx];
                }
            }
        }
    }
    output[B][C_out][Y][X] = sum;
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding, int dilation) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(0);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    // Compute output dimensions
    auto output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 threads(32, 8);  // X and Y thread dimensions
    dim3 blocks(output_width, (output_height + threads.y - 1)/threads.y, batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_cuda", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4>(),
            weight.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            kernel_size, stride, padding, dilation);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv2d_cpp_source = (
    "torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding, int dilation);"
)

conv2d_op = load_inline(
    name="conv2d_op",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_cuda_source,
    functions=["conv2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        # Initialize weights like PyTorch Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Same as PyTorch default
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x):
        out = conv2d_op.conv2d_cuda(
            x, self.weight, self.kernel_size, self.stride, self.padding, self.dilation
        )
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out