import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused depthwise and pointwise convolution CUDA kernel
fused_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void fused_conv2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> depthwise_weight,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> pointwise_weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_size, int stride, int padding, int dilation) {

    const int B = blockIdx.z;
    const int Y = blockIdx.y * blockDim.y + threadIdx.y;
    const int X = blockIdx.x * blockDim.x + threadIdx.x;

    if (Y >= output.size(2) || X >= output.size(3)) return;

    scalar_t result = 0;

    // Depthwise convolution computation
    for (int c = 0; c < in_channels; ++c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int input_y = Y * stride + ky - padding;
                int input_x = X * stride + kx - padding;
                if (input_y >= 0 && input_y < input_height && input_x >= 0 && input_x < input_width) {
                    scalar_t val = input[B][c][input_y][input_x] *
                                   depthwise_weight[c][0][ky][kx];
                    result += val;
                }
            }
        }
    }

    // Pointwise convolution computation
    for (int oc = 0; oc < out_channels; ++oc) {
        scalar_t sum = 0;
        for (int ic = 0; ic < in_channels; ++ic) {
            sum += result * pointwise_weight[oc][ic][0][0];
        }
        output[B][oc][Y][X] = sum;
    }
}

torch::Tensor fused_conv2d(
    torch::Tensor input,
    torch::Tensor depthwise_weight,
    torch::Tensor pointwise_weight,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = pointwise_weight.size(0);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    // Compute output dimensions
    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    const dim3 threads(16, 16);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_conv2d", ([&] {
        fused_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            depthwise_weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            pointwise_weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, in_channels, out_channels,
            input_height, input_width,
            kernel_size, stride, padding, dilation);
    }));

    return output;
}
"""

# Compile the fused CUDA kernel
fused_conv = load_inline(
    name="fused_conv",
    cpp_sources="",
    cuda_sources=fused_conv_source,
    functions=["fused_conv2d"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-arch=sm_86"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   stride=stride, padding=padding, 
                                   dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.fused_conv = fused_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The kernel expects weights as parameters, so we need to extract them
        depthwise_weight = self.depthwise.weight
        pointwise_weight = self.pointwise.weight

        # Execute the fused kernel
        return self.fused_conv.fused_conv2d(
            x, depthwise_weight, pointwise_weight,
            self.depthwise.kernel_size[0],
            self.depthwise.stride[0],
            self.depthwise.padding[0],
            self.depthwise.dilation[0]
        )