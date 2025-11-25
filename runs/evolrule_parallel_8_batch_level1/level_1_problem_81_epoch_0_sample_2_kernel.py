import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    const int in_channels, const int out_channels,
    const int kernel_size, const int stride,
    const int padding, const int dilation) {

    const int B = input.size(0);
    const int output_H = output.size(2);
    const int output_W = output.size(3);
    const int input_H = input.size(2);
    const int input_W = input.size(3);

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= B * out_channels * output_H * output_W) return;

    int b = n / (out_channels * output_H * output_W);
    int c_out = (n / (output_H * output_W)) % out_channels;
    int y_out = (n / output_W) % output_H;
    int x_out = n % output_W;

    scalar_t val = 0;

    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            // Compute the effective kernel position with dilation
            int dy = ky * dilation;
            int dx = kx * dilation;

            // Compute the corresponding input position
            int y_in = y_out - dy;
            int x_in = x_out - dx;

            // Adjust for padding and stride
            y_in += padding;
            x_in += padding;

            // Check if within input bounds
            if (y_in < 0 || y_in >= input_H || x_in < 0 || x_in >= input_W) 
                continue;

            // Correct weight indexing: weight[in_c][out_c][ky][kx]
            for (int c_in = 0; c_in < in_channels; ++c_in) {
                val += weight[c_in][c_out][ky][kx] * input[b][c_in][y_in][x_in];
            }
        }
    }

    output[b][c_out][y_out][x_out] = val;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {

    const auto B = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_H = input.size(2);
    const auto input_W = input.size(3);

    const auto out_channels = weight.size(1); // Corrected weight dimensions
    const auto kernel_size = weight.size(2);

    // Calculate output dimensions correctly for transposed conv
    const int output_H = (input_H - 1) * stride - 2 * padding 
                        + dilation * (kernel_size - 1) + 1;
    const int output_W = (input_W - 1) * stride - 2 * padding 
                        + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({B, out_channels, output_H, output_W}, input.options());

    dim3 threadsPerBlock(256);
    dim3 numBlocks((B * out_channels * output_H * output_W + threadsPerBlock - 1) 
                  / threadsPerBlock);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            in_channels, out_channels, kernel_size, stride, padding, dilation);
    }));

    if (bias.defined()) {
        output += bias.view({1, -1, 1, 1});
    }

    return output;
}
"""

conv_transpose_2d = load_inline(
    name="conv_transpose_2d",
    cpp_sources="",
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights with correct dimensions [in_channels, out_channels, kernel, kernel]
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.conv_transpose_2d = conv_transpose_2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.bias is not None else torch.tensor([])
        return self.conv_transpose_2d.conv_transpose2d_cuda(
            x, self.weight, bias, self.stride, self.padding, self.dilation
        )