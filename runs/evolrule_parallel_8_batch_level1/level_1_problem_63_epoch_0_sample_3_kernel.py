import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class Conv2dCustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        # Save necessary variables for backward pass
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        # Output dimensions calculation
        batch_size, in_channels, height, width = input.shape
        out_channels, _, kernel_size, _ = weight.shape
        out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        output = torch.zeros(batch_size, out_channels, out_height, out_width, device=input.device)

        # CUDA kernel launch configuration
        threads_per_block = (16, 16)
        blocks_per_grid_x = (out_width + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (out_height + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, batch_size * out_channels)

        # Launch the CUDA kernel
        conv2d_forward_kernel[blocks_per_grid, threads_per_block](
            input, weight, bias, output, 
            stride, padding, dilation, groups,
            height, width, kernel_size, out_height, out_width
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Implement backward pass if needed
        raise NotImplementedError("Backward not implemented yet")

# Define the CUDA kernel for forward pass
conv2d_forward_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv2d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits> bias,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    const int stride, const int padding, const int dilation, const int groups,
    const int input_height, const int input_width, const int kernel_size,
    const int output_height, const int output_width) {
    
    const int batch_idx = blockIdx.z / output_channels;
    const int out_channel = blockIdx.z % output_channels;
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= output_width || out_y >= output_height) return;

    const int in_channel_group = in_channels / groups;
    const int weight_offset = (out_channel / groups) * in_channel_group * kernel_size * kernel_size;

    scalar_t sum = bias[out_channel];
    for (int i = 0; i < in_channel_group; ++i) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int input_x = out_x * stride - padding + kx * dilation;
                const int input_y = out_y * stride - padding + ky * dilation;
                if (input_x >= 0 && input_x < input_width && input_y >=0 && input_y < input_height) {
                    sum += weight[weight_offset + i * kernel_size * kernel_size + ky * kernel_size + kx] *
                           input[batch_idx][i][input_y][input_x];
                }
            }
        }
    }
    output[batch_idx][out_channel][out_y][out_x] = sum;
}
"""

conv2d_forward_cpp = "template void conv2d_forward_kernel(...);"

# Load the CUDA kernel
conv2d_forward = load_inline(
    name="conv2d_forward",
    cpp_sources=conv2d_forward_cpp,
    cuda_sources=conv2d_forward_source,
    functions=[],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Initialize weights similar to PyTorch's Conv2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.bias is not None:
            return Conv2dCustomFunction.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return Conv2dCustomFunction.apply(x, self.weight, torch.zeros(0), self.stride, self.padding, self.dilation, self.groups)