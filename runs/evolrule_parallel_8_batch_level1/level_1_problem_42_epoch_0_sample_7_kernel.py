import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom MaxPool2d CUDA kernel implementation
maxpool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void maxpool2d_kernel(const scalar_t* __restrict__ input,
                                scalar_t* __restrict__ output,
                                const int batch_size,
                                const int channels,
                                const int in_height,
                                const int in_width,
                                const int out_height,
                                const int out_width,
                                const int kernel_size,
                                const int stride,
                                const int padding,
                                const int dilation) {
    // Calculate the linear index
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * channels * out_height * out_width) return;

    // Compute output spatial indices
    const int w = index % out_width;
    const int h = (index / out_width) % out_height;
    const int c = (index / (out_width * out_height)) % channels;
    const int n = index / (channels * out_height * out_width);

    // Compute input spatial starting position with padding
    int in_h_start = -padding + h * stride;
    int in_w_start = -padding + w * stride;

    scalar_t max_val = -FLT_MAX;

    // Iterate over the kernel area
    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            int in_h = in_h_start + ky * dilation;
            int in_w = in_w_start + kx * dilation;

            // Check if the current position is within input bounds
            if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                const int input_offset = ((n * channels + c) * in_height + in_h) * in_width + in_w;
                scalar_t val = input[input_offset];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }

    // Compute output offset and write result
    const int output_offset = ((n * channels + c) * out_height + h) * out_width + w;
    output[output_offset] = max_val;
}

torch::Tensor maxpool2d_forward_cuda(torch::Tensor input,
                                    int kernel_size,
                                    int stride,
                                    int padding,
                                    int dilation) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);

    // Compute output dimensions
    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Output tensor
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());

    // Number of threads and blocks
    const int threads_per_block = 256;
    const int num_elements = batch_size * channels * out_height * out_width;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "maxpool2d_forward_cuda", ([&] {
        maxpool2d_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            channels,
            in_height,
            in_width,
            out_height,
            out_width,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    return output;
}
"""

maxpool2d_cpp_source = """
torch::Tensor maxpool2d_forward_cuda(torch::Tensor input,
                                    int kernel_size,
                                    int stride,
                                    int padding,
                                    int dilation);
"""

# Compile the CUDA kernel
maxpool2d = load_inline(
    name="maxpool2d",
    cpp_sources=maxpool2d_cpp_source,
    cuda_sources=maxpool2d_source,
    functions=["maxpool2d_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.forward_func = maxpool2d.maxpool2d_forward_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_func(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )

def get_inputs():
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]