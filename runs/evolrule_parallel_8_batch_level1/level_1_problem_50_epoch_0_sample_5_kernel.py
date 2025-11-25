import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA implementation of 2D convolution
conv2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv2d_kernel(const scalar_t* __restrict__ input,
                             const scalar_t* __restrict__ weight,
                             scalar_t* __restrict__ output,
                             const int batch_size,
                             const int in_channels,
                             const int in_height,
                             const int in_width,
                             const int out_channels,
                             const int kernel_size,
                             const int stride,
                             const int pad) {

    const int out_height = (in_height + 2 * pad - kernel_size) / stride + 1;
    const int out_width = (in_width + 2 * pad - kernel_size) / stride + 1;

    const int output_idx = blockIdx.z * (out_height * out_width) +
                          blockIdx.y * out_width + blockIdx.x;
    const int b = blockIdx.z;
    const int c_out = threadIdx.z;
    const int y_out = blockIdx.y;
    const int x_out = blockIdx.x;

    if (c_out >= out_channels) return;

    scalar_t acc = 0;
    for (int i = 0; i < in_channels; ++i) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int y_in = -pad + y_out * stride + ky;
                const int x_in = -pad + x_out * stride + kx;
                if (y_in >= 0 && y_in < in_height && x_in >= 0 && x_in < in_width) {
                    acc += weight[c_out * in_channels * kernel_size * kernel_size +
                                 i * kernel_size * kernel_size + ky * kernel_size + kx] *
                           input[b * in_channels * in_height * in_width +
                                 i * in_height * in_width +
                                 y_in * in_width + x_in];
                }
            }
        }
    }

    output[b * out_channels * out_height * out_width +
           c_out * out_height * out_width +
           y_out * out_width + x_out] = acc;
}

torch::Tensor conv2d_forward(torch::Tensor input,
                            torch::Tensor weight,
                            int kernel_size,
                            int stride,
                            int pad) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_channels = weight.size(0);

    int out_height = (in_height + 2 * pad - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * pad - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width},
                              torch::device(input.device()).dtype(input.dtype()));

    dim3 threads(1, 1, 32); // Z dimension for out_channels
    dim3 blocks(out_width, out_height, batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_forward", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_size,
            stride,
            pad);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_forward(torch::Tensor input,
                            torch::Tensor weight,
                            int kernel_size,
                            int stride,
                            int pad);
"""

# Compile the CUDA code
conv2d = load_inline(
    name="conv2d",
    cpp_sources=[conv2d_cpp_source],
    cuda_sources=[conv2d_source],
    functions=["conv2d_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        # Initialize parameters similar to Conv2d
        self.in_channels = 3
        self.out_channels = 96
        self.kernel_size = 11
        self.stride = 4
        self.padding = 2

        # Weight initialization (same as PyTorch's default)
        self.weight = nn.Parameter(torch.randn(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        ) * (2 / (self.in_channels * self.kernel_size**2))**0.5)

        # Bias initialization
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

        # Register custom conv2d function
        self.conv2d = conv2d

    def forward(self, x):
        # Apply convolution
        out = self.conv2d.conv2d_forward(
            x,
            self.weight.view(self.out_channels, -1),  # Flatten kernel for kernel args
            self.kernel_size,
            self.stride,
            self.padding
        )
        # Add bias (element-wise addition across channels)
        out = out + self.bias.view(1, -1, 1, 1)
        return out

def get_inputs():
    batch_size = 256
    return [torch.rand(batch_size, 3, 224, 224).cuda()]

def get_init_inputs():
    return [1000]