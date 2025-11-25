import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from torch import Tensor
import math

# Define the CUDA kernel source code
conv2d_custom_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void conv2d_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int dilation_h,
    const int dilation_w,
    const int output_height,
    const int output_width) {

    CUDA_1D_KERNEL_LOOP(output_idx, batch_size * out_channels * output_height * output_width) {
        const int w_out = output_idx % output_width;
        const int h_out = (output_idx / output_width) % output_height;
        const int c_out = (output_idx / (output_width * output_height)) % out_channels;
        const int n = output_idx / (out_channels * output_height * output_width);

        scalar_t sum = (bias) ? bias[c_out] : 0;

        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int dilated_kh = kh * dilation_h;
                const int dilated_kw = kw * dilation_w;

                const int input_h = h_out * stride_h - padding_h + dilated_kh;
                const int input_w = w_out * stride_w - padding_w + dilated_kw;

                if (input_h >= 0 && input_h < input_height &&
                    input_w >= 0 && input_w < input_width) {
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        const int input_offset = n * in_channels * input_height * input_width +
                                                c_in * input_height * input_width +
                                                input_h * input_width + input_w;
                        const int weight_offset = c_out * in_channels * kernel_h * kernel_w +
                                                 c_in * kernel_h * kernel_w +
                                                 kh * kernel_w + kw;
                        sum += input[input_offset] * weights[weight_offset];
                    }
                }
            }
        }

        output[output_idx] = sum;
    }
}

// Wrapper function to call the kernel
torch::Tensor conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weights.size(0);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    // Compute output dimensions
    const int output_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int output_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    const dim3 blocks = 256;
    const dim3 grids = (batch_size * out_channels * output_height * output_width + blocks.x - 1) / blocks.x;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv2d_forward_cuda", ([&] {
        conv2d_forward_kernel<scalar_t><<<grids, blocks>>>(
            input.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            (bias.defined()) ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_height,
            input_width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            output_height,
            output_width);
    }));

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));

    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w);
"""

# Compile the CUDA extension
conv2d_module = load_inline(
    name="conv2d_custom",
    cpp_sources=[conv2d_cpp_source],
    cuda_sources=[conv2d_custom_source],
    functions=["conv2d_forward_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights and bias similar to PyTorch's Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()
        
        # Configuration parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride)  # Convert to tuple for consistency
        self.padding = padding
        self.dilation = dilation

        # Initialize parameters with Xavier uniform
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # Extract parameters
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation

        # Call the custom CUDA kernel
        output = conv2d_module.conv2d_forward_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.Tensor(),
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w
        )

        return output