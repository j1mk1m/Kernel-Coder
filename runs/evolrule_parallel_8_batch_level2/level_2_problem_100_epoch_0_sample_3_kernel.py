import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the fused CUDA kernel for ConvTranspose3d + clamp + divide
fused_conv_transpose_clamp_divide_source = """
#include <torch/extension.h>
#include <ATen/ConvUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void fused_conv_transpose_clamp_divide_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> output,
    const int64_t out_channels, const int64_t in_channels,
    const int kernel_depth, const int kernel_height, const int kernel_width,
    const int stride_depth, const int stride_height, const int stride_width,
    const int padding_depth, const int padding_height, const int padding_width,
    const scalar_t min_value, const scalar_t divisor) {

    const int batch_size = input.size(0);
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);
    const int output_depth = output.size(2);
    const int output_height = output.size(3);
    const int output_width = output.size(4);

    CUDA_1D_KERNEL_LOOP(index, batch_size * out_channels * output_depth * output_height * output_width) {
        int batch = index / (out_channels * output_depth * output_height * output_width);
        int c_out = (index / (output_depth * output_height * output_width)) % out_channels;
        int di = (index / (output_height * output_width)) % output_depth;
        int hi = (index / output_width) % output_height;
        int wi = index % output_width;

        scalar_t sum = 0;
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int kd = 0; kd < kernel_depth; ++kd) {
                for (int kh = 0; kh < kernel_height; ++kh) {
                    for (int kw = 0; kw < kernel_width; ++kw) {
                        int input_d = (di - kd - padding_depth) / stride_depth;
                        int input_h = (hi - kh - padding_height) / stride_height;
                        int input_w = (wi - kw - padding_width) / stride_width;
                        if (input_d < 0 || input_d >= input_depth ||
                            input_h < 0 || input_h >= input_height ||
                            input_w < 0 || input_w >= input_width) {
                            continue;
                        }
                        sum += input[batch][c_in][input_d][input_h][input_w] * 
                               weight[c_out][c_in][kd][kh][kw];
                    }
                }
            }
        }

        scalar_t result = sum / divisor;
        result = (result < min_value) ? min_value : result;
        output[batch][c_out][di][hi][wi] = result;
    }
}

torch::Tensor fused_conv_transpose_clamp_divide_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
    double min_value, double divisor) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_depth = input.size(2);
    const auto input_height = input.size(3);
    const auto input_width = input.size(4);

    const auto out_channels = weight.size(0);
    const auto kernel_depth = weight.size(2);
    const auto kernel_height = weight.size(3);
    const auto kernel_width = weight.size(4);

    // Calculate output dimensions using proper transposed conv formula
    const int output_depth = (input_depth - 1) * stride_depth - 2 * padding_depth + kernel_depth;
    const int output_height = (input_height - 1) * stride_height - 2 * padding_height + kernel_height;
    const int output_width = (input_width - 1) * stride_width - 2 * padding_width + kernel_width;

    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    const int threads = 256;
    const int elements = batch_size * out_channels * output_depth * output_height * output_width;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_conv_transpose_clamp_divide_forward", ([&] {
        fused_conv_transpose_clamp_divide_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            weight.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            out_channels, in_channels,
            kernel_depth, kernel_height, kernel_width,
            stride_depth, stride_height, stride_width,
            padding_depth, padding_height, padding_width,
            static_cast<scalar_t>(min_value),
            static_cast<scalar_t>(divisor));
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

fused_conv_transpose_clamp_divide_cpp_source = (
    "torch::Tensor fused_conv_transpose_clamp_divide_forward(torch::Tensor input, torch::Tensor weight, "
    "int stride_depth, int stride_height, int stride_width, int padding_depth, int padding_height, int padding_width, "
    "double min_value, double divisor);"
)

# Compile the CUDA code
fused_conv_transpose_clamp_divide = load_inline(
    name="fused_conv_transpose_clamp_divide",
    cpp_sources=[fused_conv_transpose_clamp_divide_cpp_source],
    cuda_sources=[fused_conv_transpose_clamp_divide_source],
    functions=["fused_conv_transpose_clamp_divide_forward"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.stride_depth = stride if isinstance(stride, int) else stride[0]
        self.stride_height = stride[1] if isinstance(stride, tuple) else stride
        self.stride_width = stride[2] if isinstance(stride, tuple) else stride
        self.padding_depth = padding if isinstance(padding, int) else padding[0]
        self.padding_height = padding[1] if isinstance(padding, tuple) else padding
        self.padding_width = padding[2] if isinstance(padding, tuple) else padding
        self.min_value = min_value
        self.divisor = divisor
        # Initialize weights similar to PyTorch's ConvTranspose3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return fused_conv_transpose_clamp_divide.fused_conv_transpose_clamp_divide_forward(
            x, self.weight,
            self.stride_depth, self.stride_height, self.stride_width,
            self.padding_depth, self.padding_height, self.padding_width,
            self.min_value, self.divisor
        )

def get_inputs():
    batch_size = 16
    in_channels = 64
    depth, height, width = 24, 48, 48
    return [torch.randn(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    stride = 2
    padding = 1
    min_value = -1.0
    divisor = 2.0
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]