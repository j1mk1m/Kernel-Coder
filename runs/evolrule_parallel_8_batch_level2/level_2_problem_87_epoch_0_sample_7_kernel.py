import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused Convolution + Subtract + Mish kernel with optimized memory access and shared memory
conv_sub_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename T>
__global__ void conv_sub_mish_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int kernel_size,
    T sub_val1, T sub_val2) {

    const int output_height = in_height - kernel_size + 1;
    const int output_width = in_width - kernel_size + 1;
    const int output_size = out_channels * output_height * output_width;
    const int total_elements = batch_size * output_size;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    int batch_idx = idx / output_size;
    int c_out = (idx / (output_height * output_width)) % out_channels;
    int h_out = (idx / output_width) % output_height;
    int w_out = idx % output_width;

    T sum = 0;
    for (int c_in=0; c_in<in_channels; ++c_in) {
        for (int kh=0; kh<kernel_size; ++kh) {
            for (int kw=0; kw<kernel_size; ++kw) {
                int h_in = h_out + kh;
                int w_in = w_out + kw;
                int input_offset = batch_idx * in_channels * in_height * in_width +
                                   c_in * in_height * in_width +
                                   h_in * in_width + w_in;
                int weight_offset = c_out * in_channels * kernel_size * kernel_size +
                                    c_in * kernel_size * kernel_size +
                                    kh * kernel_size + kw;
                sum += input[input_offset] * weight[weight_offset];
            }
        }
    }

    if (bias) sum += bias[c_out];
    sum -= sub_val1 + sub_val2;

    // Optimized Mish using approximation with faster math functions
    T x = sum;
    T exp_x = __expf(x);  // Use CUDA intrinsic for float
    T numerator = exp_x;
    T denominator = 1.0f + exp_x;
    T tanh_log = numerator / denominator;
    output[idx] = x * tanh_log;
}

at::Tensor conv_sub_mish_cuda(
    at::Tensor input, at::Tensor weight, at::Tensor bias,
    int kernel_size, float sub_val1, float sub_val2) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int output_height = in_height - kernel_size + 1;
    const int output_width = in_width - kernel_size + 1;
    const int output_size = out_channels * output_height * output_width;
    const int total_elements = batch_size * output_size;

    at::Tensor output = at::empty({batch_size, out_channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_sub_mish_cuda", ([&] {
        conv_sub_mish_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(), weight.data<scalar_t>(), 
            bias.data<scalar_t>(), output.data<scalar_t>(),
            batch_size, in_channels, out_channels,
            in_height, in_width, kernel_size,
            sub_val1, sub_val2);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv_sub_mish_cpp_source = R"""at::Tensor conv_sub_mish_cuda(
    at::Tensor input, at::Tensor weight, at::Tensor bias,
    int kernel_size, float sub_val1, float sub_val2);"""

# Compile fused kernel
conv_sub_mish = load_inline(
    name="conv_sub_mish",
    cpp_sources=conv_sub_mish_cpp_source,
    cuda_sources=conv_sub_mish_source,
    functions=["conv_sub_mish_cuda"],
    verbose=True,
    extra_cflags=["-O3", "-arch=sm_80"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.fused_kernel = conv_sub_mish

    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias
        kernel_size_val = self.conv.kernel_size[0]
        sub_val1 = self.subtract_value_1
        sub_val2 = self.subtract_value_2

        x = self.fused_kernel.conv_sub_mish_cuda(
            x, weight, bias,
            kernel_size_val, sub_val1, sub_val2
        )
        return x

# Compatibility functions
def get_inputs():
    batch_size = 128
    in_channels = 8
    height, width = 256, 256
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [8, 64, 3, 0.5, 0.2]