import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for convolution and activations
fused_conv_act_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <mma.h>

#define CUDA_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void fused_conv_act_kernel(
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> bias,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> act_bias,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> output,
    int in_channels, int out_channels, int kernel_size,
    int pad_d, int pad_h, int pad_w,
    int stride_d, int stride_h, int stride_w) {

    const int n = blockIdx.z;
    const int c_out = blockIdx.y;
    const int d_out = threadIdx.z;
    const int h_out = threadIdx.y;
    const int w_out = threadIdx.x;

    scalar_t val = bias[c_out][0][0][0];

    for (int k_d = 0; k_d < kernel_size; ++k_d) {
        for (int k_h = 0; k_h < kernel_size; ++k_h) {
            for (int k_w = 0; k_w < kernel_size; ++k_w) {
                const int d_in = d_out * stride_d - pad_d + k_d;
                const int h_in = h_out * stride_h - pad_h + k_h;
                const int w_in = w_out * stride_w - pad_w + k_w;

                if (d_in >= 0 && d_in < input.size(2) &&
                    h_in >= 0 && h_in < input.size(3) &&
                    w_in >= 0 && w_in < input.size(4)) {
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        val += weight[c_out][c_in][k_d][k_h][k_w] *
                               input[n][c_in][d_in][h_in][w_in];
                    }
                }
            }
        }
    }

    // Apply fused activations
    val = fmax(val, 0.f); // ReLU
    val = val >= 0 ? val : 0.01f * val; // LeakyReLU
    // GELU approximation using tanh
    val = val * 0.5f * (1.0f + tanhf((M_SQRT1_2 * (val + 0.044715f * val * val * val))));
    val = 1.0f / (1.0f + exp(-val)); // Sigmoid
    val += act_bias[c_out][0][0][0]; // Add activation bias

    output[n][c_out][d_out][h_out][w_out] = val;
}

torch::Tensor fused_conv_act_cuda(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias,
                                  torch::Tensor act_bias,
                                  int kernel_size,
                                  int pad_d, int pad_h, int pad_w,
                                  int stride_d, int stride_h, int stride_w) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(0);
    const auto input_depth = input.size(2);
    const auto input_height = input.size(3);
    const auto input_width = input.size(4);

    const int output_depth = (input_depth + 2 * pad_d - kernel_size) / stride_d + 1;
    const int output_height = (input_height + 2 * pad_h - kernel_size) / stride_h + 1;
    const int output_width = (input_width + 2 * pad_w - kernel_size) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, output_depth,
                               output_height, output_width}, input.options());

    const int threads = 256;
    dim3 blocks(batch_size, out_channels,
               output_depth * output_height * output_width / threads);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_conv_act_cuda", ([&] {
        fused_conv_act_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            bias.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            act_bias.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            in_channels, out_channels, kernel_size,
            pad_d, pad_h, pad_w,
            stride_d, stride_h, stride_w);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

fused_conv_act_cpp_source = """
torch::Tensor fused_conv_act_cuda(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias,
                                  torch::Tensor act_bias,
                                  int kernel_size,
                                  int pad_d, int pad_h, int pad_w,
                                  int stride_d, int stride_h, int stride_w);
"""

# Compile the fused kernel
fused_conv_act_cuda = load_inline(
    name="fused_conv_act_cuda",
    cpp_sources=fused_conv_act_cpp_source,
    cuda_sources=fused_conv_act_cuda_source,
    functions=["fused_conv_act_cuda"],
    verbose=True,
    extra_cuda_cflags=['-arch=sm_86'],
    extra_cflags=['-O3', '-std=c++17'],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.kernel_size = kernel_size
        self.padding = self.conv.padding
        self.stride = self.conv.stride

    def forward(self, x):
        # Extract convolution parameters
        weight = self.conv.weight
        bias = self.conv.bias if self.conv.bias is not None else torch.zeros_like(self.bias)
        pad_d, pad_h, pad_w = self.padding
        stride_d, stride_h, stride_w = self.stride

        # Call fused kernel
        return fused_conv_act_cuda(
            x, weight, bias, self.bias,
            self.kernel_size, pad_d, pad_h, pad_w,
            stride_d, stride_h, stride_w
        )