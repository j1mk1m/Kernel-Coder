import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Helper function to perform convolution
__device__ void convolve(float* out, const float* in, const float* weight, int in_height, int in_width, int out_height, int out_width, int channels, int kernel_size) {
    int out_idx = blockIdx.y * out_width + blockIdx.x;
    if (out_idx >= out_height * out_width) return;

    int channel_offset = out_idx % channels;
    out[out_idx] = 0.0f;
    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            int in_idx = ((out_idx / channels) * kernel_size + ky) * in_width + (channel_offset * kernel_size + kx);
            out[out_idx] += in[in_idx] * weight[ky * kernel_size + kx];
        }
    }
}

torch::Tensor convolution_cuda(torch::Tensor in, torch::Tensor weight) {
    auto in_height = in.size(2);
    auto in_width = in.size(3);
    auto out_height = (in_height - weight.size(2) + 1) / 2;
    auto out_width = (in_width - weight.size(3) + 1) / 2;
    auto channels = in.size(1);
    auto kernel_size = weight.size(2);

    auto out = torch::zeros({in.size(0), channels, out_height, out_width}, in.options());

    dim3 blocks(out_width, out_height);
    dim3 threads(kernel_size, kernel_size);

    convolution<<<dim3(blocks), dim3(threads)>>>(out.data_ptr<float>(), in.data_ptr<float>(), weight.data_ptr<float>(), in_height, in_width, out_height, out_width, channels, kernel_size);

    return out;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor in, torch::Tensor weight);"
)

# Compile the inline CUDA code for convolution
convolution = load_inline(
    name="convolution",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for multiplication by a learnable scalar
multiplication_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void multiply_kernel(const float* in, const float* multiplier, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = in[idx] * multiplier[0];
    }
}

torch::Tensor multiply_cuda(torch::Tensor in, torch::Tensor multiplier) {
    auto size = in.numel();
    auto out = torch::zeros_like(in);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    multiply_kernel<<<num_blocks, block_size>>>(in.data_ptr<float>(), multiplier.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

multiplication_cpp_source = (
    "torch::Tensor multiply_cuda(torch::Tensor in, torch::Tensor multiplier);"
)

# Compile the inline CUDA code for multiplication by a learnable scalar
multiply = load_inline(
    name="multiply",
    cpp_sources=multiplication_cpp_source,
    cuda_sources=multiplication_source,
    functions=["multiply_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for LeakyReLU
leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* in, float* out, int size, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = in[idx] > 0 ? in[idx] : negative_slope * in[idx];
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor in, float negative_slope) {
    auto size = in.numel();
    auto out = torch::zeros_like(in);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    leaky_relu_kernel<<<num_blocks, block_size>>>(in.data_ptr<float>(), out.data_ptr<float>(), size, negative_slope);

    return out;
}
"""

leaky_relu_cpp_source = (
    "torch::Tensor leaky_relu_cuda(torch::Tensor in, float negative_slope);"
)

# Compile the inline CUDA code for LeakyReLU
leaky_relu = load_inline(
    name="leaky_relu",
    cpp_sources=leaky_relu_cpp_source,
    cuda_sources=leaky_relu_source,
    functions=["leaky_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for GELU
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = 0.5f * in[idx] * (1.0f + tanh(sqrt(2.0f / M_PI) * (in[idx] + 0.044715f * in[idx] * in[idx] * in[idx])));
    }
}

torch::Tensor gelu_cuda(torch::Tensor in) {
    auto size = in.numel();
    auto out = torch::zeros_like(in);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(in.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

gelu_cpp_source = (
    "torch::Tensor gelu_cuda(torch::Tensor in);"
)

# Compile the inline CUDA code for GELU
gelu = load_inline(
    name="gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.multiply = multiply
        self.leaky_relu = leaky_relu
        self.gelu = gelu

    def forward(self, x):
        x = self.conv.convolution_cuda(x, self.weight)
        x = self.multiply.multiply_cuda(x, self.multiplier)
        x = self.leaky_relu.leaky_relu_cuda(x, 0.01)
        x = self.gelu.gelu_cuda(x)
        return x


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]