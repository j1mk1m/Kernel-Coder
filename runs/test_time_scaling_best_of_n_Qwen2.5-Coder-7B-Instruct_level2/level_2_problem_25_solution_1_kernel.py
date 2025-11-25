import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int h_out = blockIdx.x * blockDim.y + threadIdx.y;
    int w_out = blockIdx.w * blockDim.z + threadIdx.z;

    if (h_out >= height || w_out >= width) return;

    float sum = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_out + kh - kernel_size / 2;
                int w_in = w_out + kw - kernel_size / 2;
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int i_idx = n * in_channels * height * width + c_in * height * width + h_in * width + w_in;
                    int k_idx = c_out * in_channels * kernel_size * kernel_size + c_in * kernel_size * kernel_size + kh * kernel_size + kw;
                    sum += input[i_idx] * weight[k_idx];
                }
            }
        }
    }

    int o_idx = n * out_channels * height * width + c_out * height * width + h_out * width + w_out;
    output[o_idx] = sum;
}
"""

convolution_cpp_source = (
    "void convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size);"
)

# Compile the inline CUDA code for convolution
convolution = load_inline(
    name="convolution",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["convolution_kernel"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for minimum operation along the channel dimension
min_operation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void min_operation_kernel(const float* input, float* output, int batch_size, int channels, int height, int width) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    int h = blockIdx.z * blockDim.y + threadIdx.y;
    int w = blockIdx.w * blockDim.z + threadIdx.z;

    if (h >= height || w >= width) return;

    float min_val = input[n * channels * height * width + c * height * width + h * width + w];
    for (int ch = 0; ch < channels; ++ch) {
        int i_idx = n * channels * height * width + ch * height * width + h * width + w;
        if (input[i_idx] < min_val) {
            min_val = input[i_idx];
        }
    }

    int o_idx = n * channels * height * width + c * height * width + h * width + w;
    output[o_idx] = min_val;
}
"""

min_operation_cpp_source = (
    "void min_operation_kernel(const float* input, float* output, int batch_size, int channels, int height, int width);"
)

# Compile the inline CUDA code for minimum operation
min_operation = load_inline(
    name="min_operation",
    cpp_sources=min_operation_cpp_source,
    cuda_sources=min_operation_source,
    functions=["min_operation_kernel"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for combined Tanh operations
tanh_combined_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_combined_kernel(const float* input, float* output, int batch_size, int channels, int height, int width) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    int h = blockIdx.z * blockDim.y + threadIdx.y;
    int w = blockIdx.w * blockDim.z + threadIdx.z;

    if (h >= height || w >= width) return;

    float val = input[n * channels * height * width + c * height * width + h * width + w];
    output[n * channels * height * width + c * height * width + h * width + w] = tanh(val);
}
"""

tanh_combined_cpp_source = (
    "void tanh_combined_kernel(const float* input, float* output, int batch_size, int channels, int height, int width);"
)

# Compile the inline CUDA code for combined Tanh operations
tanh_combined = load_inline(
    name="tanh_combined",
    cpp_sources=tanh_combined_cpp_source,
    cuda_sources=tanh_combined_source,
    functions=["tanh_combined_kernel"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.min_operation = min_operation
        self.tanh_combined = tanh_combined

    def forward(self, x):
        x = self.conv.convolution_kernel(x.contiguous().data_ptr(), self.weight.data_ptr(), x.contiguous().data_ptr(), x.size(0), x.size(1), x.size(2), x.size(3), x.size(4))
        x = self.min_operation.min_operation_kernel(x.contiguous().data_ptr(), x.contiguous().data_ptr(), x.size(0), x.size(1), x.size(2), x.size(3))
        x = self.tanh_combined.tanh_combined_kernel(x.contiguous().data_ptr(), x.contiguous().data_ptr(), x.size(0), x.size(1), x.size(2), x.size(3))
        x = self.tanh_combined.tanh_combined_kernel(x.contiguous().data_ptr(), x.contiguous().data_ptr(), x.size(0), x.size(1), x.size(2), x.size(3))
        return x