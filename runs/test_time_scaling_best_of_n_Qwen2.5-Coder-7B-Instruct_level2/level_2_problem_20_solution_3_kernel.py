import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed convolution
transposed_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void transposed_convolution_kernel(const float* input, const float* weight, const float* bias, float* output, int batch_size, int in_channels, int out_channels, int depth_in, int height_in, int width_in, int depth_out, int height_out, int width_out, int stride, int padding, int output_padding) {
    int n = blockIdx.z;
    int c = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (d >= depth_out) return;

    float sum = 0.0f;
    for (int k = 0; k < in_channels; ++k) {
        for (int i = 0; i <= depth_in + padding - 1; ++i) {
            for (int j = 0; j <= height_in + padding - 1; ++j) {
                for (int l = 0; l <= width_in + padding - 1; ++l) {
                    int di = d + stride * i - padding;
                    int dj = d + stride * j - padding;
                    int dl = d + stride * l - padding;
                    if (di >= 0 && di < depth_out && dj >= 0 && dj < height_out && dl >= 0 && dl < width_out) {
                        int ii = i + output_padding;
                        int jj = j + output_padding;
                        int ll = l + output_padding;
                        sum += input[n * in_channels * depth_in * height_in * width_in + k * depth_in * height_in * width_in + ii * height_in * width_in + jj * width_in + ll] * weight[c * in_channels * depth_out * height_out * width_out + k * depth_out * height_out * width_out + di * height_out * width_out + dj * width_out + dl];
                    }
                }
            }
        }
    }

    output[n * out_channels * depth_out * height_out * width_out + c * depth_out * height_out * width_out + d] = sum + bias[c];
}

torch::Tensor transposed_conv_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int output_padding) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto depth_in = input.size(2);
    auto height_in = input.size(3);
    auto width_in = input.size(4);
    auto depth_out = weight.size(2);
    auto height_out = weight.size(3);
    auto width_out = weight.size(4);

    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (depth_out + block_size - 1) / block_size;

    dim3 grid(num_blocks, out_channels, batch_size);
    dim3 block(block_size);

    transposed_convolution_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth_in, height_in, width_in, depth_out, height_out, width_out, stride, padding, output_padding);

    return output;
}
"""

transposed_conv_cpp_source = (
    "torch::Tensor transposed_conv_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int output_padding);"
)

# Compile the inline CUDA code for transposed convolution
transposed_conv = load_inline(
    name="transposed_conv",
    cpp_sources=transposed_conv_cpp_source,
    cuda_sources=transposed_conv_source,
    functions=["transposed_conv_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for sum operation
sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor sum_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    sum_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

sum_cpp_source = (
    "torch::Tensor sum_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for sum operation
sum_op = load_inline(
    name="sum_op",
    cpp_sources=sum_cpp_source,
    cuda_sources=sum_source,
    functions=["sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for residual add operation
residual_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void residual_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor residual_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    residual_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

residual_add_cpp_source = (
    "torch::Tensor residual_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for residual add operation
residual_add = load_inline(
    name="residual_add",
    cpp_sources=residual_add_cpp_source,
    cuda_sources=residual_add_source,
    functions=["residual_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for multiplication operation
multiplication_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void multiplication_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

torch::Tensor multiplication_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    multiplication_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

multiplication_cpp_source = (
    "torch::Tensor multiplication_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for multiplication operation
multiplication = load_inline(
    name="multiplication",
    cpp_sources=multiplication_cpp_source,
    cuda_sources=multiplication_source,
    functions=["multiplication_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for final residual add operation
final_residual_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void final_residual_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor final_residual_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    final_residual_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

final_residual_add_cpp_source = (
    "torch::Tensor final_residual_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for final residual add operation
final_residual_add = load_inline(
    name="final_residual_add",
    cpp_sources=final_residual_add_cpp_source,
    cuda_sources=final_residual_add_source,
    functions=["final_residual_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.transposed_conv = transposed_conv
        self.sum_op = sum_op
        self.residual_add = residual_add
        self.multiplication = multiplication
        self.final_residual_add = final_residual_add
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.transposed_conv.transposed_conv_cuda(x, self.weight, self.bias, stride=self.stride, padding=self.padding, output_padding=self.output_padding)
        original_x = x.clone().detach()
        x = self.sum_op.sum_cuda(x, self.bias)
        x = self.residual_add.residual_add_cuda(x, original_x)
        x = self.multiplication.multiplication_cuda(x, original_x)
        x = self.final_residual_add.final_residual_add_cuda(x, original_x)
        return x