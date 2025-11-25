import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);

    auto c = torch::zeros({m, n}, a.options());

    const int block_size = 16;
    const int num_blocks_x = (n + block_size - 1) / block_size;
    const int num_blocks_y = (m + block_size - 1) / block_size;

    dim3 grid_dim(num_blocks_x, num_blocks_y);
    dim3 block_dim(block_size, block_size);

    matmul_kernel<<<grid_dim, block_dim>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

    return c;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for matrix multiplication
matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Swish activation
swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] *= 1.0f / (1.0f + exp(-x[idx]));
    }
}

void swish_cuda(torch::Tensor x) {
    auto size = x.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), size);
}
"""

swish_cpp_source = (
    "void swish_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for Swish activation
swish = load_inline(
    name="swish",
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_source,
    functions=["swish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for bias addition
bias_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bias_add_kernel(const float* x, const float* bias, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = x[idx] + bias[0];
    }
}

torch::Tensor bias_add_cuda(torch::Tensor x, torch::Tensor bias) {
    auto size = x.numel();

    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    bias_add_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

bias_add_cpp_source = (
    "torch::Tensor bias_add_cuda(torch::Tensor x, torch::Tensor bias);"
)

# Compile the inline CUDA code for bias addition
bias_add = load_inline(
    name="bias_add",
    cpp_sources=bias_add_cpp_source,
    cuda_sources=bias_add_source,
    functions=["bias_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for GroupNorm
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_norm_kernel(const float* x, const float* gamma, const float* beta, float* y, int num_groups, int channels, int height, int width) {
    int g = blockIdx.z;
    int ch = blockIdx.y * blockDim.y + threadIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;

    if (g < num_groups && ch < channels && hw < height * width) {
        int idx = g * channels * height * width + ch * height * width + hw;
        float mean = 0.0f;
        float var = 0.0f;

        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int local_idx = idx + h * width + w;
                mean += x[local_idx];
                var += x[local_idx] * x[local_idx];
            }
        }

        mean /= (height * width);
        var /= (height * width) - 1.0f;

        y[idx] = gamma[ch] * (x[idx] - mean) / sqrt(var + 1e-5f) + beta[ch];
    }
}

torch::Tensor group_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int num_groups) {
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);

    auto y = torch::zeros_like(x);

    const int block_size = 16;
    const int num_blocks_x = (width + block_size - 1) / block_size;
    const int num_blocks_y = (channels + block_size - 1) / block_size;
    const int num_blocks_z = (num_groups + block_size - 1) / block_size;

    dim3 grid_dim(num_blocks_x, num_blocks_y, num_blocks_z);
    dim3 block_dim(block_size, block_size, 1);

    group_norm_kernel<<<grid_dim, block_dim>>>(x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), y.data_ptr<float>(), num_groups, channels, height, width);

    return y;
}
"""

group_norm_cpp_source = (
    "torch::Tensor group_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int num_groups);"
)

# Compile the inline CUDA code for GroupNorm
group_norm = load_inline(
    name="group_norm",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = matmul
        self.swish = swish
        self.bias_add = bias_add
        self.group_norm = group_norm

    def forward(self, x):
        x = self.matmul.matmul_cuda(x.view(x.size(0), -1), self.matmul.weight)
        x = self.swish.swish_cuda(x)
        x = self.bias_add.bias_add_cuda(x, self.bias)
        x = self.group_norm.group_norm_cuda(x, self.group_norm.weight, self.group_norm.bias, self.group_norm.num_groups)
        return x