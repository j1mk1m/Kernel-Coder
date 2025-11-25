import torch
import torch.nn as nn
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

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((n + threads_per_block.x - 1) / threads_per_block.x, (m + threads_per_block.y - 1) / threads_per_block.y);

    matmul_kernel<<<blocks_per_grid, threads_per_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

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


# Define the custom CUDA kernel for batch normalization
bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bn_kernel(const float* x, const float* mean, const float* var, const float* gamma, const float* beta, float* y, int n, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float inv_var = 1.0f / sqrt(var[idx] + eps);
        y[idx] = gamma[idx] * (x[idx] - mean[idx]) * inv_var + beta[idx];
    }
}

torch::Tensor bn_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, torch::Tensor gamma, torch::Tensor beta, float eps) {
    auto n = x.numel();
    auto y = torch::zeros_like(x);

    dim3 threads_per_block(256);
    dim3 blocks_per_grid((n + threads_per_block.x - 1) / threads_per_block.x);

    bn_kernel<<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), y.data_ptr<float>(), n, eps);

    return y;
}
"""

bn_cpp_source = (
    "torch::Tensor bn_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, torch::Tensor gamma, torch::Tensor beta, float eps);"
)

# Compile the inline CUDA code for batch normalization
bn = load_inline(
    name="bn",
    cpp_sources=bn_cpp_source,
    cuda_sources=bn_source,
    functions=["bn_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for bias addition
bias_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bias_add_kernel(const float* x, const float* bias, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        y[idx] = x[idx] + bias[0];
    }
}

torch::Tensor bias_add_cuda(torch::Tensor x, torch::Tensor bias) {
    auto n = x.numel();
    auto y = torch::zeros_like(x);

    dim3 threads_per_block(256);
    dim3 blocks_per_grid((n + threads_per_block.x - 1) / threads_per_block.x);

    bias_add_kernel<<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(), n);

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


# Define the custom CUDA kernel for division
division_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void division_kernel(const float* x, float* y, int n, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        y[idx] = x[idx] / divisor;
    }
}

torch::Tensor division_cuda(torch::Tensor x, float divisor) {
    auto n = x.numel();
    auto y = torch::zeros_like(x);

    dim3 threads_per_block(256);
    dim3 blocks_per_grid((n + threads_per_block.x - 1) / threads_per_block.x);

    division_kernel<<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), y.data_ptr<float>(), n, divisor);

    return y;
}
"""

division_cpp_source = (
    "torch::Tensor division_cuda(torch::Tensor x, float divisor);"
)

# Compile the inline CUDA code for division
division = load_inline(
    name="division",
    cpp_sources=division_cpp_source,
    cuda_sources=division_source,
    functions=["division_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Swish activation
swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        y[idx] = x[idx] * sigmoid(x[idx]);
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    auto n = x.numel();
    auto y = torch::zeros_like(x);

    dim3 threads_per_block(256);
    dim3 blocks_per_grid((n + threads_per_block.x - 1) / threads_per_block.x);

    swish_kernel<<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}
"""

swish_cpp_source = (
    "torch::Tensor swish_cuda(torch::Tensor x);"
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


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = matmul
        self.bn = bn
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value
        self.swish = swish

    def forward(self, x):
        x = self.matmul.matmul_cuda(x)
        x = self.bn.bn_cuda(x, self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias, self.bn.eps)
        x = self.bias_add.bias_add_cuda(x, self.bias)
        x = self.division.division_cuda(x, self.divide_value)
        x = self.swish.swish_cuda(x)
        return x