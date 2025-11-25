import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GEMM
gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
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

void gemm_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int m = a.size(0);
    int n = b.size(1);
    int k = a.size(1);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);
}
"""

gemm_cpp_source = (
    "void gemm_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c);"
)

# Compile the inline CUDA code for GEMM
gemm = load_inline(
    name="gemm",
    cpp_sources=gemm_cpp_source,
    cuda_sources=gemm_source,
    functions=["gemm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for GroupNorm
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_norm_kernel(const float* x, const float* mean, const float* var, float* y, const float* weight, const float* bias, float eps, int channels, int batch_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < channels * batch_size) {
        int channel = index / batch_size;
        int batch = index % batch_size;
        float normed_x = (x[index] - mean[channel]) / sqrt(var[channel] + eps);
        y[index] = weight[channel] * normed_x + bias[channel];
    }
}

void group_norm_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, torch::Tensor y, float eps) {
    int channels = x.size(1);
    int batch_size = x.size(0);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((channels * batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    group_norm_kernel<<<blocksPerGrid, threadsPerBlock>>>(x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), y.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), eps, channels, batch_size);
}
"""

group_norm_cpp_source = (
    "void group_norm_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, torch::Tensor y, float eps);"
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


# Define the custom CUDA kernel for Multiply
multiply_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void multiply_kernel(const float* x, const float* weight, float* y, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        y[row * n + col] = x[row * n + col] * weight[col];
    }
}

void multiply_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor y) {
    int m = x.size(0);
    int n = x.size(1);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    multiply_kernel<<<blocksPerGrid, threadsPerBlock>>>(x.data_ptr<float>(), weight.data_ptr<float>(), y.data_ptr<float>(), m, n);
}
"""

multiply_cpp_source = (
    "void multiply_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor y);"
)

# Compile the inline CUDA code for Multiply
multiply = load_inline(
    name="multiply",
    cpp_sources=multiply_cpp_source,
    cuda_sources=multiply_source,
    functions=["multiply_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = gemm
        self.group_norm = group_norm
        self.multiply = multiply
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

    def forward(self, x):
        # (batch_size, in_features) -> (batch_size, out_features)
        x = self.gemm.gemm_cuda(x, self.gemm.weight, self.gemm.bias)
        # (batch_size, out_features) -> (batch_size, out_features)
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True)
        x = self.group_norm.group_norm_cuda(x, mean, var, self.group_norm.weight, self.group_norm.bias, 1e-5)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = x * torch.sigmoid(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = self.multiply.multiply_cuda(x, self.multiply_weight, self.multiply_weight)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = x * torch.sigmoid(x)
        return x