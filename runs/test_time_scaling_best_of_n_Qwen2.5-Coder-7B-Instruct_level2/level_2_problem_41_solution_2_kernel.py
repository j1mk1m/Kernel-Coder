import torch
import torch.nn as nn
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

torch::Tensor gemm_cuda(torch::Tensor a, torch::Tensor b) {
    int m = a.size(0);
    int n = b.size(1);
    int k = a.size(1);

    auto c = torch::zeros({m, n}, a.options());

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

    return c;
}
"""

gemm_cpp_source = (
    "torch::Tensor gemm_cuda(torch::Tensor a, torch::Tensor b);"
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

# Define the custom CUDA kernel for BatchNorm
batchnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batchnorm_kernel(const float* x, const float* mean, const float* var, float* y, int n, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        y[idx] = (x[idx] - mean[0]) / sqrt(var[0] + eps);
    }
}

torch::Tensor batchnorm_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, float eps) {
    int n = x.size(0);

    auto y = torch::zeros_like(x);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    batchnorm_kernel<<<blocksPerGrid, threadsPerBlock>>>(x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), y.data_ptr<float>(), n, eps);

    return y;
}
"""

batchnorm_cpp_source = (
    "torch::Tensor batchnorm_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, float eps);"
)

# Compile the inline CUDA code for BatchNorm
batchnorm = load_inline(
    name="batchnorm",
    cpp_sources=batchnorm_cpp_source,
    cuda_sources=batchnorm_source,
    functions=["batchnorm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for GELU
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        y[idx] = 0.5 * x[idx] * (1.0 + tanh(sqrt(2.0 / M_PI) * (x[idx] + 0.044715 * x[idx] * x[idx] * x[idx])));
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    int n = x.size(0);

    auto y = torch::zeros_like(x);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}
"""

gelu_cpp_source = (
    "torch::Tensor gelu_cuda(torch::Tensor x);"
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

# Define the custom CUDA kernel for ReLU
relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        y[idx] = fmaxf(x[idx], 0.0f);
    }
}

torch::Tensor relu_cuda(torch::Tensor x) {
    int n = x.size(0);

    auto y = torch::zeros_like(x);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}
"""

relu_cpp_source = (
    "torch::Tensor relu_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for ReLU
relu = load_inline(
    name="relu",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = gemm
        self.batch_norm = batchnorm
        self.gelu = gelu
        self.relu = relu

    def forward(self, x):
        x = self.gemm.gemm_cuda(x, self.gemm.weight.t())
        x = self.batch_norm.batchnorm_cuda(x, self.batch_norm.running_mean, self.batch_norm.running_var, self.batch_norm.eps)
        x = self.gelu.gelu_cuda(x)
        x = self.relu.relu_cuda(x)
        return x