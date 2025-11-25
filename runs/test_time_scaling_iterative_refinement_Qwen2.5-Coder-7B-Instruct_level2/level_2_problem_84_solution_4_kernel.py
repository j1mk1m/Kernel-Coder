import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Matrix Multiplication (GEMM)
gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);
    auto C = torch::zeros({M, N}, A.options());

    const int block_size = 16;
    dim3 blocks((N + block_size - 1) / block_size, (M + block_size - 1) / block_size);
    dim3 threads(block_size, block_size);

    gemm_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

gemm_cpp_source = (
    "torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B);"
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


# Define the custom CUDA kernel for Batch Normalization
batch_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batch_norm_kernel(const float* X, float* Y, const float* mean, const float* var, const float* gamma, const float* beta, float eps, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float inv_var = 1.0f / sqrt(var[idx] + eps);
        Y[idx] = gamma[idx] * (X[idx] - mean[idx]) * inv_var + beta[idx];
    }
}

torch::Tensor batch_norm_cuda(torch::Tensor X, torch::Tensor mean, torch::Tensor var, torch::Tensor gamma, torch::Tensor beta, float eps) {
    int N = X.numel();
    auto Y = torch::zeros_like(X);

    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;

    batch_norm_kernel<<<num_blocks, block_size>>>(X.data_ptr<float>(), Y.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), eps, N);

    return Y;
}
"""

batch_norm_cpp_source = (
    "torch::Tensor batch_norm_cuda(torch::Tensor X, torch::Tensor mean, torch::Tensor var, torch::Tensor gamma, torch::Tensor beta, float eps);"
)

# Compile the inline CUDA code for Batch Normalization
batch_norm = load_inline(
    name="batch_norm",
    cpp_sources=batch_norm_cpp_source,
    cuda_sources=batch_norm_source,
    functions=["batch_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Scaling
scaling_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scaling_kernel(const float* X, float* Y, const float* scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        Y[idx] = X[idx] * scale[0];
    }
}

torch::Tensor scaling_cuda(torch::Tensor X, torch::Tensor scale) {
    int N = X.numel();
    auto Y = torch::zeros_like(X);

    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;

    scaling_kernel<<<num_blocks, block_size>>>(X.data_ptr<float>(), Y.data_ptr<float>(), scale.data_ptr<float>(), N);

    return Y;
}
"""

scaling_cpp_source = (
    "torch::Tensor scaling_cuda(torch::Tensor X, torch::Tensor scale);"
)

# Compile the inline CUDA code for Scaling
scaling = load_inline(
    name="scaling",
    cpp_sources=scaling_cpp_source,
    cuda_sources=scaling_source,
    functions=["scaling_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* X, float* Y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float max_val = X[idx];
        for (int i = 0; i < N; ++i) {
            if (X[i] > max_val) {
                max_val = X[i];
            }
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum_exp += exp(X[i] - max_val);
        }

        Y[idx] = exp(X[idx] - max_val) / sum_exp;
    }
}

torch::Tensor softmax_cuda(torch::Tensor X) {
    int N = X.numel();
    auto Y = torch::zeros_like(X);

    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;

    softmax_kernel<<<num_blocks, block_size>>>(X.data_ptr<float>(), Y.data_ptr<float>(), N);

    return Y;
}
"""

softmax_cpp_source = (
    "torch::Tensor softmax_cuda(torch::Tensor X);"
)

# Compile the inline CUDA code for Softmax
softmax = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = gemm
        self.bn = batch_norm
        self.scale = scaling
        self.softmax = softmax

    def forward(self, x):
        x = self.gemm.gemm_cuda(x, x)
        x = self.bn.batch_norm_cuda(x, x, x, x, x, bn_eps)
        x = self.scale.scaling_cuda(x, x)
        x = self.softmax.softmax_cuda(x)
        return x