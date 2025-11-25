import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
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

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto N = B.size(1);
    auto K = A.size(1);
    auto C = torch::zeros({M, N}, A.options());

    const int block_size = 256;
    const int grid_x = (N + block_size - 1) / block_size;
    const int grid_y = (M + block_size - 1) / block_size;

    matmul_kernel<<<grid_y, grid_x>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
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


# Define the custom CUDA kernel for GELU activation
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(float* X, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        X[idx] = 0.5f * X[idx] * (1.0f + tanh(sqrt(2.0f / M_PI) * (X[idx] + 0.044715f * X[idx] * X[idx] * X[idx])));
    }
}

void gelu_cuda(torch::Tensor X) {
    auto size = X.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(X.data_ptr<float>(), size);
}
"""

gelu_cpp_source = (
    "void gelu_cuda(torch::Tensor X);"
)

# Compile the inline CUDA code for GELU activation
gelu = load_inline(
    name="gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Softmax operation
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(float* X, int size) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        sdata[tid] = exp(X[i]);
        __syncthreads();

        int s = blockDim.x;
        while (s > 0) {
            if (tid < s / 2) {
                sdata[tid] += sdata[tid + s / 2];
            }
            __syncthreads();
            s /= 2;
        }

        if (tid == 0) {
            X[blockIdx.x * blockDim.x] = sdata[0];
        }
    }
}

void softmax_cuda(torch::Tensor X) {
    auto size = X.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    softmax_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(X.data_ptr<float>(), size);
}
"""

softmax_cpp_source = (
    "void softmax_cuda(torch::Tensor X);"
)

# Compile the inline CUDA code for Softmax operation
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
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        gelu_cuda(x)
        softmax_cuda(x)
        return x