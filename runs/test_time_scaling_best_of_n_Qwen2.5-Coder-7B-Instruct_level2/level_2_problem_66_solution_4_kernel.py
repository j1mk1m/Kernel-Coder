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

    matmul_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

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

# Define the custom CUDA kernel for applying dropout
dropout_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void dropout_kernel(const float* x, float* y, float p, curandState* state, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curand_init(idx, 0, 0, &state[idx]);
        float rand_val = curand_uniform(&state[idx]);
        if (rand_val > p) {
            y[idx] = x[idx] / (1.0f - p);
        } else {
            y[idx] = 0.0f;
        }
    }
}

torch::Tensor dropout_cuda(torch::Tensor x, float p) {
    auto size = x.numel();
    auto y = torch::zeros_like(x);
    auto state = torch::empty(size, x.options()).type_as(curandState());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    dropout_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), p, state.data_ptr<curandState>(), size);

    return y;
}
"""

dropout_cpp_source = (
    "torch::Tensor dropout_cuda(torch::Tensor x, float p);"
)

# Compile the inline CUDA code for dropout
dropout = load_inline(
    name="dropout",
    cpp_sources=dropout_cpp_source,
    cuda_sources=dropout_source,
    functions=["dropout_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for applying softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax_kernel(const float* x, float* y, int batch_size, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < n; ++i) {
            if (x[row * n + i] > max_val) {
                max_val = x[row * n + i];
            }
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum_exp += exp(x[row * n + i] - max_val);
        }

        for (int i = 0; i < n; ++i) {
            y[row * n + i] = exp(x[row * n + i] - max_val) / sum_exp;
        }
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto n = x.size(1);
    auto y = torch::zeros_like(x);

    const int block_size = 16;
    const int num_blocks_x = (n + block_size - 1) / block_size;
    const int num_blocks_y = (batch_size + block_size - 1) / block_size;

    softmax_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, n);

    return y;
}
"""

softmax_cpp_source = (
    "torch::Tensor softmax_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for softmax
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
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = matmul
        self.dropout = dropout
        self.softmax = softmax

    def forward(self, x):
        x = self.matmul.matmul_cuda(x)
        x = self.dropout.dropout_cuda(x, self.dropout.p)
        x = self.softmax.softmax_cuda(x)
        return x