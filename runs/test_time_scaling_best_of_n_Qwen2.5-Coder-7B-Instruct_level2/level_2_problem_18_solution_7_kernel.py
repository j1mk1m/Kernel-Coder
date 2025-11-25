import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for various operations
matmul_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_relu_kernel(const float* a, const float* b, float* c, int rows_a, int cols_a, int cols_b) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_a && col < cols_b) {
        float sum = 0.0f;
        for (int k = 0; k < cols_a; ++k) {
            sum += a[row * cols_a + k] * b[k * cols_b + col];
        }
        c[row * cols_b + col] = sum > 0 ? sum : 0.0f;
    }
}

torch::Tensor matmul_relu_cuda(torch::Tensor a, torch::Tensor b) {
    auto rows_a = a.size(0);
    auto cols_a = a.size(1);
    auto cols_b = b.size(1);
    auto c = torch::zeros({rows_a, cols_b}, a.options());

    const int block_size = 32;
    const int grid_x = (cols_b + block_size - 1) / block_size;
    const int grid_y = (rows_a + block_size - 1) / block_size;

    matmul_relu_kernel<<<grid_x, grid_y>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), rows_a, cols_a, cols_b);

    return c;
}
"""

matmul_relu_cpp_source = (
    "torch::Tensor matmul_relu_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for matrix multiplication followed by ReLU
matmul_relu = load_inline(
    name="matmul_relu",
    cpp_sources=matmul_relu_cpp_source,
    cuda_sources=matmul_relu_source,
    functions=["matmul_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)  # (batch_size, out_features)
        x = matmul_relu.matmul_relu_cuda(x, torch.ones_like(x))  # (batch_size, out_features)
        x = torch.sum(x, dim=1, keepdim=True)  # (batch_size, 1)
        x = torch.max(x, dim=1, keepdim=True)[0]  # (batch_size, 1)
        x = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True)  # (batch_size, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True)  # (batch_size, 1)
        return x