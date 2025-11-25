import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication followed by ReLU
matmul_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_relu_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = max(sum, 0.0f);
    }
}

torch::Tensor matmul_relu_cuda(torch::Tensor a, torch::Tensor b) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);
    auto c = torch::zeros({m, n}, a.options());

    const int block_size = 32;
    dim3 grid_dim((n + block_size - 1) / block_size, (m + block_size - 1) / block_size);
    dim3 block_dim(block_size, block_size);

    matmul_relu_kernel<<<grid_dim, block_dim>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

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
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.matmul_relu = matmul_relu
        self.divisor = divisor

    def forward(self, x):
        x = self.matmul_relu.matmul_relu_cuda(x, self.linear.weight.t())
        x = x / self.divisor
        return x