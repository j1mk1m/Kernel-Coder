import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication and Leaky ReLU
matmul_leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_leaky_relu_kernel(const float* a, const float* b, float* c, float* d, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }

        // Leaky ReLU activation
        d[row * n + col] = sum > 0 ? sum : sum * 0.01f;
    }
}

torch::Tensor matmul_leaky_relu_cuda(torch::Tensor a, torch::Tensor b) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);

    auto c = torch::zeros({m, n}, a.options());
    auto d = torch::zeros_like(c);

    const int block_size = 32;
    const int num_blocks_x = (n + block_size - 1) / block_size;
    const int num_blocks_y = (m + block_size - 1) / block_size;

    matmul_leaky_relu_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), d.data_ptr<float>(), m, n, k);

    return d;
}
"""

matmul_leaky_relu_cpp_source = (
    "torch::Tensor matmul_leaky_relu_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for matrix multiplication and Leaky ReLU
matmul_leaky_relu = load_inline(
    name="matmul_leaky_relu",
    cpp_sources=matmul_leaky_relu_cpp_source,
    cuda_sources=matmul_leaky_relu_source,
    functions=["matmul_leaky_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = matmul_leaky_relu
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.sum = lambda x: x + x

    def forward(self, x):
        x = self.fc(x, torch.ones_like(x))  # Placeholder for the second argument of matmul_leaky_relu
        x = self.gn(x)
        x = self.sum(x)
        return x