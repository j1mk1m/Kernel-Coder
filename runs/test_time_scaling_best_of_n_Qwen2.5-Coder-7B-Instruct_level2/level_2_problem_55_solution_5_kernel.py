import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication and relu
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
        c[row * n + col] = sum > 0 ? sum : 0.0f;
    }
}

torch::Tensor matmul_relu_cuda(torch::Tensor a, torch::Tensor b) {
    int m = a.size(0);
    int n = b.size(1);
    int k = a.size(1);
    auto c = torch::zeros({m, n}, a.options());

    const int block_size = 16;
    const int num_blocks_x = (n + block_size - 1) / block_size;
    const int num_blocks_y = (m + block_size - 1) / block_size;

    matmul_relu_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

    return c;
}
"""

matmul_relu_cpp_source = (
    "torch::Tensor matmul_relu_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for matrix multiplication and relu
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
    """
    Optimized model using custom CUDA operators for matrix multiplication and relu.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul_relu = matmul_relu

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul_relu.matmul_relu_cuda(x, x)
        x = F.max_pool1d(x.unsqueeze(1), kernel_size).squeeze(1)
        x = torch.sum(x, dim=1)
        x = x * self.scale_factor
        return x