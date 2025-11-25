import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication followed by GELU
matmul_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void matmul_gelu_kernel(const float* a, const float* b, float* out, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) return;

    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        sum += a[row * k + i] * b[i * n + col];
    }

    // GELU approximation using Taylor series expansion
    float gelu_approx = 0.5f * sum * (1.0f + tanh(sqrt(2.0f / M_PI) * (sum + 0.044715f * sum * sum * sum)));

    out[row * n + col] = gelu_approx;
}

torch::Tensor matmul_gelu_cuda(torch::Tensor a, torch::Tensor b) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);
    auto out = torch::zeros({m, n}, a.options());

    const int block_size_x = 32;
    const int block_size_y = 32;
    const int num_blocks_x = (n + block_size_x - 1) / block_size_x;
    const int num_blocks_y = (m + block_size_y - 1) / block_size_y;

    matmul_gelu_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size_x, block_size_y)>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), m, n, k);

    return out;
}
"""

matmul_gelu_cpp_source = (
    "torch::Tensor matmul_gelu_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for matrix multiplication followed by GELU
matmul_gelu = load_inline(
    name="matmul_gelu",
    cpp_sources=matmul_gelu_cpp_source,
    cuda_sources=matmul_gelu_source,
    functions=["matmul_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that uses custom CUDA kernels for matrix multiplication followed by GELU.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.matmul_gelu = matmul_gelu

    def forward(self, x):
        x = self.matmul_gelu.matmul_gelu_cuda(x, self.weight)
        x = torch.nn.functional.softmax(x, dim=1)
        return x