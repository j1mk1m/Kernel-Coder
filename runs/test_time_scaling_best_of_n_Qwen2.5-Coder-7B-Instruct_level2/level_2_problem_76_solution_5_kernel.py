import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batched GEMM
batched_gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batched_gemm_relu_kernel(const float* a, const float* b, float* c, float* d, int batch_size, int m, int n, int k) {
    int b_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b_idx < batch_size && row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[b_idx * m * k + row * k + i] * b[b_idx * k * n + i * n + col];
        }
        c[b_idx * m * n + row * n + col] = sum;
        d[b_idx * m * n + row * n + col] = fmaxf(sum, 0.0f);
    }
}

void batched_gemm_relu_cuda(float* a, float* b, float* c, float* d, int batch_size, int m, int n, int k) {
    const int block_size = 32;
    dim3 grid_dim((n + block_size - 1) / block_size, (m + block_size - 1) / block_size, batch_size);
    dim3 block_dim(block_size, block_size);

    batched_gemm_relu_kernel<<<grid_dim, block_dim>>>(a, b, c, d, batch_size, m, n, k);
}
"""

batched_gemm_cpp_source = (
    "void batched_gemm_relu_cuda(float* a, float* b, float* c, float* d, int batch_size, int m, int n, int k);"
)

# Compile the inline CUDA code for batched GEMM with ReLU
batched_gemm_relu = load_inline(
    name="batched_gemm_relu",
    cpp_sources=batched_gemm_cpp_source,
    cuda_sources=batched_gemm_source,
    functions=["batched_gemm_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_shape = bias_shape
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        batch_size = x.size(0)
        y = torch.zeros((batch_size, self.out_features)).cuda()

        self.batched_gemm_relu.batched_gemm_relu_cuda(x.contiguous().data_ptr(), self.weight.contiguous().data_ptr(), y.data_ptr(), y.data_ptr(), batch_size, self.out_features, self.in_features, self.in_features)

        return y + self.bias