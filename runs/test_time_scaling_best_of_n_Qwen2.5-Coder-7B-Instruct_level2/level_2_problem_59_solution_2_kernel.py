import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication followed by Swish activation
matmul_swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_swish_kernel(const float* a, const float* b, float* out, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }

        out[row * n + col] = sum * __fdividef(sum, fmaxf(sum, 1e-20));
    }
}

torch::Tensor matmul_swish_cuda(torch::Tensor a, torch::Tensor b, float scaling_factor) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);
    auto out = torch::zeros({m, n}, a.options());

    const int block_size = 16;
    const int num_blocks_x = (n + block_size - 1) / block_size;
    const int num_blocks_y = (m + block_size - 1) / block_size;

    matmul_swish_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), m, n, k);

    return out * scaling_factor;
}
"""

matmul_swish_cpp_source = (
    "torch::Tensor matmul_swish_cuda(torch::Tensor a, torch::Tensor b, float scaling_factor);"
)

# Compile the inline CUDA code for matrix multiplication followed by Swish activation
matmul_swish = load_inline(
    name="matmul_swish",
    cpp_sources=matmul_swish_cpp_source,
    cuda_sources=matmul_swish_source,
    functions=["matmul_swish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul_swish = matmul_swish

    def forward(self, x):
        return self.matmul_swish.matmul_swish_cuda(x, x.t(), self.scaling_factor)