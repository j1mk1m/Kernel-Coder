import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication and scaling
matmul_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_scale_kernel(const float* a, const float* b, float* c, int rows_a, int cols_a, int cols_b, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_a && col < cols_b) {
        float sum = 0.0f;
        for (int k = 0; k < cols_a; ++k) {
            sum += a[row * cols_a + k] * b[k * cols_b + col];
        }
        c[row * cols_b + col] = sum * scale;
    }
}

torch::Tensor matmul_scale_cuda(torch::Tensor a, torch::Tensor b, float scale) {
    auto rows_a = a.size(0);
    auto cols_a = a.size(1);
    auto cols_b = b.size(1);
    auto c = torch::zeros({rows_a, cols_b}, a.options());

    dim3 block_size(16, 16);
    dim3 grid_size((cols_b + block_size.x - 1) / block_size.x, (rows_a + block_size.y - 1) / block_size.y);

    matmul_scale_kernel<<<grid_size, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), rows_a, cols_a, cols_b, scale);

    return c;
}
"""

matmul_scale_cpp_source = (
    "torch::Tensor matmul_scale_cuda(torch::Tensor a, torch::Tensor b, float scale);"
)

# Compile the inline CUDA code for matrix multiplication and scaling
matmul_scale = load_inline(
    name="matmul_scale",
    cpp_sources=matmul_scale_cpp_source,
    cuda_sources=matmul_scale_source,
    functions=["matmul_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul_scale = matmul_scale

    def forward(self, x):
        x = self.matmul_scale.matmul_scale_cuda(x, x.t(), self.scaling_factor)
        return x