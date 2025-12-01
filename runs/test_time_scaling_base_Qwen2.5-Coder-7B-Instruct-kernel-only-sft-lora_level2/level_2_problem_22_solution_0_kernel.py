import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication and scaling
matmul_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_scale_kernel(const float* a, const float* b, float* c, int m, int n, int k, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum * scale;
    }
}

torch::Tensor matmul_scale_cuda(torch::Tensor a, torch::Tensor b, float scale) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);
    auto out = torch::zeros({m, n}, a.options());

    const int block_size = 32;
    const int num_blocks_x = (n + block_size - 1) / block_size;
    const int num_blocks_y = (m + block_size - 1) / block_size;

    matmul_scale_kernel<<<num_blocks_y, num_blocks_x, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), m, n, k, scale);

    return out;
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
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.matmul_scale = matmul_scale

    def forward(self, x):
        x = self.matmul_scale.matmul_scale_cuda(x, self.weight, self.scale_factor)
        x = x + x
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = x * torch.nn.functional.mish(x)  # Mish activation
        return x

    def initialize_weights(self):
        self.weight = nn.Parameter(torch.randn(self.hidden_size, self.input_size))