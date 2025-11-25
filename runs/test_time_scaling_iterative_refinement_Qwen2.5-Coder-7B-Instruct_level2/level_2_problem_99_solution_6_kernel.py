import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication followed by GELU
matmul_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

__device__ float gelu_device(float x) {
    return 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

__global__ void matmul_gelu_kernel(const float* A, const float* B, float* C, int m, int n, int k) {
    __shared__ float sA[WARP_SIZE][WARP_SIZE];
    __shared__ float sB[WARP_SIZE][WARP_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int i = 0; i < k; i += blockDim.z) {
        if (row < m && i + threadIdx.z < k) {
            sA[threadIdx.y][threadIdx.x] = A[row * k + i + threadIdx.z];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && i + threadIdx.z < k) {
            sB[threadIdx.y][threadIdx.x] = B[(i + threadIdx.z) * n + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int j = 0; j < WARP_SIZE; ++j) {
            sum += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = gelu_device(sum);
    }
}

torch::Tensor matmul_gelu_cuda(torch::Tensor A, torch::Tensor B) {
    auto m = A.size(0);
    auto n = B.size(1);
    auto k = A.size(1);
    auto out = torch::zeros({m, n}, A.options());

    const int block_size = 16;
    const int grid_size_x = (n + block_size - 1) / block_size;
    const int grid_size_y = (m + block_size - 1) / block_size;
    const int grid_size_z = (k + block_size - 1) / block_size;

    matmul_gelu_kernel<<<grid_size_y, grid_size_x, block_size * block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), m, n, k);

    return out;
}
"""

matmul_gelu_cpp_source = (
    "torch::Tensor matmul_gelu_cuda(torch::Tensor A, torch::Tensor B);"
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
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.matmul_gelu = matmul_gelu

    def forward(self, x):
        x = self.linear(x)
        x = self.matmul_gelu.matmul_gelu_cuda(x, x.new_zeros(x.size()))
        return x


def get_inputs():
    batch_size = 1024
    in_features = 8192
    out_features = 8192
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]