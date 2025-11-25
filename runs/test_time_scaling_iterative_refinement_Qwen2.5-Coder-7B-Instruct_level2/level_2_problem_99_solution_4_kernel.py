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

__device__ float gelu(float x) {
    return x * 0.5f * (1.0f + tanh(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

__global__ void matmul_gelu_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float s_a[WARP_SIZE][WARP_SIZE];
    __shared__ float s_b[WARP_SIZE][WARP_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int i = 0; i < k; i += blockDim.z) {
        int ai = row * k + i;
        int bi = i * n + col;

        if (row < m && i < k) s_a[threadIdx.y][threadIdx.x] = a[ai];
        else s_a[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < n && i < k) s_b[threadIdx.y][threadIdx.x] = b[bi];
        else s_b[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int j = 0; j < WARP_SIZE; ++j) {
            sum += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = gelu(sum);
    }
}

torch::Tensor matmul_gelu_cuda(torch::Tensor a, torch::Tensor b) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);
    auto out = torch::zeros({m, n}, a.options());

    const int block_size_x = 32;
    const int block_size_y = 32;
    const int block_size_z = 4;
    const int grid_size_x = (n + block_size_x - 1) / block_size_x;
    const int grid_size_y = (m + block_size_y - 1) / block_size_y;

    matmul_gelu_kernel<<<grid_size_y, block_size_x * block_size_y, 0, at::cuda::getCurrentCUDAStream()>>>(
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
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.matmul_gelu = matmul_gelu

    def forward(self, x):
        x = self.matmul_gelu.matmul_gelu_cuda(x, self.weight)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


def get_inputs():
    batch_size = 1024
    in_features = 8192
    out_features = 8192
    return [torch.randn(batch_size, in_features).cuda(), torch.randn(out_features, in_features).cuda()]

def get_init_inputs():
    return [8192, 8192]

# Initialize the model
model_new = ModelNew(get_init_inputs()[0], get_init_inputs()[1]).cuda()

# Get inputs
inputs = get_inputs()

# Forward pass
output = model_new(inputs[0])
print(output.shape)