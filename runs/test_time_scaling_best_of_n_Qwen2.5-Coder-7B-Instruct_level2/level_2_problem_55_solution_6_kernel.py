import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    int m = a.size(0);
    int n = b.size(1);
    int k = a.size(1);
    auto c = torch::zeros({m, n}, a.options());

    const int block_size = 16;
    const int num_blocks_x = (n + block_size - 1) / block_size;
    const int num_blocks_y = (m + block_size - 1) / block_size;

    matmul_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

    return c;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for matrix multiplication
matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for max pooling
max_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_pool_kernel(const float* input, float* output, int width, int height, int pool_width, int pool_height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int p_row = 0; p_row < pool_height; ++p_row) {
            for (int p_col = 0; p_col < pool_width; ++p_col) {
                int input_idx = ((row * width + col) * pool_height + p_row) * pool_width + p_col;
                if (input[input_idx] > max_val) {
                    max_val = input[input_idx];
                }
            }
        }
        output[row * width + col] = max_val;
    }
}

torch::Tensor max_pool_cuda(torch::Tensor input, int pool_width, int pool_height) {
    int width = input.size(0);
    int height = input.size(1);
    auto output = torch::zeros({width, height}, input.options());

    const int block_size = 16;
    const int num_blocks_x = (width + block_size - 1) / block_size;
    const int num_blocks_y = (height + block_size - 1) / block_size;

    max_pool_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(input.data_ptr<float>(), output.data_ptr<float>(), width, height, pool_width, pool_height);

    return output;
}
"""

max_pool_cpp_source = (
    "torch::Tensor max_pool_cuda(torch::Tensor input, int pool_width, int pool_height);"
)

# Compile the inline CUDA code for max pooling
max_pool = load_inline(
    name="max_pool",
    cpp_sources=max_pool_cpp_source,
    cuda_sources=max_pool_source,
    functions=["max_pool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = matmul
        self.max_pool = max_pool
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.matmul.matmul_cuda(x, x.t())
        x = self.max_pool.max_pool_cuda(x.view(-1, x.size(0)), kernel_size, kernel_size)
        x = x.squeeze().sum()
        x = x * self.scale_factor
        return x