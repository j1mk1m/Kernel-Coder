import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication followed by mish activation
matmul_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Function to compute the mish activation
__device__ float mish(float x) {
    return x * tanh(log1p(exp(x)));
}

// Kernel to perform matrix multiplication followed by mish activation
__global__ void matmul_mish_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = mish(sum);
    }
}

torch::Tensor matmul_mish_cuda(torch::Tensor a, torch::Tensor b) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);
    auto c = torch::zeros({m, n}, a.options());

    const int block_size = 32;
    dim3 grid((n + block_size - 1) / block_size, (m + block_size - 1) / block_size);
    dim3 block(block_size, block_size);

    matmul_mish_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

    return c;
}
"""

matmul_mish_cpp_source = (
    "torch::Tensor matmul_mish_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for matrix multiplication followed by mish activation
matmul_mish = load_inline(
    name="matmul_mish",
    cpp_sources=matmul_mish_cpp_source,
    cuda_sources=matmul_mish_source,
    functions=["matmul_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.matmul_mish = matmul_mish

    def forward(self, x):
        x = self.matmul_mish.matmul_mish_cuda(x, self.linear.weight.t())
        x = self.matmul_mish.matmul_mish_cuda(x, self.linear.weight.t())
        return x

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]