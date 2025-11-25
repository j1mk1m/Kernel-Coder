import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_gelu_divide_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

__global__ void matmul_gelu_divide_kernel(const float* a, const float* b, float* c, float* d, int m, int n, int k, float divisor) {
    extern __shared__ float smem[];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float acc = 0.0f;
    float sum = 0.0f;
    float avg = 0.0f;
    float sd = 0.0f;
    float gelu_out = 0.0f;

    if (row < m && col < n) {
        for (int i = 0; i < k; i += WARP_SIZE) {
            float val_a = (i + threadIdx.x < k) ? a[row * k + i + threadIdx.x] : 0.0f;
            float val_b = (i + threadIdx.x < k) ? b[i * n + col] : 0.0f;
            acc += val_a * val_b;
        }

        sum += acc;
        avg = sum / (k + WARP_SIZE - 1);
        sd = sqrt(avg * avg + (acc - avg) * (acc - avg) / (k + WARP_SIZE - 1));

        gelu_out = 0.5 * (1.0 + tanh(sqrt(2.0 / M_PI) * (avg + 0.044715 * sd)));

        c[row * n + col] = acc;
        d[row * n + col] = gelu_out / divisor;
    }
}

torch::Tensor matmul_gelu_divide_cuda(torch::Tensor a, torch::Tensor b, float divisor) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);

    auto c = torch::zeros({m, n}, a.options());
    auto d = torch::zeros({m, n}, a.options());

    const int block_size = 16;
    const int shared_memory_size = (block_size + WARP_SIZE - 1) * block_size * sizeof(float);

    dim3 grid((n + block_size - 1) / block_size, (m + block_size - 1) / block_size);
    dim3 block(block_size, block_size);

    matmul_gelu_divide_kernel<<<grid, block, shared_memory_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), d.data_ptr<float>(), m, n, k, divisor);

    return d;
}
"""

matmul_gelu_divide_cpp_source = (
    "torch::Tensor matmul_gelu_divide_cuda(torch::Tensor a, torch::Tensor b, float divisor);"
)

# Compile the inline CUDA code for matrix multiplication, GELU, and division
matmul_gelu_divide = load_inline(
    name="matmul_gelu_divide",
    cpp_sources=matmul_gelu_divide_cpp_source,
    cuda_sources=matmul_gelu_divide_source,
    functions=["matmul_gelu_divide_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.matmul_gelu_divide = matmul_gelu_divide

    def forward(self, x):
        return self.matmul_gelu_divide.matmul_gelu_divide_cuda(x, x.new_zeros(x.size()), divisor)