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
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);
    auto c = torch::zeros({m, n}, a.options());

    const int block_size = 32;
    dim3 grid((n + block_size - 1) / block_size, (m + block_size - 1) / block_size);
    dim3 block(block_size, block_size);

    matmul_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

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

# Define the custom CUDA kernel for division
division_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void division_kernel(const float* a, float* out, int size, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] / divisor;
    }
}

torch::Tensor division_cuda(torch::Tensor a, float divisor) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    division_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), out.data_ptr<float>(), size, divisor);

    return out;
}
"""

division_cpp_source = (
    "torch::Tensor division_cuda(torch::Tensor a, float divisor);"
)

# Compile the inline CUDA code for division
division = load_inline(
    name="division",
    cpp_sources=division_cpp_source,
    cuda_sources=division_source,
    functions=["division_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for GELU activation
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ float gelu_device(float x) {
    return 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_kernel(const float* a, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = gelu_device(a[idx]);
    }
}

torch::Tensor gelu_cuda(torch::Tensor a) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

gelu_cpp_source = (
    "torch::Tensor gelu_cuda(torch::Tensor a);"
)

# Compile the inline CUDA code for GELU activation
gelu = load_inline(
    name="gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.matmul = matmul
        self.division = division
        self.gelu = gelu

    def forward(self, x):
        x = self.linear(x)
        x = self.matmul.matmul_cuda(x, self.linear.weight.t())
        x = self.division.division_cuda(x, self.divisor)
        x = self.gelu.gelu_cuda(x)
        return x


def get_inputs():
    batch_size = 1024
    input_size = 8192
    output_size = 8192
    divisor = 10.0
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, output_size, divisor]