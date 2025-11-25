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

    const int block_size = 16;
    const int num_blocks_x = (n + block_size - 1) / block_size;
    const int num_blocks_y = (m + block_size - 1) / block_size;

    matmul_kernel<<<num_blocks_y, num_blocks_x>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

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


# Define the custom CUDA kernel for scaling
scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scale_kernel(const float* a, float* b, float factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idx] = a[idx] * factor;
    }
}

torch::Tensor scale_cuda(torch::Tensor a, float factor) {
    auto size = a.numel();
    auto b = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    scale_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), factor, size);

    return b;
}
"""

scale_cpp_source = (
    "torch::Tensor scale_cuda(torch::Tensor a, float factor);"
)

# Compile the inline CUDA code for scaling
scale = load_inline(
    name="scale",
    cpp_sources=scale_cpp_source,
    cuda_sources=scale_source,
    functions=["scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for residual connection
residual_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void residual_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

torch::Tensor residual_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto c = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    residual_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), size);

    return c;
}
"""

residual_cpp_source = (
    "torch::Tensor residual_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for residual connection
residual = load_inline(
    name="residual",
    cpp_sources=residual_cpp_source,
    cuda_sources=residual_source,
    functions=["residual_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for clamping
clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void clamp_kernel(const float* a, float* b, float min_val, float max_val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idx] = a[idx] > max_val ? max_val : (a[idx] < min_val ? min_val : a[idx]);
    }
}

torch::Tensor clamp_cuda(torch::Tensor a, float min_val, float max_val) {
    auto size = a.numel();
    auto b = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    clamp_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), min_val, max_val, size);

    return b;
}
"""

clamp_cpp_source = (
    "torch::Tensor clamp_cuda(torch::Tensor a, float min_val, float max_val);"
)

# Compile the inline CUDA code for clamping
clamp = load_inline(
    name="clamp",
    cpp_sources=clamp_cpp_source,
    cuda_sources=clamp_source,
    functions=["clamp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for LogSumExp
logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void logsumexp_kernel(const float* a, float* b, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? a[i] : -FLT_MAX;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(b, sdata[0]);
    }
}

torch::Tensor logsumexp_cuda(torch::Tensor a) {
    auto size = a.numel();
    auto b = torch::zeros(1, a.options());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    logsumexp_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(a.data_ptr<float>(), b.data_ptr<float>(), size);

    return b;
}
"""

logsumexp_cpp_source = (
    "torch::Tensor logsumexp_cuda(torch::Tensor a);"
)

# Compile the inline CUDA code for LogSumExp
logsumexp = load_inline(
    name="logsumexp",
    cpp_sources=logsumexp_cpp_source,
    cuda_sources=logsumexp_source,
    functions=["logsumexp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Mish activation
mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mish_kernel(const float* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idx] = a[idx] * tanh(log(1.0f + exp(a[idx])));
    }
}

torch::Tensor mish_cuda(torch::Tensor a) {
    auto size = a.numel();
    auto b = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mish_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), size);

    return b;
}
"""

mish_cpp_source = (
    "torch::Tensor mish_cuda(torch::Tensor a);"
)

# Compile the inline CUDA code for Mish activation
mish = load_inline(
    name="mish",
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_source,
    functions=["mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = matmul
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.matmul.matmul_cuda(x, self.matmul.weight)
        x = self.scale_cuda.scale_cuda(x, self.scale_factor)
        x = self.residual_cuda.residual_cuda(x, x)
        x = self.clamp_cuda.clamp_cuda(x, self.clamp_min, self.clamp_max)
        x = self.logsumexp_cuda.logsumexp_cuda(x)
        x = self.mish_cuda.mish_cuda(x)
        return x

# Example usage:
if __name__ == "__main__":
    batch_size = 1024
    input_size = 8192
    hidden_size = 8192
    scale_factor = 2.0
    clamp_min = -10.0
    clamp_max = 10.0

    model = ModelNew(input_size, hidden_size, scale_factor, clamp_min, clamp_max)
    inputs = get_inputs()[0].cuda()
    outputs = model(inputs)
    print(outputs.shape)