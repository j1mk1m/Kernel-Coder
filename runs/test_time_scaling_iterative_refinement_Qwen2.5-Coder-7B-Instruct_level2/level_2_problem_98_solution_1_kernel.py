import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Matmul
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
    const int grid_x = (n + block_size - 1) / block_size;
    const int grid_y = (m + block_size - 1) / block_size;

    matmul_kernel<<<grid_y, grid_x>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

    return c;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for Matmul
matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for AvgPool
avgpool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avgpool_kernel(const float* a, float* b, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[(row * k + i) * n + col];
        }
        b[row * n + col] = sum / k;
    }
}

torch::Tensor avgpool_cuda(torch::Tensor a, int kernel_size) {
    auto m = a.size(0);
    auto n = a.size(1);

    auto b = torch::zeros({m, n}, a.options());

    const int block_size = 16;
    const int grid_x = (n + block_size - 1) / block_size;
    const int grid_y = (m + block_size - 1) / block_size;

    avgpool_kernel<<<grid_y, grid_x>>>(a.data_ptr<float>(), b.data_ptr<float>(), m, n, kernel_size);

    return b;
}
"""

avgpool_cpp_source = (
    "torch::Tensor avgpool_cuda(torch::Tensor a, int kernel_size);"
)

# Compile the inline CUDA code for AvgPool
avgpool = load_inline(
    name="avgpool",
    cpp_sources=avgpool_cpp_source,
    cuda_sources=avgpool_source,
    functions=["avgpool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for GELU
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idx] = a[idx] * (0.5f + 0.5f * tanh(sqrt(2.0f / M_PI) * (a[idx] + 0.044715f * a[idx] * a[idx] * a[idx])));
    }
}

torch::Tensor gelu_cuda(torch::Tensor a) {
    auto size = a.numel();

    auto b = torch::zeros_like(a);

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    gelu_kernel<<<grid_size, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), size);

    return b;
}
"""

gelu_cpp_source = (
    "torch::Tensor gelu_cuda(torch::Tensor a);"
)

# Compile the inline CUDA code for GELU
gelu = load_inline(
    name="gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for Scale
scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scale_kernel(const float* a, float* b, float scale_factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idx] = a[idx] * scale_factor;
    }
}

torch::Tensor scale_cuda(torch::Tensor a, float scale_factor) {
    auto size = a.numel();

    auto b = torch::zeros_like(a);

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    scale_kernel<<<grid_size, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), scale_factor, size);

    return b;
}
"""

scale_cpp_source = (
    "torch::Tensor scale_cuda(torch::Tensor a, float scale_factor);"
)

# Compile the inline CUDA code for Scale
scale = load_inline(
    name="scale",
    cpp_sources=scale_cpp_source,
    cuda_sources=scale_source,
    functions=["scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for Max
max_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

extern __shared__ float sdata[];

__global__ void max_kernel(const float* a, float* b, int size) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? a[i] : -FLT_MAX;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        b[blockIdx.x] = sdata[0];
    }
}

torch::Tensor max_cuda(torch::Tensor a) {
    auto size = a.numel();

    auto b = torch::zeros({1}, a.options());

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    max_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(a.data_ptr<float>(), b.data_ptr<float>(), size);

    return b;
}
"""

max_cpp_source = (
    "torch::Tensor max_cuda(torch::Tensor a);"
)

# Compile the inline CUDA code for Max
max = load_inline(
    name="max",
    cpp_sources=max_cpp_source,
    cuda_sources=max_source,
    functions=["max_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = matmul
        self.avgpool = avgpool
        self.gelu = gelu
        self.scale = scale
        self.max = max
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.matmul.matmul_cuda(x, self.matmul.weight.t())
        x = self.avgpool.avgpool_cuda(x, self.avgpool.kernel_size)
        x = self.gelu.gelu_cuda(x)
        x = self.scale.scale_cuda(x, self.scale_factor)
        x = self.max.max_cuda(x)
        return x

# Example usage
if __name__ == "__main__":
    batch_size = 1024
    in_features = 8192
    out_features = 8192
    pool_kernel_size = 16
    scale_factor = 2.0

    model_new = ModelNew(in_features, out_features, pool_kernel_size, scale_factor)
    inputs = torch.randn(batch_size, in_features).cuda()

    outputs = model_new(inputs)
    print(outputs.shape)