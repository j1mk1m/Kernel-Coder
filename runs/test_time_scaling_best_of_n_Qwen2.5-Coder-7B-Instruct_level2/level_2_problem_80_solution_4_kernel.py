import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GEMM
gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
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

torch::Tensor gemm_cuda(torch::Tensor a, torch::Tensor b) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);
    auto c = torch::zeros({m, n}, a.options());

    const int block_size = 32;
    const int num_blocks_x = (n + block_size - 1) / block_size;
    const int num_blocks_y = (m + block_size - 1) / block_size;

    gemm_kernel<<<num_blocks_y, num_blocks_x>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

    return c;
}
"""

gemm_cpp_source = (
    "torch::Tensor gemm_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for GEMM
gemm = load_inline(
    name="gemm",
    cpp_sources=gemm_cpp_source,
    cuda_sources=gemm_source,
    functions=["gemm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for max operation
max_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_kernel(const float* a, float* max_values, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        max_values[0] = fmax(max_values[0], a[idx]);
    }
}

torch::Tensor max_cuda(torch::Tensor a) {
    auto size = a.numel();
    auto max_values = torch::tensor(-FLT_MAX, a.options()).unsqueeze(0);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    max_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), max_values.data_ptr<float>(), size);

    return max_values;
}
"""

max_cpp_source = (
    "torch::Tensor max_cuda(torch::Tensor a);"
)

# Compile the inline CUDA code for max operation
max_op = load_inline(
    name="max_op",
    cpp_sources=max_cpp_source,
    cuda_sources=max_source,
    functions=["max_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for subtraction
subtraction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void subtraction_kernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] - b[idx];
    }
}

torch::Tensor subtraction_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto result = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    subtraction_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), result.data_ptr<float>(), size);

    return result;
}
"""

subtraction_cpp_source = (
    "torch::Tensor subtraction_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for subtraction
subtraction = load_inline(
    name="subtraction",
    cpp_sources=subtraction_cpp_source,
    cuda_sources=subtraction_source,
    functions=["subtraction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for mean operation
mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_kernel(const float* a, float* mean_value, int size) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? a[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(mean_value, sdata[0]);
    }
}

torch::Tensor mean_cuda(torch::Tensor a) {
    auto size = a.numel();
    auto mean_value = torch::zeros({}, a.options());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mean_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(a.data_ptr<float>(), mean_value.data_ptr<float>(), size);

    return mean_value / static_cast<float>(num_blocks);
}
"""

mean_cpp_source = (
    "torch::Tensor mean_cuda(torch::Tensor a);"
)

# Compile the inline CUDA code for mean operation
mean_op = load_inline(
    name="mean_op",
    cpp_sources=mean_cpp_source,
    cuda_sources=mean_source,
    functions=["mean_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for GELU activation
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* a, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * 0.5f * (1.0f + tanh(sqrt(2.0f / M_PI) * (a[idx] + 0.044715f * a[idx] * a[idx] * a[idx])));
    }
}

torch::Tensor gelu_cuda(torch::Tensor a) {
    auto size = a.numel();
    auto result = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), result.data_ptr<float>(), size);

    return result;
}
"""

gelu_cpp_source = (
    "torch::Tensor gelu_cuda(torch::Tensor a);"
)

# Compile the inline CUDA code for GELU activation
gelu_op = load_inline(
    name="gelu_op",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = gemm
        self.max_op = max_op
        self.subtraction = subtraction
        self.mean_op = mean_op
        self.gelu_op = gelu_op

    def forward(self, x):
        x = self.gemm.gemm_cuda(x, x.new_zeros((x.size(1), x.size(1)), dtype=torch.float32))
        x = self.max_op.max_cuda(x)
        x = self.subtraction.subtraction_cuda(x, x.new_full((x.size(0), 1), x.mean().item(), dtype=torch.float32))
        x = self.gelu_op.gelu_cuda(x)
        return x

# Initialize inputs
inputs = get_inputs()

# Create model instance
model_new = ModelNew(in_features=in_features, out_features=out_features, max_dim=max_dim)

# Forward pass
output = model_new(inputs[0])

print(output.shape)