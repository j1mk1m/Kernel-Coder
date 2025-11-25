import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Gemm
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

    const int block_size = 16;
    const int num_blocks_x = (n + block_size - 1) / block_size;
    const int num_blocks_y = (m + block_size - 1) / block_size;

    gemm_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

    return c;
}
"""

gemm_cpp_source = (
    "torch::Tensor gemm_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for Gemm
gemm = load_inline(
    name="gemm",
    cpp_sources=gemm_cpp_source,
    cuda_sources=gemm_source,
    functions=["gemm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Subtract
subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void subtract_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

torch::Tensor subtract_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    subtract_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

subtract_cpp_source = (
    "torch::Tensor subtract_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for Subtract
subtract = load_inline(
    name="subtract",
    cpp_sources=subtract_cpp_source,
    cuda_sources=subtract_source,
    functions=["subtract_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for GlobalAvgPool
global_avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void global_avg_pool_kernel(const float* a, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[0] += a[idx];
    }
}

torch::Tensor global_avg_pool_cuda(torch::Tensor a) {
    auto size = a.numel();

    auto out = torch::zeros(1, a.options());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    global_avg_pool_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), out.data_ptr<float>(), size);

    out[0] /= size;
    return out;
}
"""

global_avg_pool_cpp_source = (
    "torch::Tensor global_avg_pool_cuda(torch::Tensor a);"
)

# Compile the inline CUDA code for GlobalAvgPool
global_avg_pool = load_inline(
    name="global_avg_pool",
    cpp_sources=global_avg_pool_cpp_source,
    cuda_sources=global_avg_pool_source,
    functions=["global_avg_pool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for LogSumExp
log_sum_exp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void log_sum_exp_kernel(const float* a, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[0] += exp(a[idx]);
    }
}

torch::Tensor log_sum_exp_cuda(torch::Tensor a) {
    auto size = a.numel();

    auto out = torch::zeros(1, a.options());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    log_sum_exp_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), out.data_ptr<float>(), size);

    out[0] = log(out[0]) - log(size);
    return out;
}
"""

log_sum_exp_cpp_source = (
    "torch::Tensor log_sum_exp_cuda(torch::Tensor a);"
)

# Compile the inline CUDA code for LogSumExp
log_sum_exp = load_inline(
    name="log_sum_exp",
    cpp_sources=log_sum_exp_cpp_source,
    cuda_sources=log_sum_exp_source,
    functions=["log_sum_exp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for GELU
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* a, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * 0.5f * (1.0f + tanh(sqrt(2.0f / M_PI) * (a[idx] + 0.044715f * a[idx] * a[idx] * a[idx])));
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


# Define the custom CUDA kernel for ResidualAdd
residual_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void residual_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor residual_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    residual_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

residual_add_cpp_source = (
    "torch::Tensor residual_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for ResidualAdd
residual_add = load_inline(
    name="residual_add",
    cpp_sources=residual_add_cpp_source,
    cuda_sources=residual_add_source,
    functions=["residual_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = gemm
        self.subtract = subtract
        self.global_avg_pool = global_avg_pool
        self.log_sum_exp = log_sum_exp
        self.gelu = gelu
        self.residual_add = residual_add

    def forward(self, x):
        original_x = x.clone().detach()

        # Gemm
        x = self.gemm.gemm_cuda(x, self.weight)

        # Subtract
        x = self.subtract.subtract_cuda(x, self.subtract_param)

        # GlobalAvgPool
        x = self.global_avg_pool.global_avg_pool_cuda(x)

        # LogSumExp
        x = self.log_sum_exp.log_sum_exp_cuda(x)

        # GELU
        x = self.gelu.gelu_cuda(x)

        # ResidualAdd
        x = self.residual_add.residual_add_cuda(x, original_x)

        return x


def get_inputs():
    batch_size = 2048
    in_features = 8192
    out_features = 8192
    return [torch.randn(batch_size, in_features).cuda()]


def get_init_inputs():
    in_features = 8192
    out_features = 8192
    return [in_features, out_features]