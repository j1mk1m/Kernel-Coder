import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GEMM
gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void gemm_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, hipStreamPerThread);

    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle, transA, transB, n, m, k, &alpha, b, k, a, k, &beta, c, n);
    cublasDestroy(handle);
}

torch::Tensor gemm_cuda(torch::Tensor a, torch::Tensor b) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);

    auto c = torch::zeros({n, m}, a.options());

    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    gemm_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

    return c.transpose(0, 1);
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

__global__ void max_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = x[idx];
    }
}

torch::Tensor max_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    max_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

max_cpp_source = (
    "torch::Tensor max_cuda(torch::Tensor x);"
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


# Define the custom CUDA kernel for mean subtraction
mean_subtraction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_subtraction_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = x[idx];
    }
}

torch::Tensor mean_subtraction_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mean_subtraction_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

mean_subtraction_cpp_source = (
    "torch::Tensor mean_subtraction_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for mean subtraction
mean_subtraction = load_inline(
    name="mean_subtraction",
    cpp_sources=mean_subtraction_cpp_source,
    cuda_sources=mean_subtraction_source,
    functions=["mean_subtraction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for GELU activation
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = x[idx] * 0.5 * (1 + tanh(sqrt(2 / M_PI) * (x[idx] + 0.044715 * x[idx] * x[idx] * x[idx])));
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

gelu_cpp_source = (
    "torch::Tensor gelu_cuda(torch::Tensor x);"
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
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = gemm
        self.max_dim = max_dim

    def forward(self, x):
        x = self.gemm.gemm_cuda(x, x.t())
        x = max_op.max_cuda(x)
        x = mean_subtraction.mean_subtraction_cuda(x)
        x = gelu.gelu_cuda(x)
        return x