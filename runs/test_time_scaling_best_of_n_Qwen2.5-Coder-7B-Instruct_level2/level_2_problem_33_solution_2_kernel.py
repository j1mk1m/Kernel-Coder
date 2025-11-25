import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GEMM
gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    auto C = torch::zeros({M, N}, A.options());

    const int block_size = 32;
    const int num_blocks_x = (N + block_size - 1) / block_size;
    const int num_blocks_y = (M + block_size - 1) / block_size;

    gemm_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

gemm_cpp_source = (
    "torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B);"
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


# Define the custom CUDA kernel for Batch Normalization
bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void bn_forward_kernel(const float* x, float* mean, float* var, float* y, float* gamma, float* beta, int N, int C) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N * C) {
        int c = index % C;
        int n = index / C;
        float x_val = x[index];
        mean[c] += x_val;
        var[c] += x_val * x_val;
    }
}

__global__ void bn_backward_kernel(const float* x, const float* dy, float* dx, float* mean, float* var, float* gamma, float* beta, int N, int C) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N * C) {
        int c = index % C;
        int n = index / C;
        float x_val = x[index];
        float dy_val = dy[index];
        float var_inv = 1.0f / sqrt(var[c] + 1e-5);
        float gamma_val = gamma[c];
        dx[index] = gamma_val * var_inv * (dy_val - (x_val - mean[c]) * var_inv);
    }
}

torch::Tensor bn_forward_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta) {
    int N = x.size(0);
    int C = x.size(1);

    auto mean = torch::zeros(C, x.options());
    auto var = torch::zeros(C, x.options());
    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (N * C + block_size - 1) / block_size;

    bn_forward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), y.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), N, C);

    return y;
}

torch::Tensor bn_backward_cuda(torch::Tensor x, torch::Tensor dy, torch::Tensor gamma, torch::Tensor beta) {
    int N = x.size(0);
    int C = x.size(1);

    auto mean = torch::zeros(C, x.options());
    auto var = torch::zeros(C, x.options());
    auto dx = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (N * C + block_size - 1) / block_size;

    bn_backward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), dy.data_ptr<float>(), dx.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), N, C);

    return dx;
}
"""

bn_cpp_source = (
    "torch::Tensor bn_forward_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta);\n"
    "torch::Tensor bn_backward_cuda(torch::Tensor x, torch::Tensor dy, torch::Tensor gamma, torch::Tensor beta);"
)

# Compile the inline CUDA code for Batch Normalization
bn = load_inline(
    name="bn",
    cpp_sources=bn_cpp_source,
    cuda_sources=bn_source,
    functions=["bn_forward_cuda", "bn_backward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernels for GEMM and Batch Normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = gemm
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = bn

    def forward(self, x):
        x = self.gemm.gemm_cuda(x, self.scale.view(-1, 1))
        x = self.bn.bn_forward_cuda(x, self.scale, torch.zeros_like(self.scale))
        return x