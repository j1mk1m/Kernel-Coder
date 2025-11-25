import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Gemm followed by LeakyReLU
gemm_leakyrelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CUBLAS_CHECK(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error: %d at %s:%d\n", status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void gemm_leakyrelu_kernel(float* A, float* B, float* C, float* D, int M, int N, int K, float alpha, float beta, float negative_slope) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = alpha * sum + beta * D[row * N + col];
    D[row * N + col] = max(C[row * N + col], negative_slope * C[row * N + col]);
}

torch::Tensor gemm_leakyrelu_cuda(torch::Tensor A, torch::Tensor B, float alpha, float beta, float negative_slope) {
    auto M = A.size(0);
    auto N = B.size(1);
    auto K = A.size(1);

    auto C = torch::zeros({M, N}, A.options());
    auto D = torch::zeros({M, N}, A.options());

    const int block_size = 32;
    const int grid_x = (N + block_size - 1) / block_size;
    const int grid_y = (M + block_size - 1) / block_size;

    gemm_leakyrelu_kernel<<<grid_y, grid_x>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(), M, N, K, alpha, beta, negative_slope);

    return D;
}
"""

gemm_leakyrelu_cpp_source = (
    "torch::Tensor gemm_leakyrelu_cuda(torch::Tensor A, torch::Tensor B, float alpha, float beta, float negative_slope);"
)

# Compile the inline CUDA code for Gemm followed by LeakyReLU
gemm_leakyrelu = load_inline(
    name="gemm_leakyrelu",
    cpp_sources=gemm_leakyrelu_cpp_source,
    cuda_sources=gemm_leakyrelu_source,
    functions=["gemm_leakyrelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm_leakyrelu = gemm_leakyrelu

    def forward(self, x):
        x = self.gemm_leakyrelu.gemm_leakyrelu_cuda(x, x.t(), 1.0, 0.0, negative_slope)
        x = x * multiplier
        return x

# Example usage
batch_size = 1024
in_features = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]

model_new = ModelNew(*get_init_inputs())
model_old = Model(*get_init_inputs())

check_model_performance(model_new, model_old, batch_size, in_features, out_features, multiplier, negative_slope)