import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cublas_matmul_transpose_source = """
#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void transpose_kernel(const float* in, float* out, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int i = idx / N;
        int j = idx % N;
        out[idx] = in[j * M + i];
    }
}

torch::Tensor cublas_matmul_transpose(torch::Tensor A, torch::Tensor B) {
    // Check dimensions
    int K_A = A.size(0);
    int M_A = A.size(1);
    int N_B = B.size(0);
    int K_B = B.size(1);

    TORCH_CHECK(K_A == K_B, "Incompatible dimensions between A and B");

    int N = N_B;
    int M = M_A;

    auto options = A.options();
    auto temp = torch::empty({N, M}, options);

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* temp_data = temp.data_ptr<float>();

    int lda = A.stride(0);
    int ldb = B.stride(0);
    int ldc = temp.stride(0);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    int m = N;
    int n = M;
    int k = K_A;

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle, transa, transb, m, n, k,
                &alpha,
                B_data, ldb,
                A_data, lda,
                &beta,
                temp_data, ldc);

    cublasDestroy(handle);

    auto result = torch::empty({M, N}, options);
    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;

    transpose_kernel<<<blocks, threads>>>(temp_data, result.data_ptr<float>(), M, N);
    cudaDeviceSynchronize();

    return result;
}
"""

cublas_matmul_transpose_header = """
#include <torch/extension.h>
torch::Tensor cublas_matmul_transpose(torch::Tensor A, torch::Tensor B);
"""

module = load_inline(
    name="cublas_matmul_transpose",
    cuda_sources=cublas_matmul_transpose_source,
    cpp_sources=cublas_matmul_transpose_header,
    functions=["cublas_matmul_transpose"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.cuda()
        B = B.cuda()
        return module.cublas_matmul_transpose(A, B)