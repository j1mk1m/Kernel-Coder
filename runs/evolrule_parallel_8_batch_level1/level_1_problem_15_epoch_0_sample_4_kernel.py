import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

triangular_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void triangular_matmul(
    const float* A, const float* B, float* C, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;

    int i = idx / N;
    int j = idx % N;

    if (i < j) {
        C[idx] = 0.0f;
        return;
    }

    float sum = 0.0f;
    for (int k = j; k <= i; k++) {
        sum += A[i * N + k] * B[k * N + j];
    }
    C[idx] = sum;
}

torch::Tensor triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    const int block_size = 256;
    const int grid_size = (N * N + block_size - 1) / block_size;

    triangular_matmul<<<grid_size, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    cudaDeviceSynchronize();
    return C;
}
"""

triangular_matmul_cpp = """
torch::Tensor triangular_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

triangular_matmul = load_inline(
    name="triangular_matmul",
    cpp_sources=triangular_matmul_cpp,
    cuda_sources=triangular_matmul_source,
    functions=["triangular_matmul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.triangular_matmul = triangular_matmul

    def forward(self, A, B):
        return self.triangular_matmul.triangular_matmul_cuda(A, B)