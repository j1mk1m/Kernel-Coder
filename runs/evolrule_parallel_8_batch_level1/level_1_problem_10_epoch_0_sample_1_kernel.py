import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D tensor-matrix multiplication
tensor_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tensor_matmul_kernel(const float* A, const float* B, float* C,
                                    int N, int M, int K, int L) {
    // Each thread computes one element of the output tensor C
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    if (batch >= N || row >= M || col >= L)
        return;

    float sum = 0.0;
    for (int k = 0; k < K; ++k) {
        sum += A[batch * M * K + row * K + k] * B[k * L + col];
    }
    C[batch * M * L + row * L + col] = sum;
}

torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    auto C = torch::empty({N, M, L}, A.options());

    dim3 threads(32, 8, 1);  // Thread block dimensions (x, y, z)
    dim3 blocks((L + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y,
                N);  // Block dimensions

    tensor_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        N, M, K, L);

    return C;
}
"""

tensor_matmul_cpp_source = (
    "torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
tensor_matmul = load_inline(
    name="tensor_matmul",
    cpp_sources=tensor_matmul_cpp_source,
    cuda_sources=tensor_matmul_source,
    functions=["tensor_matmul_cuda"],
    verbose=True,
    extra_cflags=["-DUSE_CUDA"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tensor_matmul = tensor_matmul

    def forward(self, A, B):
        return self.tensor_matmul.tensor_matmul_cuda(A, B)

# Inputs (for reference, not part of the model code)
def get_inputs():
    N = 16
    M = 1024
    K = 2048
    L = 768
    A = torch.rand(N, M, K).cuda()
    B = torch.rand(K, L).cuda()
    return [A, B]

def get_init_inputs():
    return []