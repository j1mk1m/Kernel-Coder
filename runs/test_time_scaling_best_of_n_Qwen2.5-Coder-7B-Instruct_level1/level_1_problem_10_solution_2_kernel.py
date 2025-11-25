import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D tensor-matrix multiplication
tensor_matrix_multiplication_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 3D tensor-matrix multiplication
__global__ void tensor_matrix_multiplication_kernel(const float* A, const float* B, float* C, int N, int M, int K, int L) {
    int n = blockIdx.x; // Batch index
    int m = blockIdx.y; // Row index in A
    int l = blockIdx.z; // Column index in B

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[n * M * K + m * K + k] * B[k * L + l];
    }

    C[n * M * L + m * L + l] = sum;
}

torch::Tensor tensor_matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto M = A.size(1);
    auto K = A.size(2);
    auto L = B.size(1);

    auto C = torch::zeros({N, M, L}, A.options());

    dim3 blocks_per_grid(N, M, L);
    dim3 threads_per_block(1, 1, 1);

    tensor_matrix_multiplication_kernel<<<blocks_per_grid, threads_per_block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M, K, L);

    return C;
}
"""

tensor_matrix_multiplication_cpp_source = (
    "torch::Tensor tensor_matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for 3D tensor-matrix multiplication
tensor_matrix_multiplication = load_inline(
    name="tensor_matrix_multiplication",
    cpp_sources=tensor_matrix_multiplication_cpp_source,
    cuda_sources=tensor_matrix_multiplication_source,
    functions=["tensor_matrix_multiplication_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tensor_matrix_multiplication = tensor_matrix_multiplication

    def forward(self, A, B):
        return self.tensor_matrix_multiplication.tensor_matrix_multiplication_cuda(A, B)


if __name__ == "__main__":
    model = ModelNew().cuda()
    A, B = get_inputs()
    result = model(A.cuda(), B.cuda())
    print(result.shape)  # Should print torch.Size([16, 1024, 768])