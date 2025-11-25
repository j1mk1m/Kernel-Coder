import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Your custom CUDA kernel for matrix multiplication
matrix_multiplication_source = """
// Include necessary headers
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel definition
__global__ void matrix_multiplication_kernel(float* A, float* B, float* C, int M, int K, int N) {
    // Thread index within block and grid
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for A and B
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    // Initialize sum to zero
    float sum = 0.0f;

    // Loop over tiles of A and B
    for (int m = 0; m < (M + 15) / 16; ++m) {
        for (int n = 0; n < (N + 15) / 16; ++n) {
            // Load A and B tile into shared memory
            As[threadIdx.y][threadIdx.x] = (row < M && m * 16 + threadIdx.y < K) ? A[row * K + m * 16 + threadIdx.y] : 0.0f;
            Bs[threadIdx.y][threadIdx.x] = (col < N && n * 16 + threadIdx.x < K) ? B[(m * 16 + threadIdx.y) * N + col] : 0.0f;

            __syncthreads();

            // Compute dot product of A and B tile
            for (int k = 0; k < 16; ++k) {
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }

            __syncthreads();
        }
    }

    // Store result in C
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Wrapper function to call the kernel from Python
torch::Tensor matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B) {
    // Get dimensions
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Allocate output tensor
    auto C = torch::zeros({M, N}, A.options());

    // Set up grid and block sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}
"""

# Register the CUDA function
matrix_multiplication_cpp_source = (
    "torch::Tensor matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B);"
)

# Load the inline CUDA code
matrix_multiplication = load_inline(
    name="matrix_multiplication",
    cpp_sources=matrix_multiplication_cpp_source,
    cuda_sources=matrix_multiplication_source,
    functions=["matrix_multiplication_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return matrix_multiplication.matrix_multiplication_cuda(A, B)

# Example usage
if __name__ == "__main__":
    A, B = get_inputs()
    model_new = ModelNew().cuda()
    result = model_new(A.cuda(), B.cuda())
    print(result.shape)