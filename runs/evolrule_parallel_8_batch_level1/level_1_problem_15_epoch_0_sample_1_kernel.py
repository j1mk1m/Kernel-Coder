import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel code for lower triangular matrix multiplication
lower_triangular_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define T 32  // Block dimension (must be a power of 2 for best performance)

__global__ void lower_triangular_matmul(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int N) 
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int startRow = by * T;
    int startCol = bx * T;

    // Skip blocks that are in the upper triangle
    if (startRow < startCol)
        return;

    int i = startRow + ty;
    int j = startCol + tx;

    if (i >= N || j >= N)
        return;

    if (i < j)
        return;

    float sum = 0.0f;
    // Since i >= j, the minimum of i and j is j
    for (int k = 0; k <= j; ++k) {
        sum += A[i * N + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
}

// Host function to launch the CUDA kernel
void launch_lower_triangular_matmul(
    const torch::Tensor A, 
    const torch::Tensor B, 
    torch::Tensor C, 
    int N) 
{
    dim3 block(T, T);
    dim3 grid((N + T - 1) / T, (N + T - 1) / T);

    lower_triangular_matmul<<<grid, block>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        N);
    cudaDeviceSynchronize();
}
"""

# Compile the CUDA code using load_inline
lower_triangular_matmul = load_inline(
    name="lower_triangular_matmul",
    cpp_sources="""
    extern "C" {
        void launch_lower_triangular_matmul(
            torch::Tensor A, 
            torch::Tensor B, 
            torch::Tensor C, 
            int N);
    }
    """,
    cuda_sources=lower_triangular_matmul_source,
    functions=["launch_lower_triangular_matmul"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.N = 4096  # Fixed matrix size as per problem statement

    def forward(self, A, B):
        # Ensure inputs are contiguous for optimal memory access
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty_like(A)  # Output tensor
        # Launch the custom CUDA kernel
        lower_triangular_matmul.launch_lower_triangular_matmul(A, B, C, self.N)
        return C

def get_inputs():
    # Generate lower triangular matrices as inputs
    A = torch.tril(torch.rand(4096, 4096)).cuda()
    B = torch.tril(torch.rand(4096, 4096)).cuda()
    return [A, B]

def get_init_inputs():
    return []