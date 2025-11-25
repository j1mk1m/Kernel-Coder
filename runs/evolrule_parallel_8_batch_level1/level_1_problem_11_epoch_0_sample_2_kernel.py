import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tensor_matrix_mult_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ Output,
    int B_dim, int I_dim, int J_dim, int L_dim, int K_dim
) {
    extern __shared__ scalar_t shared_A[];
    
    // Each block handles a (b, i, j)
    int blockId = blockIdx.x;
    int b = blockId / (I_dim * J_dim);
    int ij = blockId % (I_dim * J_dim);
    int i = ij / J_dim;
    int j = ij % J_dim;
    
    // Each thread in the block handles a k
    int k = threadIdx.x;
    
    if (k >= K_dim) {
        return;
    }
    
    scalar_t sum = 0.0;
    
    for (int l = 0; l < L_dim; ++l) {
        if (threadIdx.x == 0) {
            int A_index = b * I_dim * J_dim * L_dim + i * J_dim * L_dim + j * L_dim + l;
            shared_A[l] = A[A_index];
        }
        __syncthreads();
        
        scalar_t A_val = shared_A[l];
        int B_index = l * K_dim + k;
        scalar_t B_val = B[B_index];
        
        sum += A_val * B_val;
    }
    
    // Compute output index
    int output_index = b * I_dim * J_dim * K_dim + i * J_dim * K_dim + j * K_dim + k;
    Output[output_index] = sum;
}

// C++ wrapper
at::Tensor tensor_matrix_mult_cuda(at::Tensor A, at::Tensor B) {
    const int B_dim = A.size(0);
    const int I_dim = A.size(1);
    const int J_dim = A.size(2);
    const int L_dim = A.size(3);
    const int K_dim = B.size(1);

    at::Tensor output = at::empty({B_dim, I_dim, J_dim, K_dim}, A.options());
    
    // Calculate grid and block dimensions
    int blocks_per_grid = B_dim * I_dim * J_dim;
    int threads_per_block = K_dim;
    
    // Check if threads_per_block is within limits (<=1024)
    if (threads_per_block > 1024) {
        AT_ERROR("Threads per block exceeds maximum limit.");
    }

    // Calculate shared memory size
    size_t shared_mem_size = L_dim * sizeof(float); // Assuming float for now
    
    // Launch kernel
    dim3 grid(blocks_per_grid);
    dim3 block(threads_per_block);
    tensor_matrix_mult_kernel<float><<<grid, block, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        B_dim, I_dim, J_dim, L_dim, K_dim
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }
    
    return output;
}
"""

cpp_source = """
#include <torch/extension.h>
at::Tensor tensor_matrix_mult_cuda(at::Tensor A, at::Tensor B);
"""

# Compile the CUDA code
tensor_matrix_mult = load_inline(
    name="tensor_matrix_mult",
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=["tensor_matrix_mult_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor_matrix_mult = tensor_matrix_mult

    def forward(self, A, B):
        return self.tensor_matrix_mult.tensor_matrix_mult_cuda(A, B)