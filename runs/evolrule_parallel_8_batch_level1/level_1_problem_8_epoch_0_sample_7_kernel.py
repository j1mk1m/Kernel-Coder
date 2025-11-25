import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel source code
matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matrix_mult_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    // Block and thread indices
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread computes a single element in the output tile
    float Cvalue = 0.0f;

    // Iterate over tiles along K
    for (int m = 0; m < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        __shared__ float As[TILE_WIDTH][TILE_WIDTH];
        __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

        // Load A tile into shared memory
        int aRow = blockRow * TILE_WIDTH + row;
        int aCol = m * TILE_WIDTH + col;
        if (aRow < M && aCol < K) {
            As[row][col] = A[aRow * lda + aCol];
        } else {
            As[row][col] = 0.0f;
        }

        // Load B tile into shared memory
        int bRow = m * TILE_WIDTH + row;
        int bCol = blockCol * TILE_WIDTH + col;
        if (bRow < K && bCol < N) {
            Bs[row][col] = B[bRow * ldb + bCol];
        } else {
            Bs[row][col] = 0.0f;
        }

        __syncthreads();

        // Compute the partial product
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[row][k] * Bs[k][col];
        }

        __syncthreads();
    }

    // Write the result to global memory
    int cRow = blockRow * TILE_WIDTH + row;
    int cCol = blockCol * TILE_WIDTH + col;
    if (cRow < M && cCol < N) {
        C[cRow * ldc + cCol] = Cvalue;
    }
}

torch::Tensor matrix_mult_cuda(
    torch::Tensor A,
    torch::Tensor B
) {
    // Check that A and B are on the same device (GPU)
    auto device = A.device();
    AT_ASSERT(device.type() == torch::kCUDA);
    AT_ASSERT(A.device() == B.device());

    // Get matrix dimensions
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    // Check that A's columns (K) == B's rows (K)
    AT_ASSERT(A.size(1) == B.size(0));

    // Create output tensor
    auto C = torch::empty({M, N}, A.options());

    // Calculate leading dimensions
    int lda = A.stride(0);
    int ldb = B.stride(0);
    int ldc = C.stride(0);

    // Define block and grid dimensions
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH
    );

    // Launch the kernel
    matrix_mult_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc
    );

    // Check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    return C;
}
"""

matrix_mult_cpp_source = """
torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the CUDA code
matrix_mult = load_inline(
    name="matrix_mult",
    cpp_sources=matrix_mult_cpp_source,
    cuda_sources=matrix_mult_source,
    functions=["matrix_mult_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix_mult = matrix_mult

    def forward(self, A, B):
        return self.matrix_mult.matrix_mult_cuda(A, B)

# The existing get_inputs and get_init_inputs remain as in the original code
M = 8205
K = 2949
N = 5921

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []