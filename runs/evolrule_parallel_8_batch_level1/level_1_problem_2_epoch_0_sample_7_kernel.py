import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA code
matrix_mult_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

// Error checking helper functions
static inline void cudaCheckError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

static inline void cudaCheckError(cudaStream_t stream) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

// Define block size
#define BLOCK_SIZE 32

__global__ void matrix_mult_kernel(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int blockRow = blockIdx.x * BLOCK_SIZE;
    int blockCol = blockIdx.y * BLOCK_SIZE;

    float temp = 0.0f;

    int numTilesK = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int p = 0; p < numTilesK; ++p) {
        // Load A into shared memory
        int aRow = blockRow + tx;
        int aCol = p * BLOCK_SIZE + ty;
        if (aRow < M && aCol < K) {
            shared_A[tx][ty] = A[aRow * K + aCol];
        } else {
            shared_A[tx][ty] = 0.0f;
        }

        // Load B into shared memory
        int bRow = p * BLOCK_SIZE + tx;
        int bCol = blockCol + ty;
        if (bRow < K && bCol < N) {
            shared_B[tx][ty] = B[bRow * N + bCol];
        } else {
            shared_B[tx][ty] = 0.0f;
        }

        __syncthreads();

        // Compute partial products
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            temp += shared_A[tx][k] * shared_B[k][ty];
        }
    }

    // Write result to C
    int cRow = blockRow + tx;
    int cCol = blockCol + ty;
    if (cRow < M && cCol < N) {
        C[cRow * N + cCol] = temp;
    }
}

torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B, cudaStream_t stream) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Check dimensions
    if (A.size(1) != B.size(0)) {
        std::cerr << "Matrix dimensions must match for multiplication" << std::endl;
        exit(-1);
    }

    auto C = torch::empty({M, N}, A.options());

    const int block_size = BLOCK_SIZE;
    dim3 grid(
        (M + block_size - 1) / block_size,
        (N + block_size - 1) / block_size
    );
    dim3 block(block_size, block_size);

    matrix_mult_kernel<<<grid, block, 0, stream>>>(
        A.data_ptr<float>(), B.data_ptr<float>(),
        C.data_ptr<float>(), M, K, N
    );

    // Check for errors and synchronize the stream
    cudaCheckError(stream);

    return C;
}
"""

matrix_mult_cuda_header = """
extern "C" {
    torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B, cudaStream_t stream);
}
"""

# Compile the CUDA code
matrix_mult = load_inline(
    name="matrix_mult",
    cpp_sources=[matrix_mult_cuda_header],
    cuda_sources=[matrix_mult_cuda_source],
    functions=["matrix_mult_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        stream = torch.cuda.current_stream()
        return matrix_mult.matrix_mult_cuda(A, B, stream)

# Update get_inputs to use CUDA tensors
def get_inputs():
    M = 1024 * 2
    K = 4096 * 2
    N = 2048 * 2
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []  # No special initialization needed