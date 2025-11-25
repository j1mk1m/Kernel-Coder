import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for tiled matrix multiplication using shared memory
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void tiled_matmul_kernel(const T* __restrict__ A, const T* __restrict__ B, T* C,
                                   int M, int N, int K, int tile_size = 32) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    __shared__ T shared_A[tile_size][tile_size];
    __shared__ T shared_B[tile_size][tile_size];

    T sum = 0;
    for (int p = 0; p < (K - 1) / tile_size + 1; ++p) {
        // Load tiles into shared memory
        if (p * tile_size + tx < K && by * tile_size + ty < M) {
            shared_A[ty][tx] = A[(by * tile_size + ty) * K + p * tile_size + tx];
        } else {
            shared_A[ty][tx] = 0;
        }
        if (p * tile_size + ty < K && bx * tile_size + tx < N) {
            shared_B[ty][tx] = B[(p * tile_size + ty) * N + bx * tile_size + tx];
        } else {
            shared_B[ty][tx] = 0;
        }

        __syncthreads();

        // Compute the dot product of the tiles
        for (int k = 0; k < tile_size; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }

        __syncthreads();
    }

    // Write the result to global memory
    if (by * tile_size + ty < M && bx * tile_size + tx < N) {
        C[(by * tile_size + ty) * N + bx * tile_size + tx] = sum;
    }
}

torch::Tensor tiled_matmul_cuda(torch::Tensor A, torch::Tensor B, int tile_size = 32) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({M, N}, options);

    dim3 block_dim(tile_size, tile_size);
    dim3 grid_dim((N - 1) / tile_size + 1, (M - 1) / tile_size + 1);

    if (A.dtype() == torch::kFloat32) {
        tiled_matmul_kernel<float><<<grid_dim, block_dim>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K, tile_size);
    } else if (A.dtype() == torch::kHalf) {
        tiled_matmul_kernel<__half><<<grid_dim, block_dim>>>(
            reinterpret_cast<const __half*>(A.data_ptr()),
            reinterpret_cast<const __half*>(B.data_ptr()),
            reinterpret_cast<__half*>(C.data_ptr()), M, N, K, tile_size);
    } else {
        throw std::runtime_error("Unsupported data type");
    }

    cudaDeviceSynchronize();
    return C;
}
"""

matmul_cpp_source = (
    "torch::Tensor tiled_matmul_cuda(torch::Tensor A, torch::Tensor B, int tile_size = 32);"
)

# Compile the inline CUDA code for tiled matrix multiplication
tiled_matmul = load_inline(
    name="tiled_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["tiled_matmul_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14", "--expt-relaxed-constexpr"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = tiled_matmul
        self.tile_size = 32  # Tune this based on GPU architecture and matrix size

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Cast to half for faster computation if needed (adjust based on precision requirements)
        A = A.cuda().half()
        B = B.cuda().half()
        C = self.matmul.tiled_matmul_cuda(A, B, self.tile_size)
        return C.float()  # Cast back to float if required by the model's next layer

# Update get_inputs to move tensors to CUDA and cast to half
def get_inputs():
    A = torch.rand(M, K).cuda().half()
    B = torch.rand(K, N).cuda().half()
    return [A, B]

# Update get_init_inputs if needed, but in this case it's still empty
def get_init_inputs():
    return []