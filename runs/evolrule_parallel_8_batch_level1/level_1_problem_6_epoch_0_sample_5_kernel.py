import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.cuda_matmul = load_inline(
            name="fast_matmul",
            cpp_sources="""
            torch::Tensor fast_matmul_cuda(torch::Tensor A, torch::Tensor B);
            """,
            cuda_sources="""
            #include <torch/extension.h>
            #include <cuda.h>
            #include <cuda_runtime.h>

            #define TILE_DIM 32
            #define BLOCK_SIZE 16

            __global__ void fast_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                                              int M, int N, int K) {
                __shared__ float shared_A[TILE_DIM][TILE_DIM + 1];  // +1 for bank conflict mitigation
                __shared__ float shared_B[TILE_DIM][TILE_DIM + 1];

                int row = blockIdx.y * TILE_DIM + threadIdx.y;
                int col = blockIdx.x * TILE_DIM + threadIdx.x;
                float sum = 0.0;

                for (int t = 0; t < (K + TILE_DIM - 1)/TILE_DIM; ++t) {
                    // Load A tile into shared memory
                    if (row < M && (t * TILE_DIM + threadIdx.x) < K) {
                        shared_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_DIM + threadIdx.x];
                    } else {
                        shared_A[threadIdx.y][threadIdx.x] = 0.0;
                    }

                    // Load B tile into shared memory (transposed)
                    if ((t * TILE_DIM + threadIdx.y) < K && col < N) {
                        shared_B[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * N + col];
                    } else {
                        shared_B[threadIdx.y][threadIdx.x] = 0.0;
                    }

                    __syncthreads();

                    // Perform computations
                    for (int k = 0; k < TILE_DIM; ++k) {
                        sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
                    }

                    __syncthreads();
                }

                if (row < M && col < N) {
                    C[row * N + col] = sum;
                }
            }

            torch::Tensor fast_matmul_cuda(torch::Tensor A, torch::Tensor B) {
                const int M = A.size(0);
                const int K = A.size(1);
                const int N = B.size(1);

                // Output tensor
                auto C = torch::empty({M, N}, A.options());

                dim3 threads(TILE_DIM, TILE_DIM);
                dim3 blocks((N + TILE_DIM - 1)/TILE_DIM, (M + TILE_DIM - 1)/TILE_DIM);

                fast_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), 
                                                       C.data_ptr<float>(), M, N, K);

                cudaDeviceSynchronize();
                return C;
            }
            """,
            functions=["fast_matmul_cuda"],
            verbose=True
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.cuda_matmul.fast_matmul_cuda(A, B)

# Original model dimensions (must be moved to CUDA)
M = 256
N = 256
K = 131072 * 4

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []