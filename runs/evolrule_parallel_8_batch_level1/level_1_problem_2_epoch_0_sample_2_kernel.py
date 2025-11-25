import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

        # Define and compile the custom CUDA kernel
        matmul_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void custom_matmul_kernel(
            const scalar_t* __restrict__ A,
            const scalar_t* __restrict__ B,
            scalar_t* __restrict__ C,
            int M, int K, int N) {
            
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < M && col < N) {
                scalar_t sum = 0;
                for (int k = 0; k < K; ++k) {
                    sum += A[row * K + k] * B[k * N + col];
                }
                C[row * N + col] = sum;
            }
        }

        torch::Tensor custom_matmul_cuda(torch::Tensor A, torch::Tensor B) {
            const int M = A.size(0);
            const int K = A.size(1);
            const int N = B.size(1);

            auto C = torch::empty({M, N}, A.options());

            dim3 threads(32, 8);
            dim3 blocks(
                (N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);

            AT_DISPATCH_FLOATING_TYPES(A.type(), "custom_matmul_cuda", ([&] {
                custom_matmul_kernel<scalar_t><<<blocks, threads>>>(
                    A.data<scalar_t>(),
                    B.data<scalar_t>(),
                    C.data<scalar_t>(),
                    M, K, N);
            }));

            return C;
        }
        """

        matmul_cpp_source = "torch::Tensor custom_matmul_cuda(torch::Tensor A, torch::Tensor B);"

        # Load the CUDA kernel
        self.custom_matmul = load_inline(
            name="custom_matmul",
            cpp_sources=matmul_cpp_source,
            cuda_sources=matmul_source,
            functions=["custom_matmul_cuda"],
            verbose=False
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.custom_matmul.custom_matmul_cuda(A, B)

def get_inputs():
    M = 1024 * 2
    K = 4096 * 2
    N = 2048 * 2
    A = torch.rand(M, K, device='cuda')
    B = torch.rand(K, N, device='cuda')
    return [A, B]

def get_init_inputs():
    return []