import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        matmul_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        __global__ void custom_matmul_kernel(
            const float* A, const float* B, float* C,
            int M, int K, int N
        ) {
            int i = blockIdx.y * blockDim.y + threadIdx.y;
            int j = blockIdx.x * blockDim.x + threadIdx.x;

            if (i < M && j < N) {
                float sum = 0.0;
                for (int k = 0; k < K; ++k) {
                    sum += A[k * M + i] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }

        torch::Tensor custom_matmul_cuda(
            torch::Tensor A, torch::Tensor B
        ) {
            const int M = A.size(1);
            const int K_val = A.size(0);
            const int N_val = B.size(1);

            auto C = torch::empty({M, N_val}, A.options());

            dim3 threads(32, 8);
            dim3 blocks(
                (N_val + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y
            );

            custom_matmul_kernel<<<blocks, threads>>>(
                A.data_ptr<float>(), B.data_ptr<float>(),
                C.data_ptr<float>(), M, K_val, N_val
            );

            cudaDeviceSynchronize();
            return C;
        }
        """

        matmul_cpp_source = (
            "torch::Tensor custom_matmul_cuda("
            "torch::Tensor A, torch::Tensor B);"
        )

        self.custom_matmul = load_inline(
            name="custom_matmul",
            cpp_sources=matmul_cpp_source,
            cuda_sources=matmul_source,
            functions=["custom_matmul_cuda"],
            verbose=False,
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.custom_matmul.custom_matmul_cuda(A, B)

def get_inputs():
    A = torch.rand(K, M).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []