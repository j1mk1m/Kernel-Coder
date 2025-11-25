import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        N = 4096
        symmetric_matmul_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename T>
        __global__ void symmetric_matmul_kernel(T* A, T* B, T* C, int N) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx >= (N * (N + 1)) / 2) return;

            int i = 0, j = 0;
            int low = 0, high = N;
            while (low <= high) {
                int mid = (low + high) / 2;
                int cum = mid * N - (mid * (mid - 1)) / 2;
                if (cum <= idx) {
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }
            i = high;
            int cum_before_i = high * N - (high * (high - 1)) / 2;
            j = idx - cum_before_i + high;

            T sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
            if (i != j) {
                C[j * N + i] = sum;
            }
        }

        extern "C" {
            torch::Tensor symmetric_matmul_cuda(torch::Tensor A, torch::Tensor B, int N) {
                const int threads_per_block = 256;
                const int num_blocks = ((N * (N + 1) / 2) + threads_per_block - 1) / threads_per_block;

                auto C = torch::empty({N, N}, A.options());
                symmetric_matmul_kernel<float><<<num_blocks, threads_per_block>>>(
                    A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
                );
                return C;
            }
        }
        """

        self.symmetric_matmul = load_inline(
            name="symmetric_matmul",
            cpp_sources="extern torch::Tensor symmetric_matmul_cuda(torch::Tensor, torch::Tensor, int);",
            cuda_sources=symmetric_matmul_source,
            functions=["symmetric_matmul_cuda"],
            verbose=False,
        )

    def forward(self, A, B):
        return self.symmetric_matmul.symmetric_matmul_cuda(A, B, N=4096)

def get_inputs():
    N = 4096
    A = torch.rand(N, N, dtype=torch.float32).cuda()
    A = (A + A.T) / 2  # Ensure symmetry
    B = torch.rand(N, N, dtype=torch.float32).cuda()
    B = (B + B.T) / 2  # Ensure symmetry
    return [A, B]

def get_init_inputs():
    return []