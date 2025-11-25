import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        # Define the custom CUDA kernel for matrix multiplication (A^T * B)
        matmul_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void custom_matmul_kernel(
            const scalar_t* __restrict__ a,
            const scalar_t* __restrict__ b,
            scalar_t* __restrict__ c,
            int m,
            int k,
            int n
        ) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < m && col < n) {
                scalar_t sum = 0;
                for (int e = 0; e < k; ++e) {
                    sum += a[row * k + e] * b[e * n + col];
                }
                c[row * n + col] = sum;
            }
        }

        torch::Tensor custom_matmul_cuda(
            torch::Tensor a,
            torch::Tensor b,
            int m,
            int k,
            int n
        ) {
            const int threads = 32;
            dim3 blocks((n + threads - 1) / threads, (m + threads - 1) / threads);
            dim3 threadsPerBlock(threads, threads);

            auto c = torch::empty({m, n}, a.options());

            AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "custom_matmul_cuda", ([&] {
                custom_matmul_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
                    a.data_ptr<scalar_t>(),
                    b.data_ptr<scalar_t>(),
                    c.data_ptr<scalar_t>(),
                    m, k, n
                );
            }));

            return c;
        }
        """

        # Compile the inline CUDA code for matrix multiplication
        self.custom_matmul = load_inline(
            name="custom_matmul",
            cpp_sources="",
            cuda_sources=[matmul_source],
            functions=["custom_matmul_cuda"],
            verbose=False,
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Transpose A first
        A = A.t()
        m, k = A.shape
        _, n = B.shape
        return self.custom_matmul.custom_matmul_cuda(A, B, m, k, n)

def get_inputs():
    # Generate CUDA tensors for testing
    A = torch.rand(K, M).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []

# Global constants (must match original code's dimensions)
M = 1024 * 2
K = 4096 * 2
N = 2048 * 2