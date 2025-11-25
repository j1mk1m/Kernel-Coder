import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

M = 16384 * 2
N = 16 * 2

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* Bt, float* C, int M, int N) {
    int i = blockIdx.x * 16 + threadIdx.x;
    int j = blockIdx.y * 16 + threadIdx.y;

    if (i < M && j < M) {
        float sum = 0.0f;
        const float4* A4 = reinterpret_cast<const float4*>(&A[i * N]);
        const float4* Bt4 = reinterpret_cast<const float4*>(&Bt[j * N]);

        for (int k = 0; k < N /4; ++k) {
            float4 a = A4[k];
            float4 b = Bt4[k];
            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        C[i * M + j] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto Bt = B.t().contiguous();
    int M_val = A.size(0);
    int N_val = A.size(1);
    TORCH_CHECK(Bt.size(0) == M_val && Bt.size(1) == N_val, "Dimensions must match after transpose");

    auto C = torch::empty({M_val, M_val}, A.options());

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        (M_val + threads_per_block.x - 1) / threads_per_block.x,
        (M_val + threads_per_block.y - 1) / threads_per_block.y
    );

    matmul_kernel<<<blocks_per_grid, threads_per_block>>>(
        A.data_ptr<float>(),
        Bt.data_ptr<float>(),
        C.data_ptr<float>(),
        M_val,
        N_val
    );

    return C;
}
"""

matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = matmul  # The loaded module

    def forward(self, A, B):
        return self.matmul.matmul_cuda(A, B)

def get_inputs():
    A = torch.rand(M, N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]

def get_init_inputs():
    return []