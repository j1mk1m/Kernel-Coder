import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

triangular_matmul_source = """
#include <torch/extension.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void triangular_matmul(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N * (N + 1) / 2) {
        return;
    }

    // Compute i and j from tid
    float temp = sqrtf(8.f * tid + 1.f);
    int i_approx = static_cast<int>((temp - 3.f) / 2.f);

    // Adjust i_approx to find the correct row
    while ((i_approx + 1) * (i_approx + 2) / 2 <= tid) {
        i_approx++;
    }
    while (i_approx * (i_approx + 1) / 2 > tid) {
        i_approx--;
    }

    int j = tid - i_approx * (i_approx + 1) / 2;
    int i = i_approx;

    if (i < j) {
        return;
    }

    // Compute the sum over k from j to i
    float sum = 0.f;
    for (int k = j; k <= i; ++k) {
        float a_val = A[i * N + k];
        float b_val = B[k * N + j];
        sum += a_val * b_val;
    }

    C[i * N + j] = sum;
}

torch::Tensor triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    if (A.size(0) != A.size(1) || B.size(0) != B.size(1) || A.size(0) != B.size(0)) {
        throw std::runtime_error("Input matrices must be square and of the same size");
    }
    if (!A.is_contiguous() || !B.is_contiguous()) {
        throw std::runtime_error("Input tensors must be contiguous");
    }

    auto C = torch::zeros({N, N}, A.options());

    int total_elements = N * (N + 1) / 2;
    const int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    triangular_matmul<<<num_blocks, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return C;
}
"""

triangular_matmul_cpp_source = """
extern "C" {
    torch::Tensor triangular_matmul_cuda(torch::Tensor A, torch::Tensor B);
}
"""

triangular_matmul = load_inline(
    name="triangular_matmul",
    cpp_sources=triangular_matmul_cpp_source,
    cuda_sources=triangular_matmul_source,
    functions=["triangular_matmul_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.triangular_matmul = triangular_matmul

    def forward(self, A, B):
        return self.triangular_matmul.triangular_matmul_cuda(A, B)

def get_inputs():
    M = 4096
    A = torch.rand(M, M).cuda()
    B = torch.rand(M, M).cuda()
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]

def get_init_inputs():
    return []