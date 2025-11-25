import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_add_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void upper_triangular_mm(
    const float* A,
    const float* B,
    float* C,
    const int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= (N * (N + 1)) / 2) {
        return;
    }

    // Find row i
    int low = 0;
    int high = N;
    while (low < high) {
        int mid = (low + high) / 2;
        int s = (mid + 1) * N - (mid * (mid + 1)) / 2;
        if (s <= tid) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    int i = low;

    // Compute S(i-1)
    int s_prev = (i) * N - (i * (i - 1)) / 2;

    int offset = tid - s_prev;
    int j = i + offset;

    float sum = 0.0f;
    for (int k = i; k <= j; ++k) {
        sum += A[i * N + k] * B[k * N + j];
    }

    C[i * N + j] = sum;
}

torch::Tensor upper_triangular_mm_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    int num_elements = N * (N + 1) / 2;
    int threads_per_block = 256;
    int blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    upper_triangular_mm<<<blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    cudaDeviceSynchronize();
    return C;
}
"""

elementwise_add_cpp_source = (
    "torch::Tensor upper_triangular_mm_cuda(torch::Tensor A, torch::Tensor B);"
)

upper_triangular_mm = load_inline(
    name="upper_triangular_mm",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["upper_triangular_mm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.upper_triangular_mm = upper_triangular_mm

    def forward(self, A, B):
        return self.upper_triangular_mm.upper_triangular_mm_cuda(A, B)