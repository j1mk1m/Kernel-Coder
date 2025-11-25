import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void upper_tri_mult_kernel(float *A, float *B_T, float *C, int N) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N * (N + 1) / 2) return;

    // Compute i and j from t
    int low = 0;
    int high = N;
    int i = 0;
    while (low < high) {
        int mid = (low + high + 1) / 2;
        int m = mid - 1;
        int s = (m + 1) * (2 * N - m) / 2;
        if (s > t) {
            high = mid - 1;
        } else {
            low = mid;
        }
    }
    i = low;

    int s_prev = i * (2 * N - i + 1) / 2;
    int pos_in_row = t - s_prev;
    int j = i + pos_in_row;

    float sum = 0.0f;
    for (int k = i; k <= j; ++k) {
        float a_val = A[i * N + k];
        float b_val = B_T[j * N + k];
        sum += a_val * b_val;
    }

    C[i * N + j] = sum;
}

extern "C" {
    void upper_tri_mult_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N) {
        const int threadsPerBlock = 256;
        const int num_elements = N * (N + 1) / 2;
        const int num_blocks = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

        // Transpose B to B_T
        torch::Tensor B_T = B.t().contiguous();

        upper_tri_mult_kernel<<<num_blocks, threadsPerBlock>>>(
            A.data_ptr<float>(),
            B_T.data_ptr<float>(),
            C.data_ptr<float>(),
            N
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err));
        }
    }
}
"""

module = load_inline(
    name='upper_tri_mult',
    cpp_sources='',
    cuda_sources=kernel_source,
    functions=['upper_tri_mult_cuda'],
    verbose=True,
    extra_cuda_cflags=['-std=c++14']
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        N = A.size(0)
        C = torch.zeros(N, N, device=A.device, dtype=A.dtype)
        module.upper_tri_mult_cuda(A, B, C, N)
        return C