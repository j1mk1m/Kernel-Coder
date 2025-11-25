import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

tensor_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define N 16
#define M 1024
#define K 2048
#define L 768

__global__ void tensor_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C
) {
    int block_id = blockIdx.x;
    int n = block_id / M;
    int m = block_id % M;

    int l = threadIdx.x;
    if (l >= L) return;

    extern __shared__ float shared_A[];

    int tid = threadIdx.x;
    int chunk_size = (K + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < chunk_size; ++i) {
        int k = tid * chunk_size + i;
        if (k < K) {
            shared_A[k] = A[n * M * K + m * K + k];
        }
    }
    __syncthreads();

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a_val = shared_A[k];
        float b_val = B[k * L + l];
        sum += a_val * b_val;
    }

    int c_idx = n * M * L + m * L + l;
    C[c_idx] = sum;
}

torch::Tensor tensor_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B
) {
    auto C = torch::empty({N, M, L}, A.options());

    dim3 block(L);
    dim3 grid(N * M);

    int smem_size = K * sizeof(float);

    tensor_matmul_kernel<<<grid, block, smem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>()
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_matmul_cuda", &tensor_matmul_cuda, "Tensor matmul CUDA kernel");
}
"""

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = load_inline(
            name="tensor_matmul",
            cpp_sources="",
            cuda_sources=tensor_matmul_source,
            functions=["tensor_matmul_cuda"],
            verbose=True
        )

    def forward(self, A, B):
        A = A.cuda()
        B = B.cuda()
        return self.module.tensor_matmul_cuda(A, B)