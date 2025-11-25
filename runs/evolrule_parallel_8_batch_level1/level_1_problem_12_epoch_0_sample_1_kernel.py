import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

diag_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(const float* A, const float* B, float* out, int N, int M) {
    extern __shared__ float shared_A[];
    
    int blockId = blockIdx.x;
    int i = blockId;
    
    if (i >= N) return;

    // Load A[i] into shared memory
    if (threadIdx.x == 0) {
        shared_A[0] = A[i];
    }
    __syncthreads();
    
    float a_val = shared_A[0];
    
    // Each thread handles M / blockDim.x elements
    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        int idx = i * M + j;
        out[idx] = B[idx] * a_val;
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure A and B are on the same device
    if (A.device() != B.device()) {
        AT_ERROR("A and B must be on the same device");
    }
    if (A.dim() != 1) {
        AT_ERROR("A must be a 1D tensor");
    }
    if (B.dim() != 2) {
        AT_ERROR("B must be a 2D tensor");
    }
    if (A.size(0) != B.size(0)) {
        AT_ERROR("A's length must match B's rows");
    }
    
    int N = A.size(0);
    int M = B.size(1);
    
    auto out = torch::empty_like(B);
    
    const int block_size = 256;
    int num_blocks = N; // Each block handles a row
    
    diag_matmul_kernel<<<num_blocks, block_size, sizeof(float)>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), N, M
    );
    
    return out;
}
"""

diag_matmul_header = "torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B);"

diag_matmul = load_inline(
    name="diag_matmul",
    cpp_sources=diag_matmul_header,
    cuda_sources=diag_matmul_source,
    functions=["diag_matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_flags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.diag_matmul = diag_matmul

    def forward(self, A, B):
        return self.diag_matmul.diag_matmul_cuda(A, B)