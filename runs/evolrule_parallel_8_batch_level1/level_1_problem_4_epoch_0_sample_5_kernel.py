import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for matrix-vector multiplication optimized for large K
matvecmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>  // For warp-level matrix multiply

template <int BlockSize>
__global__ void matvecmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* C, int M, int K) {
    extern __shared__ float shared[];
    float* sA = shared;
    float* sB = shared + BlockSize * 32;  // Reserve space for B if needed

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int stride = gridDim.x * BlockSize;

    float sum = 0.0f;

    // Process K elements in chunks to fit in shared memory
    for (int start = 0; start < K; start += BlockSize * 32) {
        int k = start + tid;
        if (k < K) {
            sA[tid] = A[bid * K + k];  // Load A's row element
            sB[tid] = B[k];           // Load B's element (since B is Kx1)
        }
        __syncthreads();

        // Compute partial sum using warp-level parallelism
        for (int i = 0; i < BlockSize * 32; i += blockDim.x) {
            int idx = tid + i;
            if (idx < BlockSize * 32 && k < K) {
                sum += sA[idx] * sB[idx];
            }
        }
        __syncthreads();
    }

    if (bid * BlockSize + tid < M) {
        C[bid * BlockSize + tid] = sum;
    }
}

torch::Tensor matvecmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    auto C = torch::zeros({M, 1}, A.options());

    const int BlockSize = 256;
    const dim3 blocks((M + BlockSize - 1) / BlockSize);
    const dim3 threads(BlockSize);

    // Shared memory size: 2 * BlockSize * 32 (for A and B tiles)
    size_t shared_size = 2 * BlockSize * 32 * sizeof(float);

    matvecmul_kernel<BlockSize><<<blocks, threads, shared_size, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K);
    
    return C;
}
"""

matvecmul_cpp_source = """
torch::Tensor matvecmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
matvecmul = load_inline(
    name="matvecmul",
    cpp_sources=matvecmul_cpp_source,
    cuda_sources=matvecmul_source,
    functions=["matvecmul_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14", "-arch=sm_75"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matvecmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure B is a vector (Kx1) instead of (K,)
        if B.dim() == 1:
            B = B.view(-1, 1)
        return self.matmul.matvecmul_cuda(A, B)

# Update input generation to use CUDA tensors
def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, 1).cuda()
    return [A, B]

def get_init_inputs():
    return []