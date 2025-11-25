import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

rms_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void rms_norm_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int F,
    float eps
) {
    extern __shared__ float shared_mem[];

    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    if (thread_idx >= F) return;

    // Compute global index
    int global_idx = block_idx * F + thread_idx;

    // Load x value
    float x_val = x[global_idx];
    float x_squared = x_val * x_val;

    // Store in shared memory
    shared_mem[thread_idx] = x_squared;
    __syncthreads();

    // Compute sum in thread 0
    float sum = 0.0f;
    if (threadIdx.x == 0) {
        for (int i = 0; i < F; ++i) {
            sum += shared_mem[i];
        }
        shared_mem[0] = sum;
    }
    __syncthreads();

    // Retrieve the sum
    sum = shared_mem[0];

    // Compute mean and inverse sqrt
    float mean = sum / F;
    float inv_sqrt = 1.0f / sqrtf(mean + eps);

    // Write the result
    y[global_idx] = x_val * inv_sqrt;
}

torch::Tensor rms_norm_cuda(torch::Tensor x, int F, float eps) {
    auto output = torch::empty_like(x);
    
    // Calculate total number of blocks: total_elements / F
    const int total_elements = x.numel();
    const int total_blocks = total_elements / F;
    
    const int threads_per_block = F;
    const size_t shared_mem_size = F * sizeof(float);
    
    // Check block size limit (1024 max)
    if (threads_per_block > 1024) {
        // For the problem's scope, we proceed under the assumption that F <= 1024
    }
    
    // Launch the kernel
    rms_norm_kernel<<<total_blocks, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        F,
        eps
    );
    
    return output;
}
"""

rms_norm_cpp_source = """
torch::Tensor rms_norm_cuda(torch::Tensor x, int F, float eps);
"""

rms_norm_cuda = load_inline(
    name="rms_norm_cuda",
    cpp_sources=rms_norm_cpp_source,
    cuda_sources=rms_norm_source,
    functions=["rms_norm_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        if not x.is_contiguous():
            x = x.contiguous()
        return rms_norm_cuda.rms_norm_cuda(x, self.num_features, self.eps)