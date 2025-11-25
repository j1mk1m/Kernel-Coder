import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

def get_launch_config(n):
    """Dynamically compute grid and block dimensions for CUDA kernels."""
    threads_per_block = 256  # Tuned for maximum occupancy on modern GPUs
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    return (blocks_per_grid, threads_per_block)

# Custom CUDA kernel for Sigmoid activation with optimizations
sigmoid_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        scalar_t z = x[idx];
        // Fast path for extreme values to avoid overflow
        if (z > 20) {
            y[idx] = 1.0;
        } else if (z < -20) {
            y[idx] = 0.0;
        } else {
            // Use exp approximation with fast math
            y[idx] = 1.0 / (1.0 + exp(-z));
        }
    }
}

// Use shared memory for better memory access
template <typename scalar_t>
__global__ void sigmoid_shared_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, int n) {
    extern __shared__ scalar_t shared[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    if (idx < n) {
        shared[tid] = x[idx];
    }
    __syncthreads();
    
    if (idx < n) {
        scalar_t z = shared[tid];
        if (z > 20) {
            y[idx] = 1.0;
        } else if (z < -20) {
            y[idx] = 0.0;
        } else {
            y[idx] = 1.0 / (1.0 + exp(-z));
        }
    }
}

template <typename scalar_t>
void launch_sigmoid_cuda(torch::Tensor x, torch::Tensor y) {
    auto n = x.numel();
    auto config = get_launch_config(n);
    
    // Use shared memory version for large tensors
    if (n > 1024) {
        int shared_size = config.block * sizeof(scalar_t);
        sigmoid_shared_kernel<scalar_t><<<config.grid, config.block, shared_size, torch::cuda::current_stream()>>>(
            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n);
    } else {
        sigmoid_kernel<scalar_t><<<config.grid, config.block, 0, torch::cuda::current_stream()>>>(
            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n);
    }
    
    // Error checking
    CUDAErrorHandler(cudaGetLastError());
    CUDAErrorHandler(cudaDeviceSynchronize());
}

// Custom error handler for CUDA calls
#define CUDAErrorHandler(cmd) \\
do { \\
    cudaError_t error = cmd; \\
    if(error != cudaSuccess) { \\
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \\
        throw std::runtime_error("CUDA error"); \\
    } \\
} while(0)

// Entry point for the custom operator
torch::Tensor sigmoid_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "sigmoid_cuda", [&] {
        launch_sigmoid_cuda<scalar_t>(x, y);
    });
    return y;
}
"""

# Inline compilation of the CUDA kernel
sigmoid_cuda = load_inline(
    name="sigmoid_cuda",
    cuda_sources=sigmoid_cuda_source,
    functions=["sigmoid_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid_cuda = sigmoid_cuda

    def forward(self, x):
        # Handle non-contiguous tensors by converting to contiguous
        if not x.is_contiguous():
            x = x.contiguous()
        return self.sigmoid_cuda.sigmoid_cuda(x)

def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []