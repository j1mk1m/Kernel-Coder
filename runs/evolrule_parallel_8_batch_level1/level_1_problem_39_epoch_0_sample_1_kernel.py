import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L2 normalization
l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ scalar_t warp_reduce(scalar_t val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <typename scalar_t>
__global__ void l2_norm_kernel(const scalar_t* __restrict__ x,
                              scalar_t* __restrict__ y,
                              const int batch_size,
                              const int dim) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ scalar_t shared_data[32 * 1024]; // 32KB shared memory for 8192 floats (max 8192 elements per block?)

    // Each thread loads a portion of the data into shared memory
    const int total_elements = dim;
    const int threads_per_block = blockDim.x;
    const int start = batch_idx * dim;
    for (int i = tid; i < total_elements; i += threads_per_block) {
        shared_data[i] = x[start + i];
    }
    __syncthreads();

    // Compute the L2 norm for this batch element using warp-level reduction
    scalar_t norm_sqr = 0.0;
    for (int i = tid; i < total_elements; i += threads_per_block) {
        scalar_t val = shared_data[i];
        norm_sqr += val * val;
    }
    __syncthreads();

    // Warp reduction
    const int warp_size = 32;
    int warp_id = tid / warp_size;
    int lane_id = tid % warp_size;
    scalar_t warp_norm = 0.0;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        warp_norm += __shfl_down_sync(0xffffffff, norm_sqr, offset);
    }
    if (lane_id == 0) {
        norm_sqr = warp_norm;
    }
    __syncthreads();

    // All threads in block must have the same norm value
    if (lane_id == 0) {
        norm_sqr = warp_reduce<scalar_t>(norm_sqr);
    }
    __syncthreads();

    scalar_t norm = sqrt(norm_sqr);
    scalar_t inv_norm = (norm > 1e-12) ? 1.0 / norm : 0.0;

    // Write back normalized values
    for (int i = tid; i < total_elements; i += threads_per_block) {
        y[start + i] = shared_data[i] * inv_norm;
    }
    __syncthreads();
}

torch::Tensor l2_norm_cuda(torch::Tensor x) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    auto y = torch::empty_like(x);

    const int threads_per_block = 256;
    const dim3 blocks(batch_size);
    const dim3 threads(threads_per_block);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "l2_norm_cuda", ([&] {
        l2_norm_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            batch_size,
            dim);
    }));

    cudaDeviceSynchronize();
    return y;
}
"""

# Define the header for the CUDA source
l2_norm_header = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

at::Tensor l2_norm_cuda(at::Tensor x);
"""

# Compile the inline CUDA code
l2_norm = load_inline(
    name="l2_norm",
    cpp_sources=l2_norm_header,
    cuda_sources=l2_norm_source,
    functions=["l2_norm_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2_norm = l2_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2_norm.l2_norm_cuda(x)