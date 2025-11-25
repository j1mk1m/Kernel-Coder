import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L2 normalization
l2norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

template <typename scalar_t>
__global__ void l2norm_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ out, scalar_t* __restrict__ norms,
                             int batch_size, int dim) {
    // Each thread processes one element
    int batch_idx = blockIdx.x;
    int element_idx = threadIdx.x;

    if (element_idx < dim) {
        scalar_t val = x[batch_idx * dim + element_idx];
        // Compute squared value and accumulate into shared memory
        extern __shared__ scalar_t shared[];
        scalar_t* sdata = shared;
        sdata[threadIdx.x] = val * val;
        __syncthreads();

        // Parallel reduction to compute the L2 norm for the row
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (element_idx == 0) {
            scalar_t norm = sqrt(sdata[0]);
            norms[batch_idx] = norm;
        }
        __syncthreads();

        // Normalize each element in the row
        scalar_t norm = norms[batch_idx];
        out[batch_idx * dim + element_idx] = val / norm;
    }
}

torch::Tensor l2norm_cuda(torch::Tensor x) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor out = torch::empty_like(x);
    torch::Tensor norms = torch::empty({batch_size}, options);

    dim3 blocks(batch_size);
    dim3 threads(dim); // Each row is processed by a block with dim threads

    // Allocate shared memory for the reduction
    size_t shared_mem = dim * sizeof(float);

    l2norm_kernel<float><<<blocks, threads, shared_mem, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), norms.data_ptr<float>(), batch_size, dim);

    return out;
}
"""

l2norm_cpp_source = (
    "torch::Tensor l2norm_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for L2 normalization
l2norm = load_inline(
    name="l2norm",
    cpp_sources=l2norm_cpp_source,
    cuda_sources=l2norm_source,
    functions=["l2norm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l2norm_cuda = l2norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2norm_cuda.l2norm_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []