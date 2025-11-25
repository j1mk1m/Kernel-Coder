import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void min_reduction_kernel(const float* x, float* out, int B, int D1, int D2, int dim) {
    int i, j, k;
    if (dim == 0) {
        j = blockIdx.x / D2;
        k = blockIdx.x % D2;
        float min_val = INFINITY;
        for (int i_t = threadIdx.x; i_t < B; i_t += blockDim.x) {
            int idx = i_t * D1 * D2 + j * D2 + k;
            float val = x[idx];
            if (val < min_val) min_val = val;
        }
    } else if (dim == 1) {
        i = blockIdx.x / D2;
        j = blockIdx.x % D2;
        float min_val = INFINITY;
        for (int k_t = threadIdx.x; k_t < D1; k_t += blockDim.x) {
            int idx = i * D1 * D2 + k_t * D2 + j;
            float val = x[idx];
            if (val < min_val) min_val = val;
        }
    } else if (dim == 2) {
        i = blockIdx.x / D1;
        j = blockIdx.x % D1;
        float min_val = INFINITY;
        for (int k_t = threadIdx.x; k_t < D2; k_t += blockDim.x) {
            int idx = i * D1 * D2 + j * D2 + k_t;
            float val = x[idx];
            if (val < min_val) min_val = val;
        }
    } else {
        float min_val = INFINITY;
    }
    
    extern __shared__ float shared_min[];
    int tid = threadIdx.x;
    shared_min[tid] = min_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_min[tid + s] < shared_min[tid]) {
                shared_min[tid] = shared_min[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        if (dim == 0) {
            out[j * D2 + k] = shared_min[0];
        } else if (dim == 1) {
            out[i * D2 + j] = shared_min[0];
        } else if (dim == 2) {
            out[i * D1 + j] = shared_min[0];
        }
    }
}

torch::Tensor min_reduction_cuda(torch::Tensor x, int dim) {
    int B = x.size(0);
    int D1 = x.size(1);
    int D2 = x.size(2);
    
    int output_size0, output_size1;
    if (dim == 0) {
        output_size0 = D1;
        output_size1 = D2;
    } else if (dim == 1) {
        output_size0 = B;
        output_size1 = D2;
    } else {
        output_size0 = B;
        output_size1 = D1;
    }
    auto out = torch::empty({output_size0, output_size1}, x.options());
    
    int block_size = 256;
    int num_blocks = output_size0 * output_size1;
    
    size_t shared_mem = block_size * sizeof(float);
    
    min_reduction_kernel<<<num_blocks, block_size, shared_mem>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, D1, D2, dim
    );
    
    return out;
}
"""

min_reduction_cpp_source = """
torch::Tensor min_reduction_cuda(torch::Tensor x, int dim);
"""

min_reduction = load_inline(
    name="min_reduction",
    cpp_sources=min_reduction_cpp_source,
    cuda_sources=min_reduction_source,
    functions=["min_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.min_reduction = min_reduction  # The CUDA kernel module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_reduction.min_reduction_cuda(x, self.dim)