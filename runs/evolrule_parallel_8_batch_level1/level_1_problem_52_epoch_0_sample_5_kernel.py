import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmin_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h> // For FLT_MAX

__global__ void argmin_kernel(
    const float* x,
    int64_t* out,
    int batch_size,
    int dim1,
    int dim2
) {
    int d2 = blockIdx.x % dim2;
    int batch = blockIdx.x / dim2;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    extern __shared__ unsigned char s_data[];

    float* s_input = (float*)s_data;
    int64_t* s_min_idx = (int64_t*)(s_data + dim1 * sizeof(float));
    float* s_min_val = (float*)(s_data + dim1 * sizeof(float) + blockDim.x * sizeof(int64_t));

    // Load data into shared memory
    for (int i = tid; i < dim1; i += stride) {
        int idx_in_x = batch * dim1 * dim2 + i * dim2 + d2;
        s_input[i] = x[idx_in_x];
    }
    __syncthreads();

    // Compute local min and index
    float local_min = FLT_MAX;
    int64_t local_idx = -1;
    for (int i = tid; i < dim1; i += blockDim.x) {
        if (s_input[i] < local_min) {
            local_min = s_input[i];
            local_idx = i;
        }
    }

    // Write to shared memory for reduction
    s_min_val[tid] = local_min;
    s_min_idx[tid] = local_idx;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_min_val[tid + s] < s_min_val[tid]) {
                s_min_val[tid] = s_min_val[tid + s];
                s_min_idx[tid] = s_min_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[batch * dim2 + d2] = s_min_idx[0];
    }
}

torch::Tensor argmin_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int dim1 = x.size(1);
    int dim2 = x.size(2);

    auto out = torch::empty({batch_size, dim2}, torch::dtype(torch::kInt64).device(x.device()));

    const int block_size = 256;
    const int grid_size = batch_size * dim2;

    const int shared_mem_size = dim1 * sizeof(float) + block_size * (sizeof(float) + sizeof(int64_t));

    argmin_kernel<<<grid_size, block_size, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        out.data_ptr<int64_t>(),
        batch_size,
        dim1,
        dim2
    );

    return out;
}
"""

argmin_cpp_source = """
torch::Tensor argmin_cuda(torch::Tensor x);
"""

# Compile the CUDA kernel
argmin_cuda = load_inline(
    name="argmin_cuda",
    cpp_sources=argmin_cpp_source,
    cuda_sources=argmin_source,
    functions=["argmin_cuda"],
    verbose=True,
    extra_cflags=["-x", "c++"],
    extra_cuda_cflags=["-x", "cu"],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim  # Not used, but for compatibility

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return argmin_cuda.argmin_cuda(x)

def get_inputs():
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    dim = 1
    return [dim]