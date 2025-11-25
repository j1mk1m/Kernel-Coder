import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for argmin
argmin_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void argmin_kernel(const scalar_t* input, int64_t* output, int batch_size, int dim1, int dim2) {
    int output_idx = blockIdx.x;
    int batch = output_idx / dim2;
    int dim2_idx = output_idx % dim2;

    __shared__ scalar_t s_min_vals[256];
    __shared__ int64_t s_min_indices[256];

    scalar_t min_val = std::numeric_limits<scalar_t>::infinity();
    int64_t min_idx = -1;

    for (int i = threadIdx.x; i < dim1; i += blockDim.x) {
        int input_offset = batch * dim1 * dim2 + i * dim2 + dim2_idx;
        scalar_t val = input[input_offset];
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
    }

    s_min_vals[threadIdx.x] = min_val;
    s_min_indices[threadIdx.x] = min_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_min_vals[threadIdx.x] > s_min_vals[threadIdx.x + s]) {
                s_min_vals[threadIdx.x] = s_min_vals[threadIdx.x + s];
                s_min_indices[threadIdx.x] = s_min_indices[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[output_idx] = s_min_indices[0];
    }
}

torch::Tensor argmin_cuda(torch::Tensor input, int dim) {
    if (dim != 1) {
        throw std::runtime_error("Only dim=1 is supported");
    }

    int batch_size = input.size(0);
    int dim1 = input.size(1);
    int dim2 = input.size(2);

    auto output = torch::empty({batch_size, dim2}, torch::dtype(torch::kLong).device(input.device()));

    const int block_size = 256;
    int num_blocks = batch_size * dim2;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmin_cuda", ([&] {
        argmin_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<int64_t>(),
            batch_size,
            dim1,
            dim2
        );
    }));

    return output;
}
"""

argmin_cpp_source = (
    "torch::Tensor argmin_cuda(torch::Tensor input, int dim);"
)

# Compile the CUDA code
argmin = load_inline(
    name="argmin",
    cpp_sources=argmin_cpp_source,
    cuda_sources=argmin_source,
    functions=["argmin_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim  # Must be 1 as per kernel's current implementation
        self.argmin = argmin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmin.argmin_cuda(x, self.dim)