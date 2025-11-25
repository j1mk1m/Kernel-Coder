import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mean_last_dim_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_last_dim_parallel(const float* input, float* output, int output_size, int dim_size) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < dim_size; i += blockDim.x) {
        sum += input[block_id * dim_size + i];
    }

    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[block_id] = shared[0] / dim_size;
    }
}

torch::Tensor mean_last_dim_cuda(torch::Tensor input, int dim_size) {
    auto output_size = input.sizes().slice(0, input.dim()-1);
    auto output = torch::empty(output_size, input.options());

    const int threads_per_block = 256;
    const int blocks_per_grid = output.numel();
    int smem_size = threads_per_block * sizeof(float);

    mean_last_dim_parallel<<<blocks_per_grid, threads_per_block, smem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        output.numel(),
        dim_size
    );

    return output;
}
"""

mean_last_dim_cpp_source = "torch::Tensor mean_last_dim_cuda(torch::Tensor input, int dim_size);"

mean_last_dim = load_inline(
    name="mean_last_dim",
    cpp_sources=mean_last_dim_cpp_source,
    cuda_sources=mean_last_dim_source,
    functions=["mean_last_dim_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mean_last_dim = mean_last_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim
        ndim = x.dim()

        # Permute input to move reduction dim to last
        permutation = list(range(ndim))
        permutation[-1], permutation[dim] = permutation[dim], permutation[-1]
        permuted_x = x.permute(permutation).contiguous()
        dim_size = permuted_x.size(-1)

        # Launch kernel
        output = self.mean_last_dim.mean_last_dim_cuda(permuted_x, dim_size)

        # Compute permutation to revert to original order except the reduced dimension
        current_dims = permutation[:-1]
        desired_dims = list(range(ndim))
        desired_dims.pop(dim)
        output_permutation = []
        for d in desired_dims:
            output_permutation.append(current_dims.index(d))

        # Apply permutation
        output = output.permute(output_permutation).contiguous()

        return output