import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void min_reduction_kernel(
    const scalar_t* input,
    scalar_t* output,
    int dim,
    int dims[3],
    int output_size
) {
    int output_idx = blockIdx.x;

    int remaining_dims[2];
    int coord0, coord1;

    if (dim == 0) {
        remaining_dims[0] = dims[1];
        remaining_dims[1] = dims[2];
        coord0 = output_idx / remaining_dims[1];
        coord1 = output_idx % remaining_dims[1];
    } else if (dim == 1) {
        remaining_dims[0] = dims[0];
        remaining_dims[1] = dims[2];
        coord0 = output_idx / remaining_dims[1];
        coord1 = output_idx % remaining_dims[1];
    } else if (dim == 2) {
        remaining_dims[0] = dims[0];
        remaining_dims[1] = dims[1];
        coord0 = output_idx / remaining_dims[1];
        coord1 = output_idx % remaining_dims[1];
    }

    int base = 0;
    int rdim_stride = 0;
    if (dim == 0) {
        base = coord0 * dims[2] + coord1;
        rdim_stride = dims[1] * dims[2];
    } else if (dim == 1) {
        base = coord0 * (dims[1] * dims[2]) + coord1;
        rdim_stride = dims[2];
    } else if (dim == 2) {
        base = coord0 * (dims[1] * dims[2]) + coord1 * dims[2];
        rdim_stride = 1;
    }

    __shared__ scalar_t shared_min[256];
    const int tid = threadIdx.x;

    scalar_t partial_min = std::numeric_limits<scalar_t>::infinity();

    for (int rdim = tid; rdim < dims[dim]; rdim += blockDim.x) {
        int input_idx = base + rdim * rdim_stride;
        scalar_t val = input[input_idx];
        if (val < partial_min) {
            partial_min = val;
        }
    }

    shared_min[tid] = partial_min;
    __syncthreads();

    int i = blockDim.x >> 1;
    while (i > 0) {
        if (tid < i) {
            if (shared_min[tid] > shared_min[tid + i]) {
                shared_min[tid] = shared_min[tid + i];
            }
        }
        __syncthreads();
        i >>= 1;
    }

    if (tid == 0) {
        output[output_idx] = shared_min[0];
    }
}

torch::Tensor min_reduction_cuda(torch::Tensor input, int dim) {
    int dims[3];
    dims[0] = input.size(0);
    dims[1] = input.size(1);
    dims[2] = input.size(2);

    int output_size = 1;
    for (int i = 0; i < 3; i++) {
        if (i != dim) {
            output_size *= input.size(i);
        }
    }

    auto output = torch::empty({output_size}, input.options());

    const int block_size = 256;
    dim3 grid(output_size);
    dim3 block(block_size);

    auto input_data = input.data_ptr<float>();
    auto output_data = output.data_ptr<float>();

    min_reduction_kernel<float><<<grid, block>>>(input_data, output_data, dim, dims, output_size);

    return output;
}
"""

min_reduction_cpp_source = """
torch::Tensor min_reduction_cuda(torch::Tensor input, int dim);
"""

min_reduction = load_inline(
    name="min_reduction",
    cuda_sources=min_reduction_source,
    cpp_sources=min_reduction_cpp_source,
    functions=["min_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return min_reduction.min_reduction_cuda(x, self.dim)