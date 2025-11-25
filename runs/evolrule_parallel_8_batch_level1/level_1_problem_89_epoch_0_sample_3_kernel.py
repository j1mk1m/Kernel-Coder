import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cumsum_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void block_prefix_sum_step1(
    const T* input,
    T* output_partial,
    T* block_sums,
    int dim_size,
    int batch_size,
    int block_size,
    int num_blocks_per_row) {
    int row = blockIdx.x / num_blocks_per_row;
    int block_idx_in_row = blockIdx.x % num_blocks_per_row;

    int row_offset = row * dim_size;
    int block_offset = block_idx_in_row * block_size;
    int global_idx = row_offset + block_offset + threadIdx.x;

    extern __shared__ T sdata[];
    sdata[threadIdx.x] = input[global_idx];
    __syncthreads();

    for (int offset = 1; offset <= blockDim.x; offset *= 2) {
        if (threadIdx.x >= offset) {
            sdata[threadIdx.x] += sdata[threadIdx.x - offset];
        }
        __syncthreads();
    }

    output_partial[global_idx] = sdata[threadIdx.x];

    if (threadIdx.x == blockDim.x - 1) {
        block_sums[blockIdx.x] = sdata[threadIdx.x];
    }
}

template <typename T>
__global__ void compute_carry_over(
    T* block_sums,
    T* carry_over,
    int num_rows,
    int num_blocks_per_row) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int row_block_start = row * num_blocks_per_row;

    extern __shared__ T sdata[];
    sdata[tid] = block_sums[row_block_start + tid];
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        if (tid >= offset) {
            sdata[tid] += sdata[tid - offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        carry_over[row_block_start] = 0;
    } else {
        carry_over[row_block_start + tid] = sdata[tid - 1];
    }
}

template <typename T>
__global__ void adjust_segments(
    const T* output_partial,
    const T* carry_over,
    T* output_final,
    int dim_size,
    int batch_size,
    int block_size,
    int num_blocks_per_row) {
    int row = blockIdx.x / num_blocks_per_row;
    int block_idx_in_row = blockIdx.x % num_blocks_per_row;

    int row_offset = row * dim_size;
    int block_offset = block_idx_in_row * block_size;
    int global_idx = row_offset + block_offset + threadIdx.x;

    T val = output_partial[global_idx];
    T carry = carry_over[row * num_blocks_per_row + block_idx_in_row];
    val += carry;
    output_final[global_idx] = val;
}

torch::Tensor cumsum_cuda(torch::Tensor input, int dim) {
    int batch_size = input.size(0);
    int dim_size = input.size(1);

    const int block_size = 1024;
    const int num_blocks_per_row = (dim_size + block_size - 1) / block_size;

    dim3 block(block_size);
    dim3 grid_step1(batch_size * num_blocks_per_row);
    dim3 grid_carry_over(batch_size);
    dim3 block_carry(num_blocks_per_row);
    dim3 grid_adjust(grid_step1);

    auto output_partial = torch::empty_like(input);
    auto block_sums = torch::empty({batch_size * num_blocks_per_row}, input.options());
    auto carry_over = torch::empty_like(block_sums);
    auto output_final = torch::empty_like(input);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cumsum_cuda", ([&] {
        block_prefix_sum_step1<scalar_t><<<grid_step1, block, block_size * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output_partial.data_ptr<scalar_t>(),
            block_sums.data_ptr<scalar_t>(),
            dim_size,
            batch_size,
            block_size,
            num_blocks_per_row
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "compute_carry_over", ([&] {
        compute_carry_over<scalar_t><<<grid_carry_over, block_carry, num_blocks_per_row * sizeof(scalar_t)>>>(
            block_sums.data_ptr<scalar_t>(),
            carry_over.data_ptr<scalar_t>(),
            batch_size,
            num_blocks_per_row
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adjust_segments", ([&] {
        adjust_segments<scalar_t><<<grid_adjust, block, 0>>>(
            output_partial.data_ptr<scalar_t>(),
            carry_over.data_ptr<scalar_t>(),
            output_final.data_ptr<scalar_t>(),
            dim_size,
            batch_size,
            block_size,
            num_blocks_per_row
        );
    }));

    return output_final;
}
"""

cumsum_cuda_cpp_source = """
#include <torch/extension.h>
"""

cumsum_cuda = load_inline(
    name="cumsum_cuda",
    cpp_sources=cumsum_cuda_cpp_source,
    cuda_sources=cumsum_cuda_source,
    functions=["cumsum_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.cumsum_cuda = cumsum_cuda

    def forward(self, x):
        if x.is_cuda:
            return self.cumsum_cuda.cumsum_cuda(x, self.dim)
        else:
            return torch.cumsum(x, dim=self.dim)

def get_inputs():
    batch_size = 32768
    input_shape = (32768,)
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [1]