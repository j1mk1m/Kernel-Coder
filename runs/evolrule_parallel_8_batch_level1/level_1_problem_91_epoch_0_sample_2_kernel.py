import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

reverse_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void reverse_cumsum_kernel(
    scalar_t* input,
    scalar_t* output,
    int dim_size,
    int dim_stride,
    int num_slices) {
    extern __shared__ scalar_t sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid >= num_slices) return;

    // Load reversed slice into shared memory
    if (tid < dim_size) {
        int global_in = bid * dim_stride + (dim_size - 1 - tid);
        sdata[tid] = input[global_in];
    }
    __syncthreads();

    // Parallel prefix sum on the reversed slice
    for (int d = 1; d < blockDim.x; d *= 2) {
        int index = tid;
        if (index >= d) {
            sdata[index] += sdata[index - d];
        }
        __syncthreads();
    }

    // Write back the results
    if (tid < dim_size) {
        int global_out = bid * dim_stride + tid;
        output[global_out] = sdata[tid];
    }
}

torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim) {
    const auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];
    int64_t num_slices = 1;
    int64_t dim_stride = 1;

    // Calculate the number of slices (product of dimensions before 'dim')
    for (int i = 0; i < dim; ++i) {
        num_slices *= sizes[i];
    }

    // Calculate the stride (product of dimensions after 'dim' multiplied by dim_size)
    for (int i = dim + 1; i < sizes.size(); ++i) {
        dim_stride *= sizes[i];
    }
    dim_stride *= dim_size;

    const int block_size = 256;
    dim3 blocks(num_slices);
    dim3 threads(block_size);
    int smem_size = dim_size * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "reverse_cumsum_cuda", ([&] {
        reverse_cumsum_kernel<scalar_t><<<blocks, threads, smem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            dim_stride,
            num_slices);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

reverse_cumsum_cpp_source = """
torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim);
"""

reverse_cumsum = load_inline(
    name="reverse_cumsum",
    cpp_sources=reverse_cumsum_cpp_source,
    cuda_sources=reverse_cumsum_source,
    functions=["reverse_cumsum_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return reverse_cumsum.reverse_cumsum_cuda(x, self.dim)