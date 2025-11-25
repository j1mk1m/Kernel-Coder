import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Load the CUDA kernel
        exclusive_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void exclusive_cumsum_kernel(
    scalar_t* out,
    const scalar_t* in,
    int64_t dim_size,
    int64_t outer_dim,
    int64_t inner_dim,
    int64_t total_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int outer = idx / (dim_size * inner_dim);
    int pos_in_dim = (idx / inner_dim) % dim_size;
    int inner = idx % inner_dim;

    // Compute the current position in the dimension
    scalar_t sum = 0;
    for (int i = 0; i < pos_in_dim; ++i) {
        int offset = outer * dim_size * inner_dim + i * inner_dim + inner;
        sum += in[offset];
    }
    out[idx] = sum;
}

torch::Tensor exclusive_cumsum_cuda(torch::Tensor input, int64_t dim) {
    auto dims = input.sizes().vec();
    int64_t dim_size = dims[dim];
    // Compute sizes for outer, inner dimensions
    int64_t outer_dim = 1;
    for (int i = 0; i < dim; ++i) {
        outer_dim *= dims[i];
    }
    int64_t inner_dim = 1;
    for (int i = dim + 1; i < dims.size(); ++i) {
        inner_dim *= dims[i];
    }
    int64_t total_elements = input.numel();

    auto out = torch::empty_like(input);

    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "exclusive_cumsum_cuda", ([&]{
        using scalar_t = scalar_type;
        exclusive_cumsum_kernel<scalar_t><<<grid_size, block_size>>>(
            out.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            dim_size,
            outer_dim,
            inner_dim,
            total_elements
        );
    }));

    cudaDeviceSynchronize();
    return out;
}
"""

        # Compile the CUDA kernel
        self.exclusive_cumsum = load_inline(
            name="exclusive_cumsum",
            cpp_sources="",
            cuda_sources=exclusive_cumsum_source,
            functions=["exclusive_cumsum_cuda"],
            verbose=True,
        )

    def forward(self, x):
        return self.exclusive_cumsum.exclusive_cumsum_cuda(x, self.dim)

def get_inputs():
    batch_size = 32768
    input_shape = (32768,)
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [1]