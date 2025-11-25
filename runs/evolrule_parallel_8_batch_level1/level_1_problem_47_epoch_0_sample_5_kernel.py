import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for sum reduction along a specified dimension
sum_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

template<typename scalar_t>
__global__ void reduce_sum_kernel(const scalar_t* __restrict__ input,
                                 scalar_t* __restrict__ output,
                                 int64_t dim_size,
                                 int64_t outer_dim,
                                 int64_t inner_dim) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_dim * inner_dim) return;

    int64_t outer = idx / inner_dim;
    int64_t inner = idx % inner_dim;

    scalar_t sum = 0;
    for (int64_t d = 0; d < dim_size; ++d) {
        int64_t input_idx = outer * dim_size * inner_dim + d * inner_dim + inner;
        sum += input[input_idx];
    }
    output[idx] = sum;
}

torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    auto input_dims = input.sizes().vec();
    int64_t outer_dim = 1, inner_dim = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_dim *= input_dims[i];
    }
    for (int64_t i = dim + 1; i < input_dims.size(); ++i) {
        inner_dim *= input_dims[i];
    }
    int64_t dim_size = input_dims[dim];

    auto output_shape = input_dims;
    output_shape[dim] = 1;
    auto output = torch::empty(output_shape, input.options());

    int64_t total_elements = outer_dim * inner_dim;
    const int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        reduce_sum_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            dim_size,
            outer_dim,
            inner_dim);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

sum_reduction_cpp_source = """
torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code for sum reduction
sum_reduce = load_inline(
    name="sum_reduce",
    cpp_sources=sum_reduction_cpp_source,
    cuda_sources=sum_reduction_source,
    functions=["sum_reduce_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduce_op = sum_reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduce_op.sum_reduce_cuda(x, self.dim)

# The original get_inputs and get_init_inputs are unchanged, so they are omitted here
# However, the user should ensure that inputs are moved to CUDA if necessary in their setup
# For example, in get_inputs(), tensors should be created with .cuda()