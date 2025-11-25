import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for max reduction along a specified dimension
max_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

template <typename scalar_t>
__device__ scalar_t warp_reduce(scalar_t val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <typename scalar_t>
__global__ void block_max_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* output,
    int64_t dim_size,
    int64_t outer_dim,
    int64_t inner_dim,
    int64_t reduce_dim,
    int64_t total_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int outer = idx / inner_dim;
    int inner = idx % inner_dim;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    if (reduce_dim == 0) {
        for (int i = 0; i < dim_size; ++i) {
            int pos = i * outer_dim + outer * dim_size + inner;
            scalar_t val = input[pos];
            if (val > max_val) max_val = val;
        }
    } else if (reduce_dim == 1) {
        for (int i = 0; i < dim_size; ++i) {
            int pos = outer * inner_dim * dim_size + i * inner_dim + inner;
            scalar_t val = input[pos];
            if (val > max_val) max_val = val;
        }
    } else {
        // General case for arbitrary dimension
        int stride = 1;
        for (int d = reduce_dim; d > 0; --d) {
            stride *= input->size(d);
        }
        for (int i = 0; i < dim_size; ++i) {
            int pos = idx + i * stride;
            scalar_t val = input[pos];
            if (val > max_val) max_val = val;
        }
    }

    // Warp-level reduction
    max_val = warp_reduce<scalar_t>(max_val);

    // Write result only if thread is in first warp
    if ((threadIdx.x & 0x1f) == 0) {
        output[idx] = max_val;
    }
}

template <typename scalar_t>
__global__ void final_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* output,
    int64_t total_elements,
    int64_t reduce_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    scalar_t max_val = input[idx];
    for (int i = 1; i < reduce_size; ++i) {
        int pos = idx + i * total_elements;
        max_val = fmaxf(max_val, input[pos]);
    }
    output[idx] = max_val;
}

std::vector<int64_t> compute_grid(const torch::Tensor& input, int64_t dim) {
    auto input_shape = input.sizes().vec();
    int64_t dim_size = input.size(dim);
    int64_t outer_dim = 1;
    for (int d = 0; d < dim; ++d) {
        outer_dim *= input.size(d);
    }
    int64_t inner_dim = 1;
    for (int d = dim + 1; d < input.dim(); ++d) {
        inner_dim *= input.size(d);
    }
    int64_t total_elements = outer_dim * inner_dim;
    return {outer_dim, inner_dim, dim_size, total_elements};
}

torch::Tensor max_reduction_cuda(torch::Tensor input, int64_t dim) {
    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto input_shape = input.sizes().vec();
    int64_t dim_size = input.size(dim);
    auto dims = compute_grid(input, dim);
    int64_t outer_dim = dims[0], inner_dim = dims[1], reduce_size = dims[2], total_elements = dims[3];

    auto output_size = input_shape;
    output_size.erase(output_size.begin() + dim);
    auto output = torch::empty(output_size, output_options);

    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    // First pass: block-level reduction
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "block_max_kernel", ([&] {
        block_max_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            outer_dim,
            inner_dim,
            dim,
            total_elements);
    }));

    // Second pass: final reduction across warps
    int final_block_size = 256;
    int final_num_blocks = (total_elements + final_block_size - 1) / final_block_size;
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "final_reduction_kernel", ([&] {
        final_reduction_kernel<scalar_t><<<final_num_blocks, final_block_size>>>(
            output.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total_elements,
            reduce_size);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

max_reduction_cpp_source = """
torch::Tensor max_reduction_cuda(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code for max reduction
max_reduction = load_inline(
    name="max_reduction",
    cpp_sources=max_reduction_cpp_source,
    cuda_sources=max_reduction_source,
    functions=["max_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.max_reduction = max_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_reduction.max_reduction_cuda(x, self.dim)

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]