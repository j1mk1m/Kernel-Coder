import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return cumprod_cuda(x, self.dim)

# CUDA kernel implementation
cumprod_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>

template <typename scalar_t>
__global__ void cumprod_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t total_elements,
    int64_t dim_size,
    int64_t outer_dim,
    int64_t inner_dim,
    int64_t dim) {

    extern __shared__ scalar_t shared[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int outer = bid / dim_size;
    int pos = bid % dim_size;

    int index = outer * dim_size * inner_dim + pos * inner_dim;

    scalar_t val = 1;
    if (pos < dim_size) {
        val = input[index];
    }

    for (int stride = 1; stride <= dim_size; stride *= 2) {
        __syncthreads();
        if (pos >= stride) {
            val *= shared[pos - stride];
        }
        __syncthreads();
    }

    for (int stride = dim_size / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (pos >= stride) {
            shared[pos] *= shared[pos - stride];
        } else if (stride > pos) {
            shared[pos] = 1;
        }
        __syncthreads();
    }

    if (pos < dim_size) {
        output[index] = val;
    }
}

std::vector<int64_t> get_strides(const torch::Tensor& x, int64_t dim) {
    auto sizes = x.sizes().vec();
    auto strides = x.strides().vec();
    int64_t outer_dim = 1;
    for (int i = 0; i < dim; ++i) {
        outer_dim *= sizes[i];
    }
    int64_t inner_dim = 1;
    for (int i = dim + 1; i < sizes.size(); ++i) {
        inner_dim *= sizes[i];
    }
    return {outer_dim, sizes[dim], inner_dim};
}

torch::Tensor cumprod_cuda(torch::Tensor input, int64_t dim) {
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty_like(input);

    auto strides = get_strides(input, dim);
    int64_t outer_dim = strides[0];
    int64_t dim_size = strides[1];
    int64_t inner_dim = strides[2];
    int64_t total_elements = input.numel();

    int block_size = 256;
    dim3 blocks(outer_dim * dim_size);
    dim3 threads(block_size);
    size_t shared_mem = block_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cumprod_cuda", ([&] {
        cumprod_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total_elements,
            dim_size,
            outer_dim,
            inner_dim,
            dim);
    }));

    return output;
}
"""

cumprod_cpp_source = """
#include <torch/extension.h>

torch::Tensor cumprod_cuda(torch::Tensor input, int64_t dim);
"""

cumprod = load_inline(
    name="cumprod",
    cpp_sources=cumprod_cpp_source,
    cuda_sources=cumprod_source,
    functions=["cumprod_cuda"],
    verbose=True
)

def get_inputs():
    batch_size = 32768
    input_shape = (32768,)
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [1]