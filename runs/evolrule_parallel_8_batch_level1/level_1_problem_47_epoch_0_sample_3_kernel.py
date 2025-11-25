import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

custom_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void custom_sum_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    int N, int D1, int D2,
    int dim,
    int output_size) {

    extern __shared__ scalar_t partial_sums[];

    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    int output_n, d1, d2;

    if (dim == 0) {
        d1 = block_idx / D2;
        d2 = block_idx % D2;
        output_n = 0;
    } else if (dim == 1) {
        output_n = block_idx / D2;
        d2 = block_idx % D2;
    } else if (dim == 2) {
        output_n = block_idx / D1;
        d1 = block_idx % D1;
    } else {
        assert(0 && "Invalid dimension");
    }

    int reduction_size = (dim == 0 ? N : (dim == 1 ? D1 : D2));
    int chunk_size = (reduction_size + blockDim.x - 1) / blockDim.x;
    int start = tid * chunk_size;
    int end = start + chunk_size;
    if (end > reduction_size) end = reduction_size;

    scalar_t sum = 0.0;

    if (dim == 0) {
        for (int k = start; k < end; ++k) {
            int input_idx = k * D1 * D2 + d1 * D2 + d2;
            sum += x[input_idx];
        }
    } else if (dim == 1) {
        for (int k = start; k < end; ++k) {
            int input_idx = output_n * D1 * D2 + k * D2 + d2;
            sum += x[input_idx];
        }
    } else if (dim == 2) {
        for (int k = start; k < end; ++k) {
            int input_idx = output_n * D1 * D2 + d1 * D2 + k;
            sum += x[input_idx];
        }
    }

    partial_sums[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[block_idx] = partial_sums[0];
    }
}

torch::Tensor custom_sum_cuda(torch::Tensor x, int dim) {
    int64_t N = x.size(0);
    int64_t D1 = x.size(1);
    int64_t D2 = x.size(2);
    int64_t output_size = x.numel() / x.size(dim);

    auto output_shape = x.sizes().vec();
    output_shape[dim] = 1;
    auto out = torch::zeros(output_shape, x.options());

    const int block_size = 256;
    int shared_mem_size = block_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "custom_sum_cuda", ([&] {
        custom_sum_kernel<scalar_t><<<output_size, block_size, shared_mem_size>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            N, D1, D2,
            dim,
            output_size);
    }));

    return out;
}
"""

custom_sum_cpp_source = """
torch::Tensor custom_sum_cuda(torch::Tensor x, int dim);
"""

custom_sum = load_inline(
    name="custom_sum",
    cpp_sources=custom_sum_cpp_source,
    cuda_sources=custom_sum_source,
    functions=["custom_sum_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.custom_sum = custom_sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_sum.custom_sum_cuda(x, self.dim)