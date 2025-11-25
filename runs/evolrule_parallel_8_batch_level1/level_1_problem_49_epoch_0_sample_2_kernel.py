import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void max_reduction_kernel(
    const scalar_t* input, scalar_t* output,
    int dim, int B, int D1, int D2
) {
    int output_idx = blockIdx.x;
    int tid = threadIdx.x;

    int reduction_size = 0;
    int base_offset = 0;
    int step = 1;

    if (dim == 0) {
        int d1 = output_idx / D2;
        int d2 = output_idx % D2;
        base_offset = d1 * D2 + d2;
        step = D1 * D2;
        reduction_size = B;
    } else if (dim == 1) {
        int b = output_idx / D2;
        int d2 = output_idx % D2;
        base_offset = b * (D1 * D2) + d2;
        step = D2;
        reduction_size = D1;
    } else { // dim == 2
        int b = output_idx / D1;
        int d1 = output_idx % D1;
        base_offset = b * (D1 * D2) + d1 * D2;
        step = 1;
        reduction_size = D2;
    }

    __shared__ scalar_t shared_max[256];
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();

    int chunk_size = (reduction_size + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < chunk_size; ++i) {
        int d = tid * chunk_size + i;
        if (d < reduction_size) {
            scalar_t val = input[base_offset + d * step];
            if (val > local_max) {
                local_max = val;
            }
        }
    }

    shared_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid] < shared_max[tid + s]) {
                shared_max[tid] = shared_max[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[output_idx] = shared_max[0];
    }
}

torch::Tensor max_reduction_cuda(torch::Tensor input, int dim) {
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);

    int out_size;
    torch::IntArrayRef output_shape;
    if (dim == 0) {
        out_size = D1 * D2;
        output_shape = {D1, D2};
    } else if (dim == 1) {
        out_size = B * D2;
        output_shape = {B, D2};
    } else {
        out_size = B * D1;
        output_shape = {B, D1};
    }

    auto output = torch::empty(output_shape, input.options());

    const int block_size = 256;
    const int num_blocks = out_size;

    dim3 threads(block_size);
    dim3 blocks(num_blocks);

    max_reduction_kernel<float><<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dim,
        B, D1, D2
    );

    return output;
}
"""

max_reduction_cpp_source = """
torch::Tensor max_reduction_cuda(torch::Tensor input, int dim);
"""

max_reduction = load_inline(
    name="max_reduction",
    cpp_sources=max_reduction_cpp_source,
    cuda_sources=max_reduction_source,
    functions=["max_reduction_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.max_reduction = max_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_reduction.max_reduction_cuda(x, self.dim)