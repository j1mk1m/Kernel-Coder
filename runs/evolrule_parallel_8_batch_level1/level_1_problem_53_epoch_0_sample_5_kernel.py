import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel code
min_reduction_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void min_reduction_kernel(
    const scalar_t* input,
    scalar_t* output,
    int B, int D1, int D2,
    int reduction_dim
) {
    int output_idx = blockIdx.x;

    int coord0, coord1;
    int S;
    if (reduction_dim == 0) {
        coord0 = output_idx / D2;
        coord1 = output_idx % D2;
        S = B;
    } else if (reduction_dim == 1) {
        coord0 = output_idx / D2;
        coord1 = output_idx % D2;
        S = D1;
    } else {
        coord0 = output_idx / D1;
        coord1 = output_idx % D1;
        S = D2;
    }

    extern __shared__ scalar_t shared[];
    int tid = threadIdx.x;
    scalar_t min_val = std::numeric_limits<scalar_t>::max();

    int chunk_size = (S + blockDim.x - 1) / blockDim.x;
    int start = tid * chunk_size;
    int end = min(start + chunk_size, S);

    for (int d_rdim = start; d_rdim < end; ++d_rdim) {
        int input_idx;
        if (reduction_dim == 0) {
            input_idx = d_rdim * D1 * D2 + coord0 * D2 + coord1;
        } else if (reduction_dim == 1) {
            input_idx = coord0 * D1 * D2 + d_rdim * D2 + coord1;
        } else {
            input_idx = coord0 * D1 * D2 + coord1 * D2 + d_rdim;
        }
        scalar_t val = input[input_idx];
        if (val < min_val) {
            min_val = val;
        }
    }

    shared[tid] = min_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared[tid + s] < shared[tid]) {
                shared[tid] = shared[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[output_idx] = shared[0];
    }
}

// Define the wrapper function
torch::Tensor min_reduction_cuda(torch::Tensor input, int dim) {
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);

    auto output_size = input.sizes().vec();
    output_size.erase(output_size.begin() + dim);
    auto output = torch::empty(output_size, input.options());

    int total_out = output.numel();

    int block_size = 256;
    dim3 blocks((total_out + block_size - 1) / block_size);
    dim3 threads(block_size);
    size_t shared_mem_size = block_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "min_reduction_cuda", ([&] {
        min_reduction_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B, D1, D2,
            dim
        );
    }));

    return output;
}
"""

min_reduction_cpp_source = """
torch::Tensor min_reduction_cuda(torch::Tensor input, int dim);
"""

# Compile the CUDA extension
min_reduction = load_inline(
    name="min_reduction",
    cpp_sources=min_reduction_cpp_source,
    cuda_sources=min_reduction_kernel_source,
    functions=["min_reduction_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle device compatibility with original code's get_inputs()
        original_device = x.device
        x_cuda = x.cuda()
        output_cuda = min_reduction.min_reduction_cuda(x_cuda, self.dim)
        return output_cuda.to(original_device)