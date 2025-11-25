import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for LogSoftmax
log_softmax_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

template <typename scalar_t>
__global__ void log_softmax_forward_kernel(const scalar_t* __restrict__ input,
                                          scalar_t* __restrict__ output,
                                          const int batch_size,
                                          const int dim_size) {
    // Each thread handles one element
    const int batch_idx = blockIdx.x;
    const int dim_idx = threadIdx.x;

    // Load the current element
    scalar_t value = input[batch_idx * dim_size + dim_idx];

    // Find the maximum value in the row for numerical stability
    __shared__ scalar_t shared_max;
    if (dim_idx == 0) {
        shared_max = std::numeric_limits<scalar_t>::lowest();
        for (int j = 0; j < dim_size; ++j) {
            scalar_t current_val = input[batch_idx * dim_size + j];
            if (current_val > shared_max) {
                shared_max = current_val;
            }
        }
    }
    __syncthreads();

    // Compute exp(x_i - C)
    scalar_t exp_val = exp(value - shared_max);

    // Compute the sum of exp(x_j - C) using parallel reduction
    __shared__ scalar_t shared_sum;
    if (dim_idx == 0) {
        shared_sum = 0.0;
    }
    __syncthreads();

    // Each thread contributes its exp_val to the sum
    atomicAdd(&shared_sum, exp_val);
    __syncthreads();

    // Compute log_sum = log(sum) + C
    scalar_t log_sum = log(shared_sum) + shared_max;

    // Write the result
    output[batch_idx * dim_size + dim_idx] = value - log_sum;
}

at::Tensor log_softmax_forward_cuda(const at::Tensor& input) {
    const auto batch_size = input.size(0);
    const auto dim_size = input.size(1);

    auto output = at::empty_like(input);

    dim3 grid(batch_size);
    dim3 block(dim_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "log_softmax_forward_cuda", ([&] {
        log_softmax_forward_kernel<scalar_t><<<grid, block>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            dim_size
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

log_softmax_cuda_header = """
at::Tensor log_softmax_forward_cuda(const at::Tensor& input);
"""

# Compile the CUDA kernel
log_softmax_cuda = load_inline(
    name="log_softmax_cuda",
    cpp_sources=log_softmax_cuda_header,
    cuda_sources=log_softmax_cuda_source,
    functions=["log_softmax_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.log_softmax = log_softmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure the input is on the correct device
        x = x.cuda()
        return self.log_softmax.log_softmax_forward_cuda(x)