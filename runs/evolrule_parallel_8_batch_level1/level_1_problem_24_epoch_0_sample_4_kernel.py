import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for LogSoftmax
logsoftmax_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void logsoftmax_forward_kernel(const scalar_t* __restrict__ input,
                                         scalar_t* __restrict__ output,
                                         int batch_size,
                                         int dim_size,
                                         int dim) {
    int batch_idx = blockIdx.x;
    int element_idx = threadIdx.x;

    __shared__ scalar_t thread_data[256]; // Shared memory for max and sum

    // Find the maximum value in the current batch's dimension
    scalar_t max_val = -FLT_MAX;
    if (element_idx < dim_size) {
        scalar_t val = input[batch_idx * dim_size + element_idx];
        if (val > max_val) {
            max_val = val;
        }
    }
    __syncthreads();

    // Reduce to find the global max using shared memory
    int idx = element_idx;
    while (idx < dim_size) {
        if (thread_data[idx] > max_val) {
            max_val = thread_data[idx];
        }
        idx += blockDim.x;
    }
    __syncthreads();

    // Compute exp(x_i - max)
    scalar_t sum = 0.0;
    if (element_idx < dim_size) {
        scalar_t exp_val = exp(input[batch_idx * dim_size + element_idx] - max_val);
        sum += exp_val;
        thread_data[element_idx] = exp_val;
    }
    __syncthreads();

    // Sum all exponentials in the dimension using parallel reduction
    for (int stride = 1; stride < dim_size; stride *= 2) {
        if (element_idx < stride) {
            thread_data[element_idx] += thread_data[element_idx + stride];
        }
        __syncthreads();
    }
    sum = thread_data[0];

    // Compute log(sum) and the final result
    if (element_idx < dim_size) {
        scalar_t log_sum = log(sum);
        output[batch_idx * dim_size + element_idx] =
            input[batch_idx * dim_size + element_idx] - max_val - log_sum;
    }
}

torch::Tensor logsoftmax_forward_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int dim_size = input.size(1);
    auto output = torch::empty_like(input);

    dim3 blocks(batch_size);
    dim3 threads(dim_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "logsoftmax_forward_cuda", ([&] {
        logsoftmax_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            dim_size,
            1);
    }));

    return output;
}
"""

logsoftmax_cpp_source = """
torch::Tensor logsoftmax_forward_cuda(torch::Tensor input);
"""

# Compile the CUDA kernel
logsoftmax = load_inline(
    name="logsoftmax",
    cpp_sources=logsoftmax_cpp_source,
    cuda_sources=logsoftmax_source,
    functions=["logsoftmax_forward_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=["-lcudart", "-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.logsoftmax = logsoftmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logsoftmax.logsoftmax_forward_cuda(x)

# Ensure inputs are on the correct device and type
def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]

def get_init_inputs():
    return []