import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for LogSoftmax
logsoftmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

template <typename scalar_t>
__device__ scalar_t log_sum_exp(scalar_t* data, int dim_size) {
    extern __shared__ scalar_t shared_data[];
    int tid = threadIdx.x;
    shared_data[tid] = data[tid];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + s]);
        }
        __syncthreads();
    }

    scalar_t max_val = shared_data[0];
    __syncthreads();

    shared_data[tid] = exp(data[tid] - max_val);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    scalar_t sum_exp = shared_data[0];
    return max_val + log(sum_exp);
}

template <typename scalar_t>
__global__ void logsoftmax_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* output,
    int batch_size,
    int dim_size
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    scalar_t* data = input + batch_idx * dim_size;
    scalar_t* out_data = output + batch_idx * dim_size;

    __shared__ scalar_t shared_data[1024]; // Adjust based on block size

    if (tid < dim_size) {
        shared_data[tid] = data[tid];
    }
    __syncthreads();

    scalar_t max_val = log_sum_exp<scalar_t>(shared_data, dim_size);
    __syncthreads();

    if (tid < dim_size) {
        out_data[tid] = data[tid] - max_val - log(exp(data[tid] - max_val));
    }
}

torch::Tensor logsoftmax_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int dim_size = input.size(1);

    auto output = torch::empty_like(input);

    const int block_size = 256;
    const dim3 blocks(batch_size);
    const dim3 threads(block_size);

    logsoftmax_forward_kernel<float><<<blocks, threads, block_size * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim_size
    );

    return output;
}
"""

logsoftmax_cpp_source = "torch::Tensor logsoftmax_cuda(torch::Tensor input);"

# Compile the inline CUDA code
logsoftmax = load_inline(
    name="logsoftmax",
    cpp_sources=logsoftmax_cpp_source,
    cuda_sources=logsoftmax_source,
    functions=["logsoftmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.logsoftmax_cuda = logsoftmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logsoftmax_cuda.logsoftmax_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []