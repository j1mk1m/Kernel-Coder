import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

logsoftmax_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void LogSoftmaxForwardKernel(const scalar_t* __restrict__ input, scalar_t* output,
                                       int batch_size, int dim_size, int dim_stride) {
    // Each block handles one sample in the batch
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ scalar_t shared_max;
    __shared__ scalar_t shared_sum;

    // Load input into shared memory
    extern __shared__ scalar_t shared_data[];
    scalar_t* sdata = shared_data;

    if (tid < dim_size) {
        sdata[tid] = input[batch_idx * dim_stride + tid];
    } else {
        sdata[tid] = -FLT_MAX;  // Set to -infinity for unused elements
    }
    __syncthreads();

    // Step 1: Compute max using parallel reduction
    for (int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x && index + s < blockDim.x) {
            if (sdata[index] < sdata[index + s]) {
                sdata[index] = sdata[index + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        shared_max = sdata[0];
    }
    __syncthreads();

    // Step 2: Compute exp(x_i - max) and accumulate sum
    scalar_t exp_val = 0;
    if (tid < dim_size) {
        exp_val = expf(sdata[tid] - shared_max);
    } else {
        exp_val = 0;
    }
    __shared__ scalar_t shared_sum_partial[blockDim.x];
    shared_sum_partial[tid] = exp_val;
    __syncthreads();

    // Parallel sum reduction for exp_val
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum_partial[tid] += shared_sum_partial[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        shared_sum = shared_sum_partial[0];
    }
    __syncthreads();

    // Compute log_sum
    scalar_t log_sum = logf(shared_sum);

    // Step 3: Compute the final log_softmax result
    if (tid < dim_size) {
        output[batch_idx * dim_stride + tid] = 
            (input[batch_idx * dim_stride + tid] - shared_max) - log_sum;
    }
}

at::Tensor log_softmax_cuda(at::Tensor input, int dim) {
    const int batch_size = input.size(0);
    const int dim_size = input.size(1);
    const int dim_stride = input.stride(0);

    at::Tensor output = at::empty_like(input);

    dim3 blocks(batch_size);
    dim3 threads(std::min(dim_size, 1024));

    // Calculate shared memory size: dim_size (for data) + blockDim.x (for sum_partial)
    int shared_mem_bytes = (dim_size + blockDim.x) * sizeof(float);
    shared_mem_bytes = (shared_mem_bytes + 255) & ~255; // Align to 256 bytes

    LogSoftmaxForwardKernel<float><<<blocks, threads, shared_mem_bytes>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, dim_size, dim_stride
    );

    return output;
}
"""

logsoftmax_cpp_source = "at::Tensor log_softmax_cuda(at::Tensor input, int dim);"

# Compile the CUDA kernel
logsoftmax = load_inline(
    name="logsoftmax",
    cpp_sources=logsoftmax_cpp_source,
    cuda_sources=logsoftmax_source,
    functions=["log_softmax_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.log_softmax_cuda = logsoftmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.cuda()  # Ensure input is on CUDA
        return self.log_softmax_cuda.log_softmax_cuda(x, self.dim)