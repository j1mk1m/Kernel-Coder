import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

logsoftmax_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

template<typename T>
__global__ void log_softmax_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int batch_size,
    int dim_size) {

    extern __shared__ T sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    const T* row_input = input + bid * dim_size;
    T* row_output = output + bid * dim_size;

    // Step 1: Compute row max
    T local_max = -INFINITY;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        T val = row_input[i];
        if (val > local_max) local_max = val;
    }

    // Block reduction for max
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>=1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    T row_max = sdata[0];
    __syncthreads();

    // Step 2: Compute exp and accumulate sum
    T local_exp_sum = 0;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        T exp_val = exp(row_input[i] - row_max);
        local_exp_sum += exp_val;
    }

    // Block reduction for sum
    sdata[tid] = local_exp_sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    T exp_sum = sdata[0];
    __syncthreads();

    // Step 3: Compute final value
    T log_sum = log(exp_sum);
    for (int i = tid; i < dim_size; i += blockDim.x) {
        row_output[i] = (row_input[i] - row_max) - log_sum;
    }
}

#define BLOCK_SIZE 256

at::Tensor log_softmax_forward_cuda(
    at::Tensor input,
    int dim) {

    const auto batch_size = input.size(0);
    const auto dim_size = input.size(1);
    auto output = at::empty_like(input);

    dim3 blocks(batch_size);
    dim3 threads(BLOCK_SIZE);
    size_t shared_size = sizeof(float) * (BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "log_softmax_forward_cuda", ([&] {
        log_softmax_forward_kernel<scalar_t><<<blocks, threads, shared_size>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            dim_size);
    }));

    return output;
}
"""

logsoftmax_cpp = """
at::Tensor log_softmax_forward_cuda(
    at::Tensor input,
    int dim);
"""

model_new = load_inline(
    name="logsoftmax_ext",
    cpp_sources=logsoftmax_cpp,
    cuda_sources=logsoftmax_source,
    functions=["log_softmax_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
        self.forward_func = model_new.log_softmax_forward_cuda

    def forward(self, x):
        return self.forward_func(x, self.dim)