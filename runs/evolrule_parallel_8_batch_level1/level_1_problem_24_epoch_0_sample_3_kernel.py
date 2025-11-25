import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

log_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void log_softmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* output,
    int batch_size,
    int dim) {

    int row = blockIdx.x;
    int tid = threadIdx.x;

    scalar_t local_max = -FLT_MAX;
    for (int i = row * dim + tid; i < (row + 1)*dim; i += blockDim.x) {
        scalar_t val = input[i];
        if (val > local_max) {
            local_max = val;
        }
    }

    __shared__ scalar_t sdata[threadDim.x];
    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid] < sdata[tid + s]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }
    scalar_t global_max = sdata[0];
    __syncthreads();

    scalar_t local_sum = 0.0;
    for (int i = row * dim + tid; i < (row + 1)*dim; i += blockDim.x) {
        scalar_t y_i = input[i] - global_max;
        local_sum += exp(y_i);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    scalar_t total_sum = sdata[0];
    __syncthreads();

    scalar_t log_denominator = log(total_sum);

    for (int i = row * dim + tid; i < (row + 1)*dim; i += blockDim.x) {
        scalar_t y_i = input[i] - global_max;
        output[i] = y_i - log_denominator;
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int dim = input.size(1);

    auto output = torch::empty_like(input);

    const int threads_per_block = 1024;
    dim3 blocks(batch_size);
    dim3 threads(threads_per_block);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "log_softmax_cuda", ([&] {
        log_softmax_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim);
    }));

    return output;
}
"""

log_softmax_cpp_source = """
#include <torch/extension.h>

torch::Tensor log_softmax_cuda(torch::Tensor input);
"""

log_softmax = load_inline(
    name="log_softmax",
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_source,
    functions=["log_softmax_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return log_softmax.log_softmax_cuda(x)