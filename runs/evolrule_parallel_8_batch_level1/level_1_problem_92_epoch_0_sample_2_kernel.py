import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

exclusive_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void exclusive_cumsum_kernel(const scalar_t* input, scalar_t* output, int batch_size, int dim_size) {
    extern __shared__ scalar_t shared_mem[];
    int batch = blockIdx.x;
    int tid = threadIdx.x;

    int elements_per_thread = (dim_size + blockDim.x - 1) / blockDim.x;

    // Load data into shared memory
    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = tid * elements_per_thread + i;
        if (idx < dim_size) {
            shared_mem[tid * elements_per_thread + i] = input[batch * dim_size + idx];
        }
    }
    __syncthreads();

    // Compute prefix sum within each thread's segment
    for (int i = 1; i < elements_per_thread; ++i) {
        int pos = tid * elements_per_thread + i;
        if (pos < dim_size) {
            shared_mem[pos] += shared_mem[pos - 1];
        }
    }
    __syncthreads();

    // Collect partial sums at the end of each segment
    __shared__ scalar_t partial_sums[1024];
    partial_sums[tid] = (tid * elements_per_thread + elements_per_thread - 1 < dim_size) ?
                        shared_mem[tid * elements_per_thread + elements_per_thread - 1] : 0;
    __syncthreads();

    // Compute prefix sum of partial_sums
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = tid * 2 * stride;
        if (index < blockDim.x) {
            partial_sums[index + stride] += partial_sums[index];
        }
        __syncthreads();
    }

    // Backward phase to propagate the partial sums
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = tid * 2 * stride;
        if (index + stride < blockDim.x) {
            partial_sums[index + stride] += partial_sums[index];
        }
        __syncthreads();
    }

    // Compute the starting value for each thread's segment
    scalar_t start_val = tid == 0 ? 0 : partial_sums[tid - 1];

    // Apply the start_val to each element in the thread's segment
    for (int i = 0; i < elements_per_thread; ++i) {
        int pos = tid * elements_per_thread + i;
        if (pos < dim_size) {
            shared_mem[pos] += start_val;
        }
    }
    __syncthreads();

    // Write back to output
    for (int i = 0; i < elements_per_thread; ++i) {
        int pos = tid * elements_per_thread + i;
        if (pos < dim_size) {
            output[batch * dim_size + pos] = shared_mem[pos];
        }
    }
}

at::Tensor exclusive_cumsum_cuda(const at::Tensor& input, int dim) {
    auto output = at::empty_like(input);

    int batch_size = input.size(0);
    int dim_size = input.size(1);

    dim3 block(1024);
    dim3 grid(batch_size);

    int shared_size = block.x * ((dim_size + block.x - 1)/block.x) * sizeof(float);
    shared_size += block.x * sizeof(float);  // For partial_sums

    exclusive_cumsum_kernel<float><<<grid, block, shared_size, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim_size
    );

    return output;
}
"""

exclusive_cumsum_cpp_source = "at::Tensor exclusive_cumsum_cuda(const at::Tensor& input, int dim);"

exclusive_cumsum = load_inline(
    name="exclusive_cumsum",
    cpp_sources=[exclusive_cumsum_cpp_source],
    cuda_sources=[exclusive_cumsum_source],
    functions=["exclusive_cumsum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.cuda_func = exclusive_cumsum

    def forward(self, x):
        return self.cuda_func(x, self.dim)