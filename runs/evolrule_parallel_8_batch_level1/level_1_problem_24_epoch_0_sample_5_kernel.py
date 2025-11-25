import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

logsoftmax_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void logsoftmax_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int dim) {

    const int batch = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ scalar_t shared_mem[];

    // Phase 1: Compute max for the batch
    scalar_t local_max = -INFINITY;
    for (int i = tid; i < dim; i += blockDim.x) {
        scalar_t val = input[batch * dim + i];
        if (val > local_max) local_max = val;
    }

    // Write to shared memory and reduce max
    scalar_t* partial_max = shared_mem;
    partial_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial_max[tid] = max(partial_max[tid], partial_max[tid + s]);
        __syncthreads();
    }
    scalar_t global_max = partial_max[0];
    __syncthreads();

    // Phase 2: Compute sum of exp(x_i - global_max)
    scalar_t local_sum = 0.0;
    for (int i = tid; i < dim; i += blockDim.x) {
        scalar_t val = input[batch * dim + i] - global_max;
        local_sum += exp(val);
    }

    // Write to shared memory and reduce sum
    scalar_t* partial_sum = shared_mem;
    partial_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial_sum[tid] += partial_sum[tid + s];
        __syncthreads();
    }
    scalar_t total_sum = partial_sum[0];
    __syncthreads();

    // Phase 3: Compute final output
    scalar_t log_sum = log(total_sum);
    for (int i = tid; i < dim; i += blockDim.x) {
        scalar_t val = input[batch * dim + i];
        output[batch * dim + i] = (val - global_max) - log_sum;
    }
}

at::Tensor logsoftmax_forward_cuda(const at::Tensor& input) {
    const int batch_size = input.size(0);
    const int dim = input.size(1);

    auto output = at::empty_like(input);

    const int block_size = 1024;
    const int shared_mem_size = 2 * block_size * sizeof(float);

    dim3 blocks(batch_size);
    dim3 threads(block_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "logsoftmax_forward_cuda", ([&] {
        logsoftmax_forward_kernel<scalar_t><<<
            blocks, threads, shared_mem_size, at::cuda::getCurrentCUDAStream()
        >>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim
        );
    }));

    return output;
}
"""

logsoftmax_header = """
at::Tensor logsoftmax_forward_cuda(const at::Tensor& input);
"""

logsoftmax_cuda = load_inline(
    name='logsoftmax_cuda',
    cpp_sources=logsoftmax_header,
    cuda_sources=logsoftmax_source,
    functions=['logsoftmax_forward_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.logsoftmax_cuda = logsoftmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logsoftmax_cuda.logsoftmax_forward_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []