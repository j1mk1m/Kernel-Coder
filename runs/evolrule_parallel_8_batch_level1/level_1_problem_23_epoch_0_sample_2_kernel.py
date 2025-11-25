import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_cpp_source = """
#include <torch/extension.h>

extern "C" {

torch::Tensor softmax_cuda(torch::Tensor input);

}
"""

softmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void softmax_kernel(T* input, T* output, int dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ T shared[];

    T local_sum = 0.0;

    // First pass: compute exponentials and accumulate sum
    for (int i = tid; i < dim; i += num_threads) {
        T val = input[row * dim + i];
        T exp_val = exp(val);
        local_sum += exp_val;
    }

    // Write to shared memory
    shared[tid] = local_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    T total_sum = shared[0];

    // Second pass: compute output
    for (int i = tid; i < dim; i += num_threads) {
        T val = input[row * dim + i];
        T exp_val = exp(val);
        output[row * dim + i] = exp_val / total_sum;
    }
}

extern "C" {

torch::Tensor softmax_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int dim = input.size(1);

    auto output = at::empty_like(input);

    const int threads_per_block = 256;
    const size_t shared_mem = threads_per_block * sizeof(float);

    dim3 blocks(batch_size);
    dim3 threads(threads_per_block);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softmax_cuda", ([&] {
        softmax_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim
        );
    }));

    return output;
}

}
"""

softmax_ext = load_inline(
    name="softmax_ext",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_cuda_source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_cuda = softmax_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_cuda.softmax_cuda(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []