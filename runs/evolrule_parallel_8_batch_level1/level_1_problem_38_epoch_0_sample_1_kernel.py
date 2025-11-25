import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void l1_norm_kernel(
    const scalar_t* input,
    scalar_t* output,
    int batch_size,
    int dim,
    scalar_t epsilon
) {
    int batch_idx = blockIdx.x;
    int row_start = batch_idx * dim;

    extern __shared__ scalar_t shared_sums[];
    scalar_t partial_sum = 0.0;

    // First pass: compute partial sums of absolute values
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        scalar_t val = input[row_start + i];
        partial_sum += fabs(val);
    }

    // Write partial sum to shared memory
    shared_sums[threadIdx.x] = partial_sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sums[threadIdx.x] += shared_sums[threadIdx.x + s];
        }
        __syncthreads();
    }

    scalar_t row_sum = shared_sums[0] + epsilon;
    __syncthreads();

    // Second pass: compute output
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        scalar_t val = input[row_start + i];
        output[row_start + i] = val / row_sum;
    }
}

at::Tensor l1_norm_cuda(at::Tensor input, float epsilon) {
    const int batch_size = input.size(0);
    const int dim = input.size(1);

    auto output = at::empty_like(input);

    const int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(block_size);

    // Allocate shared memory for each block: blockDim.x floats
    int shared_mem_size = block_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "l1_norm_cuda", ([&] {
        l1_norm_kernel<scalar_t><<<grid, block, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            dim,
            epsilon
        );
    }));

    return output;
}
"""

elementwise_l1_norm_cpp_source = """
#include <torch/extension.h>

torch::Tensor l1_norm_cuda(torch::Tensor input, float epsilon);
"""

# Compile the inline CUDA code for L1 normalization
l1_norm_cuda = load_inline(
    name="l1_norm_cuda",
    cpp_sources=elementwise_l1_norm_cpp_source,
    cuda_sources=elementwise_l1_norm_source,
    functions=["l1_norm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.epsilon = 1e-8  # As specified in problem statement

    def forward(self, x):
        return l1_norm_cuda.l1_norm_cuda(x, self.epsilon)

def get_inputs():
    batch_size = 32768
    dim = 65535
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []