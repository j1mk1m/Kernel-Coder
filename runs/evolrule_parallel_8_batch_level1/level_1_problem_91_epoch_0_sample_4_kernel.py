import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

reverse_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void reverse_cumsum_kernel(
    scalar_t* input,
    scalar_t* output,
    int batch_size,
    int dim_size) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    scalar_t* bufferA = input + batch_idx * dim_size;
    scalar_t* bufferB = output + batch_idx * dim_size;

    if (tid < dim_size) {
        bufferB[tid] = bufferA[tid];
    }
    __syncthreads();

    for (int step = 0; step < 15; step++) {
        int stride = 1 << step;
        scalar_t* temp = bufferA;
        bufferA = bufferB;
        bufferB = temp;

        for (int i = tid; i < dim_size; i += blockDim.x) {
            if (i >= stride) {
                bufferB[i - stride] = bufferA[i - stride] + bufferA[i];
            } else {
                bufferB[i] = bufferA[i];
            }
        }
        __syncthreads();
    }

    if (tid < dim_size) {
        output[batch_idx * dim_size + tid] = bufferA[tid];
    }
}

at::Tensor reverse_cumsum_cuda(at::Tensor input) {
    const int batch_size = input.size(0);
    const int dim_size = input.size(1);

    auto output = at::empty_like(input);
    const int block_size = 1024;
    const dim3 grid(batch_size);
    const dim3 block(block_size);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "reverse_cumsum_cuda", ([&] {
        reverse_cumsum_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim_size);
    }));

    return output;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("reverse_cumsum", &reverse_cumsum_cuda, "Reverse cumulative sum");
}
"""

reverse_cumsum_cpp_source = """
#include <torch/extension.h>
"""

reverse_cumsum = load_inline(
    name="reverse_cumsum",
    cpp_sources=reverse_cumsum_cpp_source,
    cuda_sources=reverse_cumsum_source,
    functions=["reverse_cumsum_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reverse_cumsum = reverse_cumsum

    def forward(self, x):
        return self.reverse_cumsum.reverse_cumsum_cuda(x)

# The original get_inputs and get_init_inputs remain unchanged
def get_inputs():
    batch_size = 32768
    input_shape = (32768,)
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [1]