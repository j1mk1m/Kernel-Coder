import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

reverse_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void reverse_cumsum_kernel(const scalar_t* input, scalar_t* output, int batch_size, int dim_size) {
    int batch_idx = blockIdx.x;
    int input_offset = batch_idx * dim_size;
    int output_offset = batch_idx * dim_size;

    scalar_t current_sum = 0;
    for (int i = dim_size - 1; i >= 0; --i) {
        current_sum += input[input_offset + i];
        output[output_offset + i] = current_sum;
    }
}

at::Tensor reverse_cumsum_cuda(at::Tensor input, int64_t dim) {
    auto output = at::empty_like(input);
    int batch_size = input.size(0);
    int dim_size = input.size(dim); // Handle any dimension

    const int threads_per_block = 1;
    const dim3 blocks(batch_size);
    const dim3 threads(1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "reverse_cumsum_cuda", ([&] {
        reverse_cumsum_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            dim_size
        );
    }));

    return output;
}
"""

reverse_cumsum_cpp_source = """
#include <torch/extension.h>
at::Tensor reverse_cumsum_cuda(at::Tensor input, int64_t dim);
"""

reverse_cumsum = load_inline(
    name="reverse_cumsum",
    cpp_sources=reverse_cumsum_cpp_source,
    cuda_sources=reverse_cumsum_source,
    functions=["reverse_cumsum_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reverse_cumsum = reverse_cumsum

    def forward(self, x):
        return self.reverse_cumsum.reverse_cumsum_cuda(x, self.dim)