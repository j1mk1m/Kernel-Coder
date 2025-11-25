import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cumprod_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void cumprod_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, int batch_size, int dim_size, int outer_dim, int inner_dim) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    // Compute the offset for the current batch element
    int outer_offset = batch_idx * outer_dim;
    for (int i = 0; i < inner_dim; ++i) {
        int idx = outer_offset + i * dim_size;
        scalar_t val = input[idx];
        if (i == 0) {
            output[idx] = val;
        } else {
            output[idx] = val * output[idx - 1];
        }
    }
}

torch::Tensor cumprod_cuda(torch::Tensor input, int dim) {
    int64_t batch_size = input.size(0);
    int64_t dim_size = input.size(dim);
    int64_t outer_dim = 1, inner_dim = 1;
    for (int i = 0; i < dim; i++) {
        outer_dim *= input.size(i);
    }
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_dim *= input.size(i);
    }

    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int grid_size = (batch_size + block_size - 1) / block_size;

    // Choose the appropriate kernel based on data type
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cumprod_cuda", ([&] {
        cumprod_kernel<scalar_t><<<grid_size, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim_size,
            outer_dim,
            inner_dim
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

cumprod_cpp_source = """
torch::Tensor cumprod_cuda(torch::Tensor input, int dim);
"""

cumprod_op = load_inline(
    name="cumprod_op",
    cpp_sources=cumprod_cpp_source,
    cuda_sources=cumprod_cuda_source,
    functions=["cumprod_cuda"],
    verbose=True,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cumprod_op = cumprod_op

    def forward(self, x):
        return self.cumprod_op.cumprod_cuda(x, self.dim)

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [dim]