import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for cumulative product
cumprod_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void cumprod_kernel(const scalar_t* input, scalar_t* output, int dim_size, int outer_dim, int inner_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_dim * inner_dim) return;

    int outer = idx / inner_dim;
    int inner = idx % inner_dim;

    scalar_t product = 1;
    for (int d = 0; d < dim_size; ++d) {
        int pos = outer * dim_size * inner_dim + d * inner_dim + inner;
        product *= input[pos];
        output[pos] = product;
    }
}

torch::Tensor cumprod_cuda(torch::Tensor input, int dim) {
    const int64_t* dims = input.sizes().data();
    int dim_size = input.size(dim);
    int outer_dim = 1;
    int inner_dim = 1;

    for (int i = 0; i < dim; ++i) {
        outer_dim *= dims[i];
    }
    for (int i = dim + 1; i < input.dim(); ++i) {
        inner_dim *= dims[i];
    }

    int total_threads = outer_dim * inner_dim;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    auto output = torch::empty_like(input);
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cumprod_cuda", ([&] {
        cumprod_kernel<scalar_t><<<grid_size, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
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

cumprod_module = load_inline(
    name="cumprod",
    cpp_sources=cumprod_cpp_source,
    cuda_sources=cumprod_kernel_source,
    functions=["cumprod_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cumprod_cuda = cumprod_module

    def forward(self, x):
        return self.cumprod_cuda.cumprod_cuda(x, self.dim)

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]