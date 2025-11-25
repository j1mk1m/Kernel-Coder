import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void masked_cumsum_kernel(
    const scalar_t* x,
    const unsigned char* mask,
    scalar_t* out,
    int batch_size,
    int dim_size,
    int dim) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size)
        return;

    if (dim != 1) {
        return;
    }

    scalar_t sum = 0.0;
    for (int i = 0; i < dim_size; ++i) {
        int idx = row * dim_size + i;
        scalar_t val = x[idx] * static_cast<scalar_t>(mask[idx]);
        sum += val;
        out[idx] = sum;
    }
}

torch::Tensor masked_cumsum_cuda(
    torch::Tensor x,
    torch::Tensor mask,
    int dim) {

    const int batch_size = x.size(0);
    const int dim_size = x.size(1);

    auto out = torch::empty_like(x);

    const int threads_per_block = 1;
    const int blocks_per_grid = batch_size;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        masked_cumsum_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            x.data_ptr<scalar_t>(),
            mask.data_ptr<unsigned char>(),
            out.data_ptr<scalar_t>(),
            batch_size,
            dim_size,
            dim);
    }));

    cudaDeviceSynchronize();

    return out;
}
"""

masked_cumsum_cpp_source = """
torch::Tensor masked_cumsum_cuda(
    torch::Tensor x,
    torch::Tensor mask,
    int dim);
"""

masked_cumsum = load_inline(
    name="masked_cumsum",
    cpp_sources=masked_cumsum_cpp_source,
    cuda_sources=masked_cumsum_source,
    functions=["masked_cumsum_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.masked_cumsum_cuda = masked_cumsum

    def forward(self, x, mask):
        x = x.cuda()
        mask = mask.cuda()
        return self.masked_cumsum_cuda.masked_cumsum_cuda(x, mask, self.dim)