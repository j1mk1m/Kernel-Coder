import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void masked_cumsum_kernel(
    const scalar_t* x,
    const bool* mask,
    scalar_t* out,
    int batch_size,
    int dim_length
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int offset = batch_idx * dim_length;
    scalar_t sum = 0.0;

    for (int i = 0; i < dim_length; ++i) {
        int idx = offset + i;
        scalar_t current_val = x[idx] * static_cast<scalar_t>(mask[idx]);
        sum += current_val;
        out[idx] = sum;
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim) {
    x = x.contiguous();
    mask = mask.contiguous();
    auto out = torch::zeros_like(x);

    // Ensure dim is the last dimension (as per problem's input shape)
    assert(dim == x.dim()-1 && "Only supports dim as last dimension");
    int batch_size = x.numel() / x.size(dim);
    int dim_length = x.size(dim);

    int threads = 1;
    int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        masked_cumsum_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            mask.data_ptr<bool>(),
            out.data_ptr<scalar_t>(),
            batch_size,
            dim_length
        );
    }));

    cudaDeviceSynchronize();
    return out;
}
"""

masked_cumsum_header = """
torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim);
"""

# Compile the CUDA code
masked_cumsum = load_inline(
    name="masked_cumsum",
    cpp_sources=masked_cumsum_header,
    cuda_sources=masked_cumsum_source,
    functions=["masked_cumsum_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.masked_cumsum = masked_cumsum

    def forward(self, x, mask):
        return self.masked_cumsum.masked_cumsum_cuda(x, mask, self.dim)