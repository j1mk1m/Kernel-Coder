import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const unsigned char* __restrict__ mask,
    scalar_t* __restrict__ out,
    int batch_size,
    int dim_size,
    int dim) {

    int row = blockIdx.x;
    if (row >= batch_size) return;

    scalar_t current_sum = 0.0;
    for (int pos = 0; pos < dim_size; ++pos) {
        int idx = row * dim_size + pos;
        if (mask[idx]) {
            current_sum += x[idx];
        }
        out[idx] = current_sum;
    }
}

// Launcher function
at::Tensor masked_cumsum_cuda(at::Tensor x, at::Tensor mask, int dim) {
    AT_ASSERTM(x.device().is_cuda(), "x must be a CUDA tensor");
    AT_ASSERTM(mask.device().is_cuda(), "mask must be a CUDA tensor");
    AT_ASSERTM(x.is_contiguous(), "x must be contiguous");
    AT_ASSERTM(mask.is_contiguous(), "mask must be contiguous");
    AT_ASSERTM(x.sizes() == mask.sizes(), "x and mask must have the same shape");

    auto batch_size = x.size(0);
    auto dim_size = x.size(1);

    auto out = at::empty_like(x);

    dim3 threads(1);
    dim3 blocks(batch_size);

    masked_cumsum_kernel<float><<<blocks, threads>>>(
        x.data_ptr<float>(),
        mask.data_ptr<uint8_t>(),
        out.data_ptr<float>(),
        batch_size,
        dim_size,
        dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }

    return out;
}
"""

masked_cumsum_cpp_source = """
at::Tensor masked_cumsum_cuda(at::Tensor x, at::Tensor mask, int dim);
"""

# Compile the CUDA code
masked_cumsum = load_inline(
    name="masked_cumsum",
    cpp_sources=masked_cumsum_cpp_source,
    cuda_sources=masked_cumsum_source,
    functions=["masked_cumsum_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_flags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.masked_cumsum = masked_cumsum

    def forward(self, x, mask):
        return self.masked_cumsum.masked_cumsum_cuda(x, mask, self.dim)