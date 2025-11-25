import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <int BLOCK_SIZE>
__global__ void masked_cumsum(
    const float* x,
    const unsigned char* mask,
    float* out,
    int dim_size,
    int batch_size,
    int dim) {

    int block_per_slice = (dim_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int slice_id = blockIdx.x / block_per_slice;
    int block_in_slice = blockIdx.x % block_per_slice;
    int start = block_in_slice * BLOCK_SIZE;
    int element_id = threadIdx.x + start;

    extern __shared__ float sdata[];

    if (element_id < dim_size) {
        int idx = slice_id * dim_size + element_id;
        sdata[threadIdx.x] = x[idx] * (mask[idx] ? 1.0f : 0.0f);
    } else {
        sdata[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Inclusive scan within the block's segment
    for (int s = 1; s <= blockDim.x; s *= 2) {
        int index = threadIdx.x - s;
        if (index >= 0) {
            sdata[threadIdx.x] += sdata[index];
        }
        __syncthreads();
    }

    if (element_id < dim_size) {
        out[slice_id * dim_size + element_id] = sdata[threadIdx.x];
    }
}

at::Tensor masked_cumsum_cuda(
    const at::Tensor& x,
    const at::Tensor& mask,
    int dim) {

    int dim_size = x.size(dim);
    int batch_size = 1;
    for (int i = 0; i < x.dim(); ++i) {
        if (i != dim) {
            batch_size *= x.size(i);
        }
    }

    const int block_size = 1024;
    int block_per_slice = (dim_size + block_size - 1) / block_size;
    int grid_size = batch_size * block_per_slice;

    auto out = at::empty_like(x);

    masked_cumsum<block_size><<<grid_size, block_size, block_size * sizeof(float)>>>(
        x.data_ptr<float>(),
        mask.data_ptr<unsigned char>(),
        out.data_ptr<float>(),
        dim_size,
        batch_size,
        dim);

    return out;
}
"""

masked_cumsum_cpp_source = "at::Tensor masked_cumsum_cuda(const at::Tensor& x, const at::Tensor& mask, int dim);"

masked_cumsum = load_inline(
    name="masked_cumsum",
    cpp_sources=masked_cumsum_cpp_source,
    cuda_sources=masked_cumsum_source,
    functions=["masked_cumsum_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, mask):
        return masked_cumsum.masked_cumsum_cuda(x, mask, self.dim)

def get_inputs():
    x = torch.rand(batch_size, *input_shape).cuda()
    mask = torch.randint(0, 2, x.shape).bool().cuda()
    return [x, mask]

def get_init_inputs():
    return [dim]