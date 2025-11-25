import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void l2_norm_kernel(const float* x, float* out, int batch_size, int dim) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    __shared__ float sdata[256];

    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = x[row * dim + i];
        sum += val * val;
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float norm = sqrtf(sdata[0]);
        sdata[0] = norm;
    }
    __syncthreads();

    float norm = sdata[0];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out[row * dim + i] = x[row * dim + i] / norm;
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int dim = x.size(1);

    auto out = torch::empty_like(x);

    const int block_size = 256;
    const dim3 grid(batch_size);
    const dim3 block(block_size);

    l2_norm_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim);

    return out;
}
"""

l2_norm_cpp_source = (
    "torch::Tensor l2_norm_cuda(torch::Tensor x);"
)

l2_norm = load_inline(
    name="l2_norm",
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=["l2_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2_norm = l2_norm

    def forward(self, x):
        return self.l2_norm.l2_norm_cuda(x)

batch_size = 32768
dim = 65535

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []