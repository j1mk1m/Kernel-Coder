import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for prefix sum
prefix_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void prefix_sum_kernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

torch::Tensor prefix_sum_cuda(torch::Tensor input) {
    int n = input.size(0);
    auto output = torch::zeros(n, input.options());

    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    prefix_sum_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);

    return output;
}
"""

prefix_sum_cpp_source = (
    "torch::Tensor prefix_sum_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for prefix sum
prefix_sum = load_inline(
    name="prefix_sum",
    cpp_sources=prefix_sum_cpp_source,
    cuda_sources=prefix_sum_source,
    functions=["prefix_sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.prefix_sum = prefix_sum

    def forward(self, x):
        cumsum = torch.zeros_like(x)
        for i in range(x.size(self.dim)):
            slice = x.narrow(self.dim, 0, i + 1)
            cumsum_narrow = self.prefix_sum.prefix_sum_cuda(slice.sum(dim=self.dim, keepdim=True))
            cumsum.narrow(self.dim, 0, i + 1).copy_(cumsum_narrow)
        return cumsum