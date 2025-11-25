import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for exclusive cumulative sum
exclusive_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void exclusive_cumsum_kernel(float* x, int batch_size, int seq_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= seq_len) {
        return;
    }

    // Perform the exclusive cumulative sum in-place
    for (int i = 1; i <= batch_size; ++i) {
        x[tid * batch_size + i] += x[tid * batch_size + i - 1];
    }
}

torch::Tensor exclusive_cumsum_cuda(torch::Tensor x, int dim) {
    auto batch_size = x.size(dim);
    auto seq_len = x.size(1 - dim);

    exclusive_cumsum_kernel<<<(seq_len + 255) / 256, 256>>>(x.data_ptr<float>(), batch_size, seq_len);

    return x;
}
"""

exclusive_cumsum_cpp_source = (
    "torch::Tensor exclusive_cumsum_cuda(torch::Tensor x, int dim);"
)

# Compile the inline CUDA code for exclusive cumulative sum
exclusive_cumsum = load_inline(
    name="exclusive_cumsum",
    cpp_sources=exclusive_cumsum_cpp_source,
    cuda_sources=exclusive_cumsum_source,
    functions=["exclusive_cumsum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.exclusive_cumsum = exclusive_cumsum

    def forward(self, x):
        exclusive_cumsum = torch.cat((torch.zeros_like(x.select(self.dim, 0).unsqueeze(self.dim)), x), dim=self.dim)[:-1]
        return self.exclusive_cumsum.exclusive_cumsum_cuda(exclusive_cumsum, self.dim)