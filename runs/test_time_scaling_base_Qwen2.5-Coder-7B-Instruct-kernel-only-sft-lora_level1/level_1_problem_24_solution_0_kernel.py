import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

logsoftmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for LogSoftmax
__global__ void logsoftmax_kernel(const float* x, float* y, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * dim) {
        return;
    }

    int batch_idx = idx / dim;
    int dim_idx = idx % dim;

    float max_val = -INFINITY;
    for (int i = 0; i < dim; ++i) {
        max_val = fmaxf(max_val, x[batch_idx * dim + i]);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum_exp += expf(x[batch_idx * dim + i] - max_val);
    }

    y[idx] = x[batch_idx * dim + dim_idx] - max_val - logf(sum_exp);
}

torch::Tensor logsoftmax_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (batch_size * dim + block_size - 1) / block_size;

    logsoftmax_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, dim);

    return y;
}
"""

logsoftmax_cpp_source = (
    "torch::Tensor logsoftmax_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for LogSoftmax
logsoftmax = load_inline(
    name="logsoftmax",
    cpp_sources=logsoftmax_cpp_source,
    cuda_sources=logsoftmax_source,
    functions=["logsoftmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.logsoftmax = logsoftmax
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logsoftmax.logsoftmax_cuda(x)