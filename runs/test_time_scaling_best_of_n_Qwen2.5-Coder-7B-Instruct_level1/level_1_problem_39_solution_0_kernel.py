import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L2 normalization
l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void l2_norm_kernel(float* data, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float norm = 0.0f;
        for (int i = 0; i < dim; ++i) {
            norm += data[idx * dim + i] * data[idx * dim + i];
        }
        norm = sqrt(norm);
        for (int i = 0; i < dim; ++i) {
            data[idx * dim + i] /= norm;
        }
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor data, int batch_size, int dim) {
    auto out = data.clone();

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    l2_norm_kernel<<<num_blocks, block_size>>>(out.data_ptr<float>(), batch_size, dim);

    return out;
}
"""

l2_norm_cpp_source = (
    "torch::Tensor l2_norm_cuda(torch::Tensor data, int batch_size, int dim);"
)

# Compile the inline CUDA code for L2 normalization
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
        super(ModelNew, self).__init__()
        self.l2_norm = l2_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, dim = x.size(0), x.size(1)
        return self.l2_norm.l2_norm_cuda(x, batch_size, dim)