import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for sum reduction
sum_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_reduction_kernel(const float* x, float* out, int batch_size, int dim1, int dim2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < dim1; ++i) {
            for (int j = 0; j < dim2; ++j) {
                sum += x[idx * dim1 * dim2 + i * dim2 + j];
            }
        }
        out[idx] = sum;
    }
}

torch::Tensor sum_reduction_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto dim1 = x.size(1);
    auto dim2 = x.size(2);
    auto out = torch::zeros({batch_size}, torch::kFloat32).cuda();

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    sum_reduction_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim1, dim2);

    return out.view({batch_size, 1, 1});
}
"""

sum_reduction_cpp_source = (
    "torch::Tensor sum_reduction_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for sum reduction
sum_reduction = load_inline(
    name="sum_reduction",
    cpp_sources=sum_reduction_cpp_source,
    cuda_sources=sum_reduction_source,
    functions=["sum_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduction = sum_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sum_reduction.sum_reduction_cuda(x)