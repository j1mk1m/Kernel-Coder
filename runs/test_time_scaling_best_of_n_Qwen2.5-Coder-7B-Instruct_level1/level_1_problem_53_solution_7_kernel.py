import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for min reduction
min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void min_reduction_kernel(const float* input, float* output, int batch_size, int dim1, int dim2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float min_val = input[idx * dim1 * dim2];
        for (int i = 1; i < dim1 * dim2; ++i) {
            min_val = fmin(min_val, input[idx * dim1 * dim2 + i]);
        }
        output[idx] = min_val;
    }
}

torch::Tensor min_reduction_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);
    auto output = torch::zeros({batch_size}, torch::kFloat32).cuda();

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    min_reduction_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim1, dim2);

    return output;
}
"""

min_reduction_cpp_source = (
    "torch::Tensor min_reduction_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for min reduction
min_reduction = load_inline(
    name="min_reduction",
    cpp_sources=min_reduction_cpp_source,
    cuda_sources=min_reduction_source,
    functions=["min_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.min_reduction = min_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_reduction.min_reduction_cuda(x)