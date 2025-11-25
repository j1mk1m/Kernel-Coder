import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for max reduction
max_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_reduction_kernel(const float* input, float* output, int batch_size, int dim1, int dim2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float max_val = -INFINITY;
        for (int i = 0; i < dim1; ++i) {
            for (int j = 0; j < dim2; ++j) {
                int linear_idx = idx * dim1 * dim2 + i * dim2 + j;
                if (input[linear_idx] > max_val) {
                    max_val = input[linear_idx];
                }
            }
        }
        output[idx] = max_val;
    }
}

torch::Tensor max_reduction_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);
    auto output = torch::zeros({batch_size}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    max_reduction_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim1, dim2);

    return output;
}
"""

max_reduction_cpp_source = (
    "torch::Tensor max_reduction_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for max reduction
max_reduction = load_inline(
    name="max_reduction",
    cpp_sources=max_reduction_cpp_source,
    cuda_sources=max_reduction_source,
    functions=["max_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.max_reduction = max_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_reduction.max_reduction_cuda(x)