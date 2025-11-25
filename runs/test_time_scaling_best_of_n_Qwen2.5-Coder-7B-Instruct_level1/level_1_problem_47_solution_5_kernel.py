import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

sum_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_reduction_kernel(const float* input, float* output, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < dim; ++i) {
            sum += input[idx * dim + i];
        }
        output[idx] = sum;
    }
}

torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim) {
    auto batch_size = input.size(0);
    auto output = torch::zeros({batch_size}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    sum_reduction_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);

    return output.view({batch_size, 1});
}
"""

sum_reduction_cpp_source = (
    "torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim);"
)

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
        return self.sum_reduction.sum_reduction_cuda(x, self.dim)

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [reduce_dim]