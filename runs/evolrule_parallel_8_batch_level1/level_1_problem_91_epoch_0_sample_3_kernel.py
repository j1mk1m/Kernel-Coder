import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 32768
input_shape = (32768,)
dim = 1

reverse_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" __global__ void reverse_cumsum_2d_dim1(const float* input, float* output, int batch_size, int dim_size) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    for (int i = dim_size - 1; i >= 0; --i) {
        int idx = row * dim_size + i;
        if (i == dim_size - 1) {
            output[idx] = input[idx];
        } else {
            output[idx] = input[idx] + output[idx + 1];
        }
    }
}

torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim) {
    int batch_size = input.size(0);
    int dim_size = input.size(1);
    auto output = torch::empty_like(input);

    const int num_blocks = batch_size;  // Each block handles a row.
    const int threads_per_block = 1;    // Each block uses a single thread.

    reverse_cumsum_2d_dim1<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim_size
    );

    return output;
}
"""

reverse_cumsum_cpp_source = (
    "torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim);"
)

reverse_cumsum = load_inline(
    name="reverse_cumsum",
    cpp_sources=reverse_cumsum_cpp_source,
    cuda_sources=reverse_cumsum_source,
    functions=["reverse_cumsum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reverse_cumsum = reverse_cumsum

    def forward(self, x):
        return self.reverse_cumsum.reverse_cumsum_cuda(x, self.dim)

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [dim]