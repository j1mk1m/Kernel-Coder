import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

exclusive_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void exclusive_cumsum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int dim_size,
    int dim) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * dim_size) return;

    int row = index / dim_size;
    int col = index % dim_size;

    scalar_t sum = 0;
    for (int i = 0; i <= col; ++i) {
        sum += input[row * dim_size + i];
    }

    output[index] = sum - input[row * dim_size + col];
}

std::tuple<torch::Tensor> exclusive_cumsum_cuda(
    torch::Tensor input,
    int dim) {

    const int batch_size = input.size(0);
    const int dim_size = input.size(dim);
    auto output = torch::zeros_like(input);

    const int total_elements = batch_size * dim_size;
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "exclusive_cumsum_cuda", ([&] {
        exclusive_cumsum_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim_size,
            dim);
    }));

    return std::make_tuple(output);
}
"""

exclusive_cumsum_cpp = (
    "std::tuple<torch::Tensor> exclusive_cumsum_cuda(torch::Tensor input, int dim);"
)

exclusive_cumsum = load_inline(
    name="exclusive_cumsum",
    cpp_sources=exclusive_cumsum_cpp,
    cuda_sources=exclusive_cumsum_source,
    functions=["exclusive_cumsum_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.exclusive_cumsum = exclusive_cumsum

    def forward(self, x):
        return self.exclusive_cumsum.exclusive_cumsum_cuda(x, self.dim)[0]