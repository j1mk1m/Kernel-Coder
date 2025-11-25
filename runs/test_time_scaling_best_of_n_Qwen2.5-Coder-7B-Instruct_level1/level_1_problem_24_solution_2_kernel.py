import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for LogSoftmax
logsoftmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define EPSILON 1e-7

__global__ void logsoftmax_kernel(const float* input, float* output, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * dim) {
        return;
    }

    int row = idx / dim;
    int col = idx % dim;

    float max_val = -FLT_MAX;
    for (int j = 0; j < dim; ++j) {
        if (input[row * dim + j] > max_val) {
            max_val = input[row * dim + j];
        }
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < dim; ++j) {
        sum_exp += exp(input[row * dim + j] - max_val);
    }

    output[idx] = input[idx] - max_val - log(sum_exp);
}

torch::Tensor logsoftmax_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto dim = input.size(1);
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (batch_size * dim + block_size - 1) / block_size;

    logsoftmax_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);

    return output;
}
"""

logsoftmax_cpp_source = (
    "torch::Tensor logsoftmax_cuda(torch::Tensor input);"
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
    """
    Simple model that performs a LogSoftmax activation using a custom CUDA kernel.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        return logsoftmax.logsoftmax_cuda(x)