import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for reverse cumulative sum
reverse_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void reverse_cumsum_kernel(const float* input, float* output, int batch_size, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * seq_len) {
        return;
    }

    int batch_idx = idx / seq_len;
    int seq_idx = idx % seq_len;

    float sum = 0.0f;
    for (int i = seq_idx; i < seq_len; ++i) {
        sum += input[batch_idx * seq_len + i];
    }

    output[idx] = sum;
}

torch::Tensor reverse_cumsum_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (batch_size * seq_len + block_size - 1) / block_size;

    reverse_cumsum_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, seq_len);

    return output;
}
"""

reverse_cumsum_cpp_source = (
    "torch::Tensor reverse_cumsum_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for reverse cumulative sum
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
        super(ModelNew, self).__init__()
        self.dim = dim
        self.reverse_cumsum = reverse_cumsum

    def forward(self, x):
        flipped_x = x.flip(self.dim)
        cumsummed_x = self.reverse_cumsum.reverse_cumsum_cuda(flipped_x)
        return cumsummed_x.flip(self.dim)