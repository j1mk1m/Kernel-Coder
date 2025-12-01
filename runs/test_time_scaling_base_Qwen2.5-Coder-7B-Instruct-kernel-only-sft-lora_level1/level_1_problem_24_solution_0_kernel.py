import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

logsoftmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void logsoftmax_kernel(const float* input, float* output, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * dim) return;

    int batch_idx = idx / dim;
    int feature_idx = idx % dim;

    float max_val = -FLT_MAX;
    for (int i = 0; i < dim; ++i) {
        float val = input[batch_idx * dim + i];
        if (val > max_val) {
            max_val = val;
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum_exp += exp(input[batch_idx * dim + i] - max_val);
    }

    output[idx] = input[idx] - max_val - log(sum_exp);
}

torch::Tensor logsoftmax_cuda(torch::Tensor input, int dim) {
    auto batch_size = input.size(0);
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (batch_size * dim + block_size - 1) / block_size;

    logsoftmax_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);

    return output;
}
"""

logsoftmax_cpp_source = (
    "torch::Tensor logsoftmax_cuda(torch::Tensor input, int dim);"
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
        self.dim = dim
        self.logsoftmax = logsoftmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logsoftmax.logsoftmax_cuda(x, self.dim)