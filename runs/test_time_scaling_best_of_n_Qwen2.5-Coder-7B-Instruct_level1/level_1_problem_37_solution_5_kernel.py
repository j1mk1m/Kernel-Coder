import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Frobenius norm calculation
frobenius_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void frobenius_norm_kernel(const float* x, float* norm, int batch_size, int features, int dim1, int dim2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * features) {
        float sum = 0.0f;
        for (int i = 0; i < dim1 * dim2; ++i) {
            sum += x[idx * dim1 * dim2 + i] * x[idx * dim1 * dim2 + i];
        }
        atomicAdd(norm, sqrt(sum));
    }
}

float frobenius_norm_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto features = x.size(1);
    auto dim1 = x.size(2);
    auto dim2 = x.size(3);
    auto norm = torch::zeros({1}, x.options().dtype(torch::kFloat32));

    const int block_size = 256;
    const int num_blocks = (batch_size * features + block_size - 1) / block_size;

    frobenius_norm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), norm.data_ptr<float>(), batch_size, features, dim1, dim2);

    return norm.item<float>();
}
"""

frobenius_norm_cpp_source = (
    "float frobenius_norm_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for Frobenius norm calculation
frobenius_norm = load_inline(
    name="frobenius_norm",
    cpp_sources=frobenius_norm_cpp_source,
    cuda_sources=frobenius_norm_source,
    functions=["frobenius_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.frobenius_norm = frobenius_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.frobenius_norm.frobenius_norm_cuda(x)
        return x / norm