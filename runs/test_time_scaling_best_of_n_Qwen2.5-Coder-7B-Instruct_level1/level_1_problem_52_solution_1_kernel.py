import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for argmin operation
argmin_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void argmin_kernel(const float* x, float* out, int batch_size, int dim1, int dim2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim1) {
        int min_val_idx = -1;
        float min_val = INFINITY;
        for (int d = 0; d < dim2; ++d) {
            int curr_idx = idx * dim2 + d;
            float val = x[curr_idx];
            if (val < min_val) {
                min_val = val;
                min_val_idx = d;
            }
        }
        out[idx] = static_cast<float>(min_val_idx);
    }
}

torch::Tensor argmin_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto dim1 = x.size(1);
    auto dim2 = x.size(2);
    auto out = torch::zeros({batch_size, dim1}, torch::kFloat32);

    const int block_size = 256;
    const int num_blocks = (batch_size * dim1 + block_size - 1) / block_size;

    argmin_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim1, dim2);

    return out;
}
"""

argmin_cpp_source = (
    "torch::Tensor argmin_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for argmin operation
argmin = load_inline(
    name="argmin",
    cpp_sources=argmin_cpp_source,
    cuda_sources=argmin_source,
    functions=["argmin_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmin = argmin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmin.argmin_cuda(x)