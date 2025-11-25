import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for argmin operation
argmin_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void argmin_kernel(const float* data, int* indices, int numel, int dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < numel; i += stride) {
        int idx = i % dim;
        if (idx == 0) {
            indices[i / dim] = i;
        } else if (data[i] < data[indices[i / dim]]) {
            indices[i / dim] = i;
        }
    }
}

torch::Tensor argmin_cuda(torch::Tensor data, int dim) {
    auto size = data.size();
    auto output_size = size;
    output_size[dim] = 1;

    auto indices = torch::zeros(output_size, data.options().dtype(torch::kInt32));

    const int block_size = 256;
    const int num_blocks = (size.numel() + block_size - 1) / block_size;

    argmin_kernel<<<num_blocks, block_size>>>(data.data_ptr<float>(), indices.data_ptr<int>(), data.numel(), dim);

    return indices;
}
"""

argmin_cpp_source = (
    "torch::Tensor argmin_cuda(torch::Tensor data, int dim);"
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
        return self.argmin.argmin_cuda(x, self.dim)

# Example usage
if __name__ == "__main__":
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    dim = 1

    x = torch.rand(batch_size, dim1, dim2).cuda()
    model_new = ModelNew(dim).cuda()

    result = model_new(x)
    print(result)