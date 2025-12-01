import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for sum reduction
sum_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_reduction_kernel(const float* input, float* output, int batch_size, int dim1, int dim2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim2) {
        int batch_idx = idx / dim2;
        int dim2_idx = idx % dim2;
        atomicAdd(&output[batch_idx], input[idx]);
    }
}

torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim) {
    auto batch_size = input.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);
    auto output = torch::zeros({batch_size, 1, dim2}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * dim2 + block_size - 1) / block_size;

    sum_reduction_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim1, dim2);

    return output;
}
"""

sum_reduction_cpp_source = (
    "torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim);"
)

# Compile the inline CUDA code for sum reduction
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
    """
    Optimized model using custom CUDA operators for sum reduction.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduction = sum_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sum_reduction.sum_reduction_cuda(x, self.dim)

# Example usage
if __name__ == "__main__":
    model_new = ModelNew(reduce_dim)
    inputs = get_inputs()
    outputs = model_new(inputs[0])
    print(outputs.shape)