import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for exclusive cumulative sum
exclusive_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void exclusive_cumsum_kernel(const float* input, float* output, int size, int dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += stride) {
        int index = i;
        for (int j = 0; j < dim; ++j) {
            index -= index & (-index);
        }
        output[i] = input[index];
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int index = tid + s;
            for (int j = 0; j < dim; ++j) {
                index -= index & (-index);
            }
            output[tid] += output[index];
        }
        __syncthreads();
    }
}

void exclusive_cumsum(float* input, float* output, int size, int dim) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    exclusive_cumsum_kernel<<<num_blocks, block_size>>>(input, output, size, dim);
}
"""

exclusive_cumsum_cpp_source = (
    "void exclusive_cumsum(float* input, float* output, int size, int dim);"
)

# Compile the inline CUDA code for exclusive cumulative sum
exclusive_cumsum = load_inline(
    name="exclusive_cumsum",
    cpp_sources=exclusive_cumsum_cpp_source,
    cuda_sources=exclusive_cumsum_source,
    functions=["exclusive_cumsum"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        batch_size = x.size(0)
        size = x.numel() // batch_size
        exclusive_cumsum_output = torch.empty_like(x)
        exclusive_cumsum(x.view(-1).cpu().numpy(), exclusive_cumsum_output.view(-1).cpu().numpy(), size, self.dim)
        return exclusive_cumsum_output.view_as(x)


# Example usage
model_new = ModelNew(dim=1)
inputs = get_inputs()
output_new = model_new(inputs[0])

model_ref = Model(dim=1)
output_ref = model_ref(inputs[0])

print("Output from ModelNew:", output_new)
print("Output from ModelRef:", output_ref)
print("Are outputs equal?", torch.allclose(output_new, output_ref))