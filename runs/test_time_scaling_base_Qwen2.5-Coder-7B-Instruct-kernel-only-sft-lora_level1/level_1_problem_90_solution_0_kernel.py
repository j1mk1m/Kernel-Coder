import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cumulative product
cumulative_product_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumulative_product_kernel(const float* input, float* output, int batch_size, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * input_size) {
        return;
    }

    int batch_idx = idx / input_size;
    int input_idx = idx % input_size;

    float prod = 1.0f;
    for (int i = 0; i <= input_idx; ++i) {
        prod *= input[batch_idx * input_size + i];
    }

    output[idx] = prod;
}

torch::Tensor cumulative_product_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (batch_size * input_size + block_size - 1) / block_size;

    cumulative_product_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, input_size);

    return output;
}
"""

cumulative_product_cpp_source = (
    "torch::Tensor cumulative_product_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for cumulative product
cumulative_product = load_inline(
    name="cumulative_product",
    cpp_sources=cumulative_product_cpp_source,
    cuda_sources=cumulative_product_source,
    functions=["cumulative_product_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cumulative_product = cumulative_product

    def forward(self, x):
        # Reshape input to ensure correct order for cumprod
        x_reshaped = x.contiguous().view(-1, x.size(-1))
        result = self.cumulative_product.cumulative_product_cuda(x_reshaped)
        # Reshape back to original shape
        result = result.view_as(x)
        return result

# Example usage
if __name__ == "__main__":
    batch_size = 32768
    input_shape = (32768,)
    dim = 1
    model_new = ModelNew(dim)
    inputs = get_inputs()[0].cuda()
    outputs = model_new(inputs)
    print(outputs.shape)