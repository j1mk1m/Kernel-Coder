import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication with dropout
matmul_dropout_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_dropout_kernel(const float* a, const float* b, float* c, int rows_a, int cols_a, int cols_b, float p) {
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < rows_a && col_idx < cols_b) {
        float sum = 0.0f;
        for (int k = 0; k < cols_a; ++k) {
            sum += a[row_idx * cols_a + k] * b[k * cols_b + col_idx];
        }

        // Apply dropout
        if ((float)rand() / RAND_MAX < p) {
            c[row_idx * cols_b + col_idx] = sum * (1.0f / p);
        } else {
            c[row_idx * cols_b + col_idx] = 0.0f;
        }
    }
}

torch::Tensor matmul_dropout_cuda(torch::Tensor a, torch::Tensor b, float p) {
    auto rows_a = a.size(0);
    auto cols_a = a.size(1);
    auto cols_b = b.size(1);
    auto out = torch::zeros({rows_a, cols_b}, a.options());

    const int block_size = 256;
    const int num_rows_blocks = (rows_a + block_size - 1) / block_size;
    const int num_cols_blocks = (cols_b + block_size - 1) / block_size;

    matmul_dropout_kernel<<<dim3(num_cols_blocks, num_rows_blocks), dim3(block_size, block_size)>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), rows_a, cols_a, cols_b, p);

    return out;
}
"""

matmul_dropout_cpp_source = (
    "torch::Tensor matmul_dropout_cuda(torch::Tensor a, torch::Tensor b, float p);"
)

# Compile the inline CUDA code for matrix multiplication with dropout
matmul_dropout = load_inline(
    name="matmul_dropout",
    cpp_sources=matmul_dropout_cpp_source,
    cuda_sources=matmul_dropout_source,
    functions=["matmul_dropout_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul_dropout = matmul_dropout

    def forward(self, x):
        x = self.matmul_dropout.matmul_dropout_cuda(x, x, self.dropout.p)
        x = torch.softmax(x, dim=1)
        return x

# Example usage
model_new = ModelNew(in_features, out_features, dropout_p)
inputs = get_inputs()
output_new = model_new(inputs[0])
print(output_new.shape)  # Should print torch.Size([128, 16384])

# Check correctness
original_model = Model(in_features, out_features, dropout_p)
output_original = original_model(inputs[0])
assert torch.allclose(output_new, output_original), "The outputs do not match."