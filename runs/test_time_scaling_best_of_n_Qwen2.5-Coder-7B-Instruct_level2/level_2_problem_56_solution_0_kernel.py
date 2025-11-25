import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication followed by sigmoid
matmul_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_sigmoid_kernel(const float* a, const float* b, float* out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        float sum = 0.0f;
        for (int k = 0; k < cols; ++k) {
            sum += a[row * cols + k] * b[k * cols + col];
        }
        out[row * cols + col] = 1.0f / (1.0f + exp(-sum));
    }
}

torch::Tensor matmul_sigmoid_cuda(torch::Tensor a, torch::Tensor b) {
    auto rows = a.size(0);
    auto cols = b.size(1);
    auto out = torch::zeros({rows, cols}, a.options());

    const int block_size = 16;
    const int num_rows = (rows + block_size - 1) / block_size;
    const int num_cols = (cols + block_size - 1) / block_size;

    dim3 grid(num_cols, num_rows);
    dim3 block(block_size, block_size);

    matmul_sigmoid_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), rows, cols);

    return out;
}
"""

matmul_sigmoid_cpp_source = (
    "torch::Tensor matmul_sigmoid_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for matrix multiplication followed by sigmoid
matmul_sigmoid = load_inline(
    name="matmul_sigmoid",
    cpp_sources=matmul_sigmoid_cpp_source,
    cuda_sources=matmul_sigmoid_source,
    functions=["matmul_sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.matmul_sigmoid = matmul_sigmoid

    def forward(self, x):
        x = self.matmul_sigmoid.matmul_sigmoid_cuda(x, torch.randn(hidden_size, input_size))
        x = torch.sum(x, dim=1, keepdim=True)
        return x

batch_size = 128
input_size = 32768
hidden_size = 32768

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size]