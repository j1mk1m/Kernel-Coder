import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication and subtraction
matmul_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_subtract_kernel(const float* A, const float* B, float* C, float subtract_value, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum - subtract_value;
    }
}

torch::Tensor matmul_subtract_cuda(torch::Tensor A, torch::Tensor B, float subtract_value) {
    auto m = A.size(0);
    auto n = B.size(1);
    auto k = A.size(1);
    auto C = torch::zeros({m, n}, A.options());

    const int block_size = 256;
    const int grid_x = (n + block_size - 1) / block_size;
    const int grid_y = (m + block_size - 1) / block_size;

    matmul_subtract_kernel<<<grid_x, grid_y, 0, at::cuda::getCurrentCUDAStream()>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), subtract_value, m, n, k);

    return C;
}
"""

matmul_subtract_cpp_source = (
    "torch::Tensor matmul_subtract_cuda(torch::Tensor A, torch::Tensor B, float subtract_value);"
)

# Compile the inline CUDA code for matrix multiplication and subtraction
matmul_subtract = load_inline(
    name="matmul_subtract",
    cpp_sources=matmul_subtract_cpp_source,
    cuda_sources=matmul_subtract_source,
    functions=["matmul_subtract_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.matmul_subtract = matmul_subtract

    def forward(self, x):
        x = self.linear(x)
        x = self.matmul_subtract.matmul_subtract_cuda(x, self.linear.weight.t(), self.subtract_value)
        x = x * self.multiply_value
        x = torch.relu(x)
        return x

# Example usage
if __name__ == "__main__":
    batch_size = 1024
    in_features = 8192
    out_features = 8192
    subtract_value = 2.0
    multiply_value = 1.5

    inputs = get_inputs()[0].cuda()
    model = ModelNew(in_features, out_features, subtract_value, multiply_value).cuda()

    output = model(inputs)
    print(output.shape)