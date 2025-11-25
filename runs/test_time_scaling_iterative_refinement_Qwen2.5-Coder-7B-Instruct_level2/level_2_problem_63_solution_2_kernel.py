import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Load the custom CUDA kernels
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(const torch::Tensor& A, const torch::Tensor& B) {
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    auto C = torch::zeros({M, N}, A.options());

    const int block_size = 16;
    const dim3 blocks((N + block_size - 1) / block_size, (M + block_size - 1) / block_size);
    const dim3 threads(block_size, block_size);

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(const torch::Tensor& A, const torch::Tensor& B);"
)

relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = max(input[idx], 0.0f);
    }
}

torch::Tensor relu_cuda(const torch::Tensor& input) {
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (input.numel() + block_size - 1) / block_size;

    relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), input.numel());

    return output;
}
"""

relu_cpp_source = (
    "torch::Tensor relu_cuda(const torch::Tensor& input);"
)

# Compile the inline CUDA code for matmul and relu
matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

relu = load_inline(
    name="relu",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor
        self.matmul = matmul
        self.relu = relu

    def forward(self, x):
        x = self.linear(x)
        x = self.relu.relu_cuda(x)
        x = x / self.divisor
        return x


# Example usage
batch_size = 1024
in_features = 8192
out_features = 8192
divisor = 2.0

model = ModelNew(in_features, out_features, divisor)
x = torch.rand(batch_size, in_features)
output = model(x)
print(output.shape)