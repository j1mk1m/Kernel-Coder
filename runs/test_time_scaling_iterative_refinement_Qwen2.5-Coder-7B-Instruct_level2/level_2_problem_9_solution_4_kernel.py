import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void matmul_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, cudaStreamPerThread);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, (const float*)1.0, b, n, a, k, (const float*)0.0, c, n);

    cublasDestroy(handle);
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);
    auto out = torch::zeros({m, n}, a.options());

    matmul_kernel<<<1, 1>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), m, n, k);

    return out;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for matrix multiplication
matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for subtraction
subtraction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void subtraction_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

torch::Tensor subtraction_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    subtraction_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

subtraction_cpp_source = (
    "torch::Tensor subtraction_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for subtraction
subtraction = load_inline(
    name="subtraction",
    cpp_sources=subtraction_cpp_source,
    cuda_sources=subtraction_source,
    functions=["subtraction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for multiplication
multiplication_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void multiplication_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

torch::Tensor multiplication_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    multiplication_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

multiplication_cpp_source = (
    "torch::Tensor multiplication_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for multiplication
multiplication = load_inline(
    name="multiplication",
    cpp_sources=multiplication_cpp_source,
    cuda_sources=multiplication_source,
    functions=["multiplication_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for ReLU activation
relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* a, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fmaxf(a[idx], 0.0f);
    }
}

torch::Tensor relu_cuda(torch::Tensor a) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    relu_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

relu_cpp_source = (
    "torch::Tensor relu_cuda(torch::Tensor a);"
)

# Compile the inline CUDA code for ReLU activation
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
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.linear(x)
        x = matmul(matmul(x, x.t()), x)
        x = subtraction(x, self.subtract_value)
        x = multiplication(x, self.multiply_value)
        x = relu(x)
        return x


batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]