import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);
    auto c = torch::zeros({m, n}, a.options());

    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);

    matmul_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

    return c;
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

# Define the custom CUDA kernel for Swish activation
swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = x[idx] * (1.0f + tanh(x[idx] * 0.5f));
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = x.clone();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_kernel<<<num_blocks, block_size>>>(out.data_ptr<float>(), size);

    return out;
}
"""

swish_cpp_source = (
    "torch::Tensor swish_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for Swish activation
swish = load_inline(
    name="swish",
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_source,
    functions=["swish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for adding bias
add_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_bias_kernel(const float* x, const float* bias, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = x[idx] + bias[0];
    }
}

torch::Tensor add_bias_cuda(torch::Tensor x, torch::Tensor bias) {
    auto size = x.numel();
    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    add_bias_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

add_bias_cpp_source = (
    "torch::Tensor add_bias_cuda(torch::Tensor x, torch::Tensor bias);"
)

# Compile the inline CUDA code for adding bias
add_bias = load_inline(
    name="add_bias",
    cpp_sources=add_bias_cpp_source,
    cuda_sources=add_bias_source,
    functions=["add_bias_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = matmul
        self.swish = swish
        self.add_bias = add_bias
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        x = self.matmul.matmul_cuda(x, self.weight)
        x = self.swish.swish_cuda(x)
        x = self.add_bias.add_bias_cuda(x, self.bias)
        x = self.group_norm(x)
        return x


# Example usage
if __name__ == "__main__":
    batch_size = 32768
    in_features = 1024
    out_features = 4096
    num_groups = 64
    bias_shape = (out_features,)

    model_new = ModelNew(in_features, out_features, num_groups, bias_shape)
    x = torch.rand(batch_size, in_features).cuda()
    output = model_new(x)
    print(output.shape)