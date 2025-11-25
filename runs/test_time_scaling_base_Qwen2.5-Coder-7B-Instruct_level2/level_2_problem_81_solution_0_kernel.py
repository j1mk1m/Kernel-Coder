import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GEMM
gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
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

torch::Tensor gemm_cuda(torch::Tensor a, torch::Tensor b) {
    int m = a.size(0);
    int n = b.size(1);
    int k = a.size(1);

    auto c = torch::zeros({m, n}, a.options());

    const int block_size = 32;
    dim3 grid((n + block_size - 1) / block_size, (m + block_size - 1) / block_size);
    dim3 block(block_size, block_size);

    gemm_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

    return c;
}
"""

gemm_cpp_source = (
    "torch::Tensor gemm_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for GEMM
gemm = load_inline(
    name="gemm",
    cpp_sources=gemm_cpp_source,
    cuda_sources=gemm_source,
    functions=["gemm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for combined swish, division, and clamping
swish_div_clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_div_clamp_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sigmoid_x = 1.0f / (1.0f + exp(-x[idx]));
        x[idx] = x[idx] * sigmoid_x / 2.0f;
        x[idx] = fmaxf(fminf(x[idx], 1.0f), -1.0f);
    }
}

torch::Tensor swish_div_clamp_cuda(torch::Tensor x) {
    auto size = x.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_div_clamp_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), size);

    return x;
}
"""

swish_div_clamp_cpp_source = (
    "torch::Tensor swish_div_clamp_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for combined swish, division, and clamping
swish_div_clamp = load_inline(
    name="swish_div_clamp",
    cpp_sources=swish_div_clamp_cpp_source,
    cuda_sources=swish_div_clamp_source,
    functions=["swish_div_clamp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for combined tanh and clamping
tanh_clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_clamp_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = tanh(x[idx]);
        x[idx] = fmaxf(fminf(x[idx], 1.0f), -1.0f);
    }
}

torch::Tensor tanh_clamp_cuda(torch::Tensor x) {
    auto size = x.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    tanh_clamp_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), size);

    return x;
}
"""

tanh_clamp_cpp_source = (
    "torch::Tensor tanh_clamp_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for combined tanh and clamping
tanh_clamp = load_inline(
    name="tanh_clamp",
    cpp_sources=tanh_clamp_cpp_source,
    cuda_sources=tanh_clamp_source,
    functions=["tanh_clamp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = gemm
        self.swish_div_clamp = swish_div_clamp
        self.tanh_clamp = tanh_clamp

    def forward(self, x):
        x = self.gemm.gemm_cuda(x, x.new_zeros(out_features))  # Simulate bias addition
        x = self.swish_div_clamp.swish_div_clamp_cuda(x)
        x = self.tanh_clamp.tanh_clamp_cuda(x)
        return x


def get_inputs():
    batch_size = 1024
    in_features = 8192
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]