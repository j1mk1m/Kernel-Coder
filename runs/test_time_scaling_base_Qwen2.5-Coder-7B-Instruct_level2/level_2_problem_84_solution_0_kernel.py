import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Gemm
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
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);
    auto c = torch::zeros({m, n}, a.options());

    const int block_size = 32;
    const int num_blocks_x = (n + block_size - 1) / block_size;
    const int num_blocks_y = (m + block_size - 1) / block_size;

    gemm_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

    return c;
}
"""

gemm_cpp_source = (
    "torch::Tensor gemm_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for Gemm
gemm = load_inline(
    name="gemm",
    cpp_sources=gemm_cpp_source,
    cuda_sources=gemm_source,
    functions=["gemm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for BatchNorm
batchnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batchnorm_kernel(const float* x, float* y, const float* mean, const float* invstd, const float* gamma, const float* beta, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = gamma[idx] * (x[idx] - mean[idx]) * invstd[idx] + beta[idx];
    }
}

torch::Tensor batchnorm_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor invstd, torch::Tensor gamma, torch::Tensor beta) {
    auto size = x.numel();
    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    batchnorm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), mean.data_ptr<float>(), invstd.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), size);

    return y;
}
"""

batchnorm_cpp_source = (
    "torch::Tensor batchnorm_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor invstd, torch::Tensor gamma, torch::Tensor beta);"
)

# Compile the inline CUDA code for BatchNorm
batchnorm = load_inline(
    name="batchnorm",
    cpp_sources=batchnorm_cpp_source,
    cuda_sources=batchnorm_source,
    functions=["batchnorm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* x, float* y, int size) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * blockDim.x + tid;

    if (i < size) {
        sdata[tid] = exp(x[i]);
    } else {
        sdata[tid] = 0.0f;
    }

    __syncthreads();

    int s = blockDim.x / 2;
    while (s > 0) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
        s >>= 1;
    }

    if (tid == 0) {
        float sum = sdata[0];
        atomicAdd(&sum, 0.0f);
        y[bid * blockDim.x] = sdata[0] / sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    softmax_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

softmax_cpp_source = (
    "torch::Tensor softmax_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for Softmax
softmax = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that uses custom CUDA kernels for Gemm, BatchNorm, and Softmax.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = gemm
        self.bn = batchnorm
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = softmax

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm.gemm_cuda(x, self.weight)
        x = self.bn.batchnorm_cuda(x, self.running_mean, self.running_var, self.weight, self.bias)
        x = self.scale * x
        x = self.softmax.softmax_cuda(x)
        return x

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Load weight from state dict
        self.weight = state_dict[prefix + 'weight']
        self.bias = state_dict[prefix + 'bias']

        # Initialize running mean and variance
        self.running_mean = torch.zeros(self.weight.size(0))
        self.running_var = torch.ones(self.weight.size(0))

        super(ModelNew, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


# Example usage
if __name__ == "__main__":
    model_new = ModelNew(in_features, out_features, bn_eps, bn_momentum, scale_shape)
    inputs = get_inputs()[0].cuda()
    outputs = model_new(inputs)
    print(outputs.shape)