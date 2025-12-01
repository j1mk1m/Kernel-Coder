import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GELU
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ float gelu_device(float x) {
    return 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = gelu_device(x[idx]);
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

gelu_cpp_source = (
    "torch::Tensor gelu_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for GELU
gelu = load_inline(
    name="gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for GEMM and BatchNorm
gemm_batchnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gemm_batchnorm_kernel(const float* a, const float* b, const float* running_mean, const float* running_var, const float* weight, const float* bias, float* out, int batch_size, int in_features, int out_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_features) {
        float sum = 0.0f;
        for (int j = 0; j < in_features; ++j) {
            sum += a[i * in_features + j] * b[j * out_features + i];
        }

        sum -= running_mean[i];
        sum /= sqrt(running_var[i] + 1e-5);
        sum *= weight[i];
        sum += bias[i];

        out[i] = sum;
    }
}

torch::Tensor gemm_batchnorm_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = a.size(0);
    auto in_features = a.size(1);
    auto out_features = b.size(1);
    auto out = torch::zeros({batch_size, out_features});

    const int block_size = 256;
    const int num_blocks = (out_features + block_size - 1) / block_size;

    gemm_batchnorm_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), running_mean.data_ptr<float>(), running_var.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), batch_size, in_features, out_features);

    return out;
}
"""

gemm_batchnorm_cpp_source = (
    "torch::Tensor gemm_batchnorm_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code for GEMM and BatchNorm
gemm_batchnorm = load_inline(
    name="gemm_batchnorm",
    cpp_sources=gemm_batchnorm_cpp_source,
    cuda_sources=gemm_batchnorm_source,
    functions=["gemm_batchnorm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.weight = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        x = self.gemm(x)
        x = self.batch_norm(x)
        x = gemm_batchnorm.gemm_batchnorm_cuda(x, x, self.batch_norm.running_mean, self.batch_norm.running_var, self.weight, self.bias)
        x = gelu.gelu_cuda(x)
        return x