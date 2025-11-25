import torch
import torch.nn as nn
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
    int m = a.size(0);
    int n = b.size(1);
    int k = a.size(1);

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


# Define the custom CUDA kernel for Batch Normalization
bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bn_forward_kernel(const float* x, const float* mean, const float* var, float* y, float* inv_var, int batch_size, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * features) {
        int feature_idx = idx % features;
        float mean_val = mean[feature_idx];
        float var_val = var[feature_idx];
        float inv_var_val = inv_var[feature_idx];

        y[idx] = (x[idx] - mean_val) * inv_var_val;
    }
}

__global__ void bn_backward_kernel(const float* grad_output, const float* x, const float* mean, const float* inv_var, float* grad_input, float* grad_mean, float* grad_var, int batch_size, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * features) {
        int feature_idx = idx % features;
        float mean_val = mean[feature_idx];
        float inv_var_val = inv_var[feature_idx];

        grad_input[idx] = grad_output[idx] * inv_var_val;

        atomicAdd(grad_mean + feature_idx, -grad_output[idx]);
        atomicAdd(grad_var + feature_idx, -grad_output[idx] * (x[idx] - mean_val));
    }
}

torch::Tensor bn_forward_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, torch::Tensor inv_var) {
    int batch_size = x.size(0);
    int features = x.size(1);

    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (batch_size * features + block_size - 1) / block_size;

    bn_forward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), y.data_ptr<float>(), inv_var.data_ptr<float>(), batch_size, features);

    return y;
}

torch::Tensor bn_backward_cuda(torch::Tensor grad_output, torch::Tensor x, torch::Tensor mean, torch::Tensor inv_var) {
    int batch_size = x.size(0);
    int features = x.size(1);

    auto grad_input = torch::zeros_like(x);
    auto grad_mean = torch::zeros(features, x.options());
    auto grad_var = torch::zeros(features, x.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * features + block_size - 1) / block_size;

    bn_backward_kernel<<<num_blocks, block_size>>>(grad_output.data_ptr<float>(), x.data_ptr<float>(), mean.data_ptr<float>(), inv_var.data_ptr<float>(), grad_input.data_ptr<float>(), grad_mean.data_ptr<float>(), grad_var.data_ptr<float>(), batch_size, features);

    return grad_input;
}
"""

bn_cpp_source = (
    "torch::Tensor bn_forward_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, torch::Tensor inv_var);\n"
    "torch::Tensor bn_backward_cuda(torch::Tensor grad_output, torch::Tensor x, torch::Tensor mean, torch::Tensor inv_var);"
)

# Compile the inline CUDA code for Batch Normalization
bn = load_inline(
    name="bn",
    cpp_sources=bn_cpp_source,
    cuda_sources=bn_source,
    functions=["bn_forward_cuda", "bn_backward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* x, float* y, int batch_size, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * features) {
        int feature_idx = idx % features;
        float max_val = -FLT_MAX;
        for (int i = 0; i < batch_size; ++i) {
            max_val = fmax(max_val, x[i * features + feature_idx]);
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            sum_exp += exp(x[i * features + feature_idx] - max_val);
        }

        y[idx] = exp(x[idx] - max_val) / sum_exp;
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int features = x.size(1);

    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (batch_size * features + block_size - 1) / block_size;

    softmax_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, features);

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
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = gemm
        self.bn = bn
        self.scale = nn.Parameter(torch.ones(scale_shape))

    def forward(self, x):
        x = self.gemm.gemm_cuda(x.view(-1, in_features), x.new_zeros(in_features, out_features))  # Replace with actual weight matrix
        x = self.bn.bn_forward_cuda(x, x.mean(dim=0), torch.sqrt(x.var(dim=0) + 1e-5), 1 / torch.sqrt(x.var(dim=0) + 1e-5))
        x = self.scale * x
        x = self.softmax.softmax_cuda(x)
        return x