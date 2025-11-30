import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Batch Matrix Multiplication
bmm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bmm_kernel(const float* a, const float* b, float* c, int n, int m, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < n && col < m) {
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * m + col];
        }
        c[row * m + col] = sum;
    }
}

torch::Tensor bmm_cuda(torch::Tensor a, torch::Tensor b) {
    auto n = a.size(0);
    auto m = b.size(1);
    auto k = a.size(1);
    auto c = torch::zeros({n, m}, a.options());

    const int block_size = 16;
    dim3 grid((m + block_size - 1) / block_size, (n + block_size - 1) / block_size);
    dim3 block(block_size, block_size);

    bmm_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n, m, k);

    return c;
}
"""

bmm_cpp_source = (
    "torch::Tensor bmm_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for BMM
bmm = load_inline(
    name="bmm",
    cpp_sources=bmm_cpp_source,
    cuda_sources=bmm_source,
    functions=["bmm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Custom CUDA kernel for Instance Normalization
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void instance_norm_kernel(const float* x, float* mean, float* var, float* out, int n, int c, int h, int w, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * c * h * w) {
        int ch = idx / (h * w);
        int hw_idx = idx % (h * w);
        mean[ch] += x[idx];
        var[ch] += x[idx] * x[idx];
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        mean[ch] /= (float)(h * w);
        var[ch] /= (float)(h * w);
        var[ch] -= mean[ch] * mean[ch];
        var[ch] += eps;
        var[ch] = rsqrt(var[ch]);
    }

    __syncthreads();

    if (idx < n * c * h * w) {
        out[idx] = x[idx] * var[ch];
    }
}

torch::Tensor instance_norm_cuda(torch::Tensor x, float eps) {
    auto n = x.size(0);
    auto c = x.size(1);
    auto h = x.size(2);
    auto w = x.size(3);
    auto mean = torch::zeros({c}, x.options());
    auto var = torch::zeros({c}, x.options());
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    dim3 grid((n * c * h * w + block_size - 1) / block_size);
    dim3 block(block_size);

    instance_norm_kernel<<<grid, block>>>(x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), out.data_ptr<float>(), n, c, h, w, eps);

    return out;
}
"""

instance_norm_cpp_source = (
    "torch::Tensor instance_norm_cuda(torch::Tensor x, float eps);"
)

# Compile the inline CUDA code for Instance Normalization
instance_norm = load_inline(
    name="instance_norm",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = bmm
        self.instance_norm = instance_norm

    def forward(self, x, y):
        x = self.bmm.bmm_cuda(x.view(-1, in_features, out_features), y.view(-1, out_features, in_features))
        x = x.view(-1, out_features)
        x = self.instance_norm.instance_norm_cuda(x.unsqueeze(1).unsqueeze(1).unsqueeze(1)).squeeze(1).squeeze(1).squeeze(1)
        x = x + y
        x = x * y
        return x