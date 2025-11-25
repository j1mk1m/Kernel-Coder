import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batch matrix multiplication
bmm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bmm_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
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

torch::Tensor bmm_cuda(torch::Tensor a, torch::Tensor b) {
    int m = a.size(0);
    int n = b.size(1);
    int k = a.size(1);

    auto c = torch::zeros({m, n}, a.options());

    const int block_size = 16;
    const int num_blocks_x = (n + block_size - 1) / block_size;
    const int num_blocks_y = (m + block_size - 1) / block_size;

    bmm_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

    return c;
}
"""

bmm_cpp_source = (
    "torch::Tensor bmm_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for batch matrix multiplication
bmm = load_inline(
    name="bmm",
    cpp_sources=bmm_cpp_source,
    cuda_sources=bmm_source,
    functions=["bmm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for instance normalization
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void instance_norm_kernel(const float* x, float* mean, float* var, float* y, float* gamma, float* beta, float eps, int channels, int height, int width) {
    int ch = blockIdx.z * blockDim.z + threadIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (ch < channels && h < height && w < width) {
        float sum = 0.0f;
        float sq_sum = 0.0f;
        for (int i = 0; i < height * width; ++i) {
            sum += x[ch * height * width + i];
            sq_sum += x[ch * height * width + i] * x[ch * height * width + i];
        }

        mean[ch] = sum / (height * width);
        var[ch] = sq_sum / (height * width) - mean[ch] * mean[ch];

        y[ch * height * width + h * width + w] = gamma[ch] * (x[ch * height * width + h * width + w] - mean[ch]) / sqrt(var[ch] + eps) + beta[ch];
    }
}

void instance_norm_backward_kernel(float* grad_x, const float* grad_y, const float* x, const float* mean, const float* var, const float* gamma, const float* beta, float eps, int channels, int height, int width) {
    for (int ch = 0; ch < channels; ++ch) {
        for (int i = 0; i < height * width; ++i) {
            float diff = x[ch * height * width + i] - mean[ch];
            float denom = sqrt(var[ch] + eps);
            float dvar = -0.5 * diff * diff / (denom * denom * denom);
            float dmean = -1.0 / denom;
            float dydx = gamma[ch] / denom;

            grad_x[ch * height * width + i] = dydx * grad_y[ch * height * width + i] + dvar * (2.0 * diff / (height * width)) + dmean / (height * width);
        }
    }
}

torch::Tensor instance_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps) {
    int channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);

    auto mean = torch::zeros({channels}, x.options());
    auto var = torch::zeros({channels}, x.options());
    auto y = torch::zeros_like(x);

    const int block_size = 16;
    const int num_blocks_x = (width + block_size - 1) / block_size;
    const int num_blocks_y = (height + block_size - 1) / block_size;
    const int num_blocks_z = (channels + block_size - 1) / block_size;

    instance_norm_kernel<<<dim3(num_blocks_x, num_blocks_y, num_blocks_z), dim3(block_size, block_size, block_size)>>>(x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), y.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), eps, channels, height, width);

    return y;
}
"""

instance_norm_cpp_source = (
    "torch::Tensor instance_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps);"
)

# Compile the inline CUDA code for instance normalization
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
        x = self.bmm.bmm_cuda(x, y)
        x = self.instance_norm.instance_norm_cuda(x, torch.ones_like(x), torch.zeros_like(x), eps)
        x = x + y
        x = x * y
        return x

batch_size = 1024  # Increased batch size
in_features = 8192  # Increased input features
out_features = 8192  # Increased output features

def get_inputs():
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, out_features)]

def get_init_inputs():
    return [in_features, out_features]