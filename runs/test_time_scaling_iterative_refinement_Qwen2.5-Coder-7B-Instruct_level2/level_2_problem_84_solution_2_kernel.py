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
    int m = a.size(0);
    int n = b.size(1);
    int k = a.size(1);

    auto options = a.options().dtype(torch::kFloat32);
    auto c = torch::empty({m, n}, options);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

    return c;
}
"""

# Define the custom CUDA kernel for Batch Normalization
bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bn_kernel(const float* x, float* y, float* mean, float* var, float* gamma, float* beta, float eps, int size) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        shared[tid] = x[i];
        __syncthreads();

        int stride = blockDim.x * gridDim.x;
        for (int s = stride / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared[tid] += shared[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            mean[blockIdx.x] = shared[0] / size;
        }
    }

    __syncthreads();

    if (i < size) {
        shared[tid] = (x[i] - mean[blockIdx.x]) * (x[i] - mean[blockIdx.x]);
        __syncthreads();

        for (int s = stride / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared[tid] += shared[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            var[blockIdx.x] = shared[0] / size + eps;
        }
    }

    __syncthreads();

    if (i < size) {
        y[i] = gamma[blockIdx.x] * (x[i] - mean[blockIdx.x]) / sqrt(var[blockIdx.x]) + beta[blockIdx.x];
    }
}

torch::Tensor bn_cuda(torch::Tensor x, float* mean, float* var, float* gamma, float* beta, float eps) {
    int size = x.numel();
    int batch_size = x.size(0);

    auto options = x.options().dtype(torch::kFloat32);
    auto y = torch::empty_like(x);
    auto means = torch::empty(batch_size, options);
    auto vars = torch::empty(batch_size, options);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    bn_kernel<<<blocksPerGrid, threadsPerBlock, sizeof(float) * threadsPerBlock>>>(x.data_ptr<float>(), y.data_ptr<float>(), means.data_ptr<float>(), vars.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), eps, size);

    return y;
}
"""

# Define the custom CUDA kernel for Scale
scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scale_kernel(const float* x, float* y, const float* scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = x[idx] * scale[0];
    }
}

torch::Tensor scale_cuda(torch::Tensor x, const float* scale) {
    int size = x.numel();

    auto options = x.options().dtype(torch::kFloat32);
    auto y = torch::empty_like(x);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    scale_kernel<<<blocksPerGrid, threadsPerBlock>>>(x.data_ptr<float>(), y.data_ptr<float>(), scale, size);

    return y;
}
"""

# Define the custom CUDA kernel for Softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < size; ++i) {
            if (x[i] > max_val) {
                max_val = x[i];
            }
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum_exp += exp(x[i] - max_val);
        }

        y[idx] = exp(x[idx] - max_val) / sum_exp;
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    int size = x.numel();

    auto options = x.options().dtype(torch::kFloat32);
    auto y = torch::empty_like(x);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

# Compile the inline CUDA code for Gemm, Batch Normalization, Scale, and Softmax
gemm = load_inline(name="gemm", cpp_sources="", cuda_sources=gemm_source, functions=["gemm_cuda"], verbose=True, extra_cflags=[""], extra_ldflags=[""])
bn = load_inline(name="bn", cpp_sources="", cuda_sources=bn_source, functions=["bn_cuda"], verbose=True, extra_cflags=[""], extra_ldflags=[""])
scale = load_inline(name="scale", cpp_sources="", cuda_sources=scale_source, functions=["scale_cuda"], verbose=True, extra_cflags=[""], extra_ldflags=[""])
softmax = load_inline(name="softmax", cpp_sources="", cuda_sources=softmax_source, functions=["softmax_cuda"], verbose=True, extra_cflags=[""], extra_ldflags=[""])

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = gemm
        self.bn = bn
        self.scale = scale
        self.softmax = softmax

    def forward(self, x):
        x = self.gemm.gemm_cuda(x)
        x = self.bn.bn_cuda(x, torch.zeros(x.size(0)), torch.zeros(x.size(0)), torch.ones(x.size(0)), torch.zeros(x.size(0)), bn_eps)
        x = self.scale.scale_cuda(x, torch.tensor([1.0]))
        x = self.softmax.softmax_cuda(x)
        return x

# Example usage
if __name__ == "__main__":
    batch_size = 1024
    in_features = 8192
    out_features = 8192
    bn_eps = 1e-5
    bn_momentum = 0.1
    scale_shape = (1,)

    model_new = ModelNew(in_features, out_features, bn_eps, bn_momentum, scale_shape)
    inputs = torch.randn(batch_size, in_features).cuda()
    outputs = model_new(inputs.cuda())
    print(outputs.shape)