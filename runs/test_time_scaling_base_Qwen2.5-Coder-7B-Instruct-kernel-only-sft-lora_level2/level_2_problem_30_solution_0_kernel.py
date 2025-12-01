import torch
import torch.nn as nn
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

void gemm_cuda(const float* a, const float* b, float* c, int m, int n, int k) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, m, n, k);
}
"""

gemm_cpp_source = (
    "void gemm_cuda(const float* a, const float* b, float* c, int m, int n, int k);"
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


# Define the custom CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_norm_kernel(const float* input, float* mean, float* var, float* gamma, float* beta, float* output, int batch_size, int channels, int height, int width, int group_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < batch_size * channels * height * width) {
        int ch = index / (height * width);
        int g = ch / group_size;
        int h = (index % (height * width)) / width;
        int w = index % width;

        float val = input[index];
        atomicAdd(&mean[g], val);
        atomicAdd(&var[g], val * val);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        mean[g] /= (height * width);
        var[g] /= (height * width);
        var[g] -= mean[g] * mean[g];
        var[g] = sqrt(var[g] + 1e-5);
    }

    __syncthreads();

    if (index < batch_size * channels * height * width) {
        int ch = index / (height * width);
        int g = ch / group_size;
        int h = (index % (height * width)) / width;
        int w = index % width;

        float normalized = (input[index] - mean[g]) / var[g];
        output[index] = gamma[ch] * normalized + beta[ch];
    }
}

void group_norm_cuda(const float* input, float* mean, float* var, float* gamma, float* beta, float* output, int batch_size, int channels, int height, int width, int group_size) {
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((batch_size * channels * height * width + threadsPerBlock.x - 1) / threadsPerBlock.x);

    group_norm_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, mean, var, gamma, beta, output, batch_size, channels, height, width, group_size);
}
"""

group_norm_cpp_source = (
    "void group_norm_cuda(const float* input, float* mean, float* var, float* gamma, float* beta, float* output, int batch_size, int channels, int height, int width, int group_size);"
)

# Compile the inline CUDA code for Group Normalization
group_norm = load_inline(
    name="group_norm",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for HardTanh
hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float* input, float* output, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(fminf(input[idx], max_val), min_val);
    }
}

void hardtanh_cuda(const float* input, float* output, int size, float min_val, float max_val) {
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    hardtanh_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, size, min_val, max_val);
}
"""

hardtanh_cpp_source = (
    "void hardtanh_cuda(const float* input, float* output, int size, float min_val, float max_val);"
)

# Compile the inline CUDA code for HardTanh
hardtanh = load_inline(
    name="hardtanh",
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_source,
    functions=["hardtanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        batch_size = x.size(0)
        height = x.size(2)
        width = x.size(3)

        # Perform GEMM using the custom CUDA kernel
        weight = torch.randn(self.out_features, self.in_features).cuda()
        bias = torch.randn(self.out_features).cuda()
        output_gemm = torch.zeros(batch_size, self.out_features, height, width).cuda()
        gemm_cuda(x.view(batch_size, -1).contiguous().data_ptr(), weight.contiguous().data_ptr(), output_gemm.view(batch_size, -1).contiguous().data_ptr(), batch_size, height * width, self.in_features)

        # Perform Group Normalization using the custom CUDA kernel
        mean = torch.zeros(self.num_groups).cuda()
        var = torch.zeros(self.num_groups).cuda()
        gamma = torch.ones(self.out_features).cuda()
        beta = torch.zeros(self.out_features).cuda()
        output_group_norm = torch.zeros_like(output_gemm).cuda()
        group_norm_cuda(output_gemm.view(batch_size, -1).contiguous().data_ptr(), mean.contiguous().data_ptr(), var.contiguous().data_ptr(), gamma.contiguous().data_ptr(), beta.contiguous().data_ptr(), output_group_norm.view(batch_size, -1).contiguous().data_ptr(), batch_size, self.out_features, height, width, self.out_features // self.num_groups)

        # Perform HardTanh using the custom CUDA kernel
        output_hardtanh = torch.zeros_like(output_group_norm).cuda()
        hardtanh_cuda(output_group_norm.view(-1).contiguous().data_ptr(), output_hardtanh.view(-1).contiguous().data_ptr(), batch_size * self.out_features * height * width, self.hardtanh_min, self.hardtanh_max)

        return output_hardtanh.view(batch_size, self.out_features, height, width)