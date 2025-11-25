import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for GEMM
gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    auto C = torch::zeros({M, N}, A.options());

    const int block_size = 32;
    dim3 grid((N + block_size - 1) / block_size, (M + block_size - 1) / block_size);
    dim3 block(block_size, block_size);

    gemm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

gemm_cpp_source = (
    "torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B);"
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


# Custom CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_norm_kernel(const float* input, float* output, float* mean, float* var, int channels, int groups, int height, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= channels * height * width) return;

    int g = index / (height * width);
    int c = (g * groups) + (index % (channels / groups));
    int h = (index % (height * width)) / width;
    int w = (index % (height * width)) % width;

    float sum = 0.0f;
    float sq_sum = 0.0f;
    for (int i = 0; i < height * width; ++i) {
        int ci = (c * height * width) + i;
        sum += input[ci];
        sq_sum += input[ci] * input[ci];
    }

    mean[g * groups + c] = sum / (height * width);
    var[g * groups + c] = sq_sum / (height * width) - (mean[g * groups + c] * mean[g * groups + c]);

    output[index] = (input[index] - mean[g * groups + c]) / sqrt(var[g * groups + c] + 1e-5);
}

torch::Tensor group_norm_cuda(torch::Tensor input, int groups) {
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto mean = torch::zeros({groups, channels / groups});
    auto var = torch::zeros({groups, channels / groups});
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    dim3 grid((channels * height * width + block_size - 1) / block_size);
    dim3 block(block_size);

    group_norm_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), channels, groups, height, width);

    return output;
}
"""

group_norm_cpp_source = (
    "torch::Tensor group_norm_cuda(torch::Tensor input, int groups);"
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


# Custom CUDA kernel for HardTanh
hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float* input, float* output, float min_val, float max_val, int elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;

    float value = input[index];
    output[index] = (value < min_val) ? min_val : ((value > max_val) ? max_val : value);
}

torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val) {
    int elements = input.numel();

    auto output = torch::zeros_like(input);

    const int block_size = 256;
    dim3 grid((elements + block_size - 1) / block_size);
    dim3 block(block_size);

    hardtanh_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), min_val, max_val, elements);

    return output;
}
"""

hardtanh_cpp_source = (
    "torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val);"
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
        self.gemm = gemm
        self.group_norm = group_norm
        self.hardtanh = hardtanh

    def forward(self, x):
        x = self.gemm.gemm_cuda(x, self.weight)
        x = self.group_norm.group_norm_cuda(x, self.groups)
        x = self.hardtanh.hardtanh_cuda(x, self.min_val, self.max_val)
        return x


def get_inputs():
    batch_size = 1024
    in_features = 8192
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    in_features = 8192
    out_features = 8192
    num_groups = 16
    hardtanh_min = -2.0
    hardtanh_max = 2.0
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]