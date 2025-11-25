import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Load the custom CUDA kernels
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
    auto M = A.size(0);
    auto N = B.size(1);
    auto K = A.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scale_kernel(const float* input, float* output, float factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * factor;
    }
}

torch::Tensor scale_cuda(torch::Tensor input, float factor) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    scale_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), factor, size);

    return output;
}
"""

hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float* input, float* output, float min_val, float max_val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > max_val ? max_val : (input[idx] < min_val ? min_val : input[idx]);
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardtanh_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), min_val, max_val, size);

    return output;
}
"""

gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715 * x * x * x)));
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

# Compile the custom CUDA kernels
gemm = load_inline(
    name="gemm",
    cpp_sources="torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B);",
    cuda_sources=gemm_source,
    functions=["gemm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

scale = load_inline(
    name="scale",
    cpp_sources="torch::Tensor scale_cuda(torch::Tensor input, float factor);",
    cuda_sources=scale_source,
    functions=["scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

hardtanh = load_inline(
    name="hardtanh",
    cpp_sources="torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val);",
    cuda_sources=hardtanh_source,
    functions=["hardtanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

gelu = load_inline(
    name="gelu",
    cpp_sources="torch::Tensor gelu_cuda(torch::Tensor input);",
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        x = gemm.gemm_cuda(x, torch.randn(self.in_features, self.out_features).cuda())
        x = scale.scale_cuda(x, self.scaling_factor)
        x = hardtanh.hardtanh_cuda(x, self.hardtanh_min, self.hardtanh_max)
        x = gelu.gelu_cuda(x)
        return x


# Example usage
model_new = ModelNew(in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max)
inputs = get_inputs()
output = model_new(inputs[0])
print(output.shape)