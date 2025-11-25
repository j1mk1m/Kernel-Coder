import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Gemm
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

    const int block_size = 16;
    const int num_blocks_x = (N + block_size - 1) / block_size;
    const int num_blocks_y = (M + block_size - 1) / block_size;

    gemm_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

gemm_cpp_source = (
    "torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B);"
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


# Define the custom CUDA kernel for LogSumExp
logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void logsumexp_kernel(const float* input, float* output, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        sdata[tid] = input[i];
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicAdd(output, sdata[0]);
        }
    }
}

torch::Tensor logsumexp_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros(1, input.options());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    logsumexp_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

logsumexp_cpp_source = (
    "torch::Tensor logsumexp_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for LogSumExp
logsumexp = load_inline(
    name="logsumexp",
    cpp_sources=logsumexp_cpp_source,
    cuda_sources=logsumexp_source,
    functions=["logsumexp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for LeakyReLU
leakyrelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leakyrelu_kernel(const float* input, float* output, int size, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = input[idx] > 0 ? input[idx] : input[idx] * negative_slope;
    }
}

torch::Tensor leakyrelu_cuda(torch::Tensor input, float negative_slope) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    leakyrelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size, negative_slope);

    return output;
}
"""

leakyrelu_cpp_source = (
    "torch::Tensor leakyrelu_cuda(torch::Tensor input, float negative_slope);"
)

# Compile the inline CUDA code for LeakyReLU
leakyrelu = load_inline(
    name="leakyrelu",
    cpp_sources=leakyrelu_cpp_source,
    cuda_sources=leakyrelu_source,
    functions=["leakyrelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for GELU
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float x = input[idx];
        float y = 0.5 * (x + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
        output[idx] = y;
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

gelu_cpp_source = (
    "torch::Tensor gelu_cuda(torch::Tensor input);"
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


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.gemm = gemm
        self.logsumexp = logsumexp
        self.leakyrelu = leakyrelu
        self.gelu = gelu

    def forward(self, x):
        # Gemm
        x = self.linear(x)
        x = self.gemm.gemm_cuda(x, x.t())
        # LogSumExp
        x = self.logsumexp.logsumexp_cuda(x)
        # LeakyReLU
        x = self.leakyrelu.leakyrelu_cuda(x, negative_slope=0.01)
        # LeakyReLU
        x = self.leakyrelu.leakyrelu_cuda(x, negative_slope=0.01)
        # GELU
        x = self.gelu.gelu_cuda(x)
        # GELU
        x = self.gelu.gelu_cuda(x)
        return x


def get_inputs():
    batch_size = 1024
    in_features = 8192
    out_features = 8192
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]