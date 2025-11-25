import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication with GELU
matmul_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

__global__ void matmul_gelu_kernel(const float* A, const float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) return;

    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        sum += A[row * k + i] * B[i * n + col];
    }

    C[row * n + col] = gelu(sum);
}

torch::Tensor matmul_gelu_cuda(torch::Tensor A, torch::Tensor B) {
    auto m = A.size(0);
    auto n = B.size(1);
    auto k = A.size(1);

    auto C = torch::zeros({m, n}, A.options());

    const int block_size = 32;
    const int num_blocks_x = (n + block_size - 1) / block_size;
    const int num_blocks_y = (m + block_size - 1) / block_size;

    matmul_gelu_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), m, n, k);

    return C;
}
"""

matmul_gelu_cpp_source = (
    "torch::Tensor matmul_gelu_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication with GELU
matmul_gelu = load_inline(
    name="matmul_gelu",
    cpp_sources=matmul_gelu_cpp_source,
    cuda_sources=matmul_gelu_source,
    functions=["matmul_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int batch_size, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * seq_len) return;

    int b = idx / seq_len;
    int s = idx % seq_len;

    float max_val = -INFINITY;
    for (int j = 0; j < seq_len; ++j) {
        max_val = fmax(max_val, input[b * seq_len + j]);
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        sum_exp += exp(input[b * seq_len + j] - max_val);
    }

    output[idx] = exp(input[b * seq_len + s] - max_val) / sum_exp;
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);

    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (batch_size * seq_len + block_size - 1) / block_size;

    softmax_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, seq_len);

    return output;
}
"""

softmax_cpp_source = (
    "torch::Tensor softmax_cuda(torch::Tensor input);"
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
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = matmul_gelu.matmul_gelu_cuda(x, torch.eye(x.size(1)).cuda())
        x = softmax.softmax_cuda(x)
        return x


batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]

model_new = ModelNew(in_features, out_features)
input_data = get_inputs()[0].to('cuda')
output = model_new(input_data.to('cuda'))
print(output.shape)