import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication followed by dropout and softmax
matmul_dropout_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand.h>

__global__ void matmul_dropout_softmax_kernel(const float* a, const float* b, float* c, int M, int N, int K, float p, unsigned long seed) {
    curandState state;
    curand_init(seed, 0, 0, &state);

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum * ((curand_uniform(&state) > p) ? 1.0f : 0.0f);
    }
}

__global__ void softmax_kernel(const float* input, float* output, int batch_size, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * features) {
        int feature = idx % features;
        float max_val = input[idx];
        for (int i = 0; i < batch_size; ++i) {
            float val = input[i * features + feature];
            if (val > max_val) {
                max_val = val;
            }
        }

        float sum = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            sum += exp(input[i * features + feature] - max_val);
        }

        output[idx] = exp(input[idx] - max_val) / sum;
    }
}

torch::Tensor matmul_dropout_softmax_cuda(torch::Tensor a, torch::Tensor b, float p, unsigned long seed) {
    auto M = a.size(0);
    auto N = b.size(1);
    auto K = a.size(1);
    auto c = torch::zeros({M, N}, a.options());

    const int block_size = 16;
    const int num_blocks_x = (N + block_size - 1) / block_size;
    const int num_blocks_y = (M + block_size - 1) / block_size;

    matmul_dropout_softmax_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K, p, seed);

    return c;
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto features = input.size(1);
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (batch_size * features + block_size - 1) / block_size;

    softmax_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, features);

    return output;
}
"""

matmul_dropout_softmax_cpp_source = (
    "torch::Tensor matmul_dropout_softmax_cuda(torch::Tensor a, torch::Tensor b, float p, unsigned long seed);"
    "torch::Tensor softmax_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for matrix multiplication followed by dropout and softmax
matmul_dropout_softmax = load_inline(
    name="matmul_dropout_softmax",
    cpp_sources=matmul_dropout_softmax_cpp_source,
    cuda_sources=matmul_dropout_softmax_source,
    functions=["matmul_dropout_softmax_cuda", "softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    A model that performs matrix multiplication using a custom CUDA kernel, applies dropout, and then applies custom softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout_p = dropout_p

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = matmul_dropout_softmax.matmul_dropout_softmax_cuda(x, self.matmul.weight, self.dropout_p, torch.randint(0, 2**32 - 1, (1,)).item())
        x = matmul_dropout_softmax.softmax_cuda(x)
        return x