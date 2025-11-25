import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for matrix multiplication followed by Swish activation and scaling
__global__ void matmul_swish_scale_kernel(const float* a, const float* b, float* c, int m, int n, int k, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }

        // Apply Swish activation
        float swish = sum * (1.0f + tanh(sum));

        // Scale the result
        c[row * n + col] = swish * scale;
    }
}

torch::Tensor matmul_swish_scale_cuda(torch::Tensor a, torch::Tensor b, float scale) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);
    auto out = torch::zeros({m, n}, a.options());

    const int block_size_x = 32;
    const int block_size_y = 32;
    const int num_blocks_x = (n + block_size_x - 1) / block_size_x;
    const int num_blocks_y = (m + block_size_y - 1) / block_size_y;

    matmul_swish_scale_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size_x, block_size_y)>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), m, n, k, scale);

    return out;
}
"""

matmul_swish_scale_cpp_source = (
    "torch::Tensor matmul_swish_scale_cuda(torch::Tensor a, torch::Tensor b, float scale);"
)

# Compile the inline CUDA code for matrix multiplication followed by Swish activation and scaling
matmul_swish_scale = load_inline(
    name="matmul_swish_scale",
    cpp_sources=matmul_swish_scale_cpp_source,
    cuda_sources=matmul_swish_scale_source,
    functions=["matmul_swish_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul_swish_scale = matmul_swish_scale

    def forward(self, x):
        return self.matmul_swish_scale.matmul_swish_scale_cuda(x, x.t(), self.scaling_factor)


def get_inputs():
    batch_size = 128
    in_features = 32768
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    in_features = 32768
    out_features = 32768
    scaling_factor = 2.0
    return [in_features, out_features, scaling_factor]