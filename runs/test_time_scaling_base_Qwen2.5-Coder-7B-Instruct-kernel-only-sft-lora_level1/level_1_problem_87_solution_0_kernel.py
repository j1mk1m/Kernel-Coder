import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for pointwise 2D convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Helper function to perform matrix multiplication
__device__ float matmul(float* A, float* B, int M, int N, int K) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[M * k + n] * B[K * n + k];
    }
    return sum;
}

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int height, int width, int out_channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_channels * height * width) {
        int b = idx / (out_channels * height * width);
        int o = (idx % (out_channels * height * width)) / (height * width);
        int h = ((idx % (out_channels * height * width)) % (height * width)) / width;
        int w = ((idx % (out_channels * height * width)) % (height * width)) % width;

        float sum = 0.0f;
        for (int c = 0; c < in_channels; ++c) {
            sum += matmul(&input[b * in_channels * height * width + c * height * width + h * width + w], &weight[o * in_channels * kernel_size * kernel_size + c * kernel_size * kernel_size], kernel_size, kernel_size, in_channels);
        }
        output[b * out_channels * height * width + o * height * width + h * width + w] = sum;
    }
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int batch_size, int in_channels, int height, int width, int out_channels) {
    auto output = torch::zeros({batch_size, out_channels, height, width});

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * height * width + block_size - 1) / block_size;

    convolution_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, height, width, out_channels);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int batch_size, int in_channels, int height, int width, int out_channels);"
)

# Compile the inline CUDA code for pointwise 2D convolution
convolution = load_inline(
    name="convolution",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.size()
        return convolution.convolution_cuda(x, self.weight, batch_size, in_channels, height, width, out_channels) + (self.bias.view(1, -1, 1, 1) if self.bias is not None else 0)