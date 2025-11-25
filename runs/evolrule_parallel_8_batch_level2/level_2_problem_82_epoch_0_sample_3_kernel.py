import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for tanh, scaling, and bias addition
fused_tanh_scale_bias_source = """
#include <torch/extension.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void fused_tanh_scale_bias_kernel(
    const float* input,
    float* output,
    const float scaling_factor,
    const float* bias,
    int N, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) return;

    // Compute n, c, h, w from linear index
    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (C * H * W);

    float x = input[idx];
    float tanh_x = tanh(x);
    float scaled = tanh_x * scaling_factor;
    float b_val = bias[c];  // bias is [C, 1, 1], so c is correct
    output[idx] = scaled + b_val;
}

torch::Tensor fused_tanh_scale_bias_cuda(
    torch::Tensor input,
    float scaling_factor,
    torch::Tensor bias
) {
    // Ensure input and bias are contiguous
    input = input.contiguous();
    bias = bias.contiguous();

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::empty_like(input);

    const int block_size = 256;
    int total_elements = N * C * H * W;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_tanh_scale_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        bias.data_ptr<float>(),
        N, C, H, W
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}
"""

fused_tanh_scale_bias_cpp = """
torch::Tensor fused_tanh_scale_bias_cuda(
    torch::Tensor input,
    float scaling_factor,
    torch::Tensor bias
);
"""

# Load the fused CUDA kernel
fused_tanh_scale_bias = load_inline(
    name="fused_tanh_scale_bias",
    cpp_sources=fused_tanh_scale_bias_cpp,
    cuda_sources=fused_tanh_scale_bias_source,
    functions=["fused_tanh_scale_bias_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-arch=sm_75"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        self.fused_tanh_scale_bias = fused_tanh_scale_bias  # Access the loaded kernel

    def forward(self, x):
        x = self.conv(x)
        # Apply fused kernel
        x = self.fused_tanh_scale_bias.fused_tanh_scale_bias_cuda(x, self.scaling_factor, self.bias)
        x = self.max_pool(x)
        return x