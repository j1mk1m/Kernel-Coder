import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for element-wise operations after convolution
fused_elementwise_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <math.h>

__global__ void fused_elementwise_kernel(
    const float* input, const float* scaling, const float* bias, float* output,
    int B, int C, int D, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * D * H * W) return;

    // Compute indices
    int w = idx % W;
    int h = (idx / W) % H;
    int d = (idx / (W * H)) % D;
    int c = (idx / (W * H * D)) % C;
    int b = idx / (C * D * H * W);

    float val = input[idx] * scaling[c];
    val = tanhf(val);
    val *= bias[c];
    val = 1.0f / (1.0f + expf(-val));
    output[idx] = val;
}

torch::Tensor fused_elementwise_cuda(torch::Tensor input,
                                    torch::Tensor scaling,
                                    torch::Tensor bias) {
    // Check dimensions (simplified for brevity)
    auto B = input.size(0);
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);

    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    const int total_elements = B * C * D * H * W;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    fused_elementwise_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        scaling.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

fused_elementwise_cpp_source = (
    "torch::Tensor fused_elementwise_cuda(torch::Tensor input, torch::Tensor scaling, torch::Tensor bias);"
)

# Compile the fused CUDA kernel
fused_elementwise = load_inline(
    name="fused_elementwise",
    cuda_sources=fused_elementwise_source,
    cpp_sources=fused_elementwise_cpp_source,
    functions=["fused_elementwise_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_elementwise = fused_elementwise  # Loaded CUDA module

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_elementwise.fused_elementwise_cuda(x, self.scaling_factor, self.bias)
        return x

def get_inputs():
    batch_size = 128
    in_channels = 3
    depth, height, width = 16, 64, 64
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    scaling_factor = 2  # Unused in current Model's __init__ (assumed parameter mistake)
    bias_shape = (out_channels, 1, 1, 1)
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]