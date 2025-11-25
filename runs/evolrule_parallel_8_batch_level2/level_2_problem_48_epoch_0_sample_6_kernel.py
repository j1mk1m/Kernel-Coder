import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for element-wise operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_operations_kernel(
    const float* input, const float* scaling_factor, const float* bias, float* output,
    int B, int C, int D, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * D * H * W) return;

    // Compute indices
    int batch = idx / (C * D * H * W);
    int remainder = idx % (C * D * H * W);
    int c = remainder / (D * H * W);
    remainder %= (D * H * W);
    int d = remainder / (H * W);
    remainder %= (H * W);
    int h = remainder / W;
    int w = remainder % W;

    // Get parameters for current channel
    float sf = scaling_factor[c];
    float b_val = bias[c];

    // Compute operations
    float val = input[idx];
    val *= sf;
    val = tanhf(val);  // tanh
    val *= b_val;
    val = 1.0f / (1.0f + expf(-val));  // sigmoid

    output[idx] = val;
}

torch::Tensor fused_operations_cuda(torch::Tensor input, 
                                   torch::Tensor scaling_factor,
                                   torch::Tensor bias) {
    auto B = input.size(0);
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);

    auto output = torch::empty_like(input);

    const int block_size = 256;
    int num_elements = B * C * D * H * W;
    dim3 blocks((num_elements + block_size - 1) / block_size);
    dim3 threads(block_size);

    fused_operations_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        scaling_factor.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W
    );

    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_operations_cuda(torch::Tensor input, 
                                   torch::Tensor scaling_factor,
                                   torch::Tensor bias);
"""

# Compile the fused CUDA kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_operations_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_operations_cuda(x, self.scaling_factor, self.bias)
        return x