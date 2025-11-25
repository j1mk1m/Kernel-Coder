import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for bias addition, scaling, and sigmoid
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_operations_kernel(
    const float* input, const float* bias, const float* scale, float* output,
    int N, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) return;

    int c = (idx / (H * W)) % C;
    int h = (idx / W) % H;
    int w = idx % W;
    int n = idx / (C * H * W);

    float val = input[idx];
    val += bias[c];
    val *= scale[c];
    output[idx] = 1.0f / (1.0f + expf(-val));
}

torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale) {
    auto device = input.device();
    bias = bias.to(device);
    scale = scale.to(device);

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::empty_like(input);

    const int block_size = 256;
    int size = N * C * H * W;
    int num_blocks = (size + block_size - 1) / block_size;

    fused_operations_kernel<<<num_blocks, block_size, 0, torch::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), scale.data_ptr<float>(),
        output.data_ptr<float>(), N, C, H, W
    );

    return output;
}
"""

fused_ops_cpp_header = """
#include <torch/extension.h>
torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale);
"""

# Compile the fused operations kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_header,
    cuda_sources=fused_ops_source,
    functions=["fused_operations_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_operations_cuda(x, self.bias, self.scale)
        x = self.group_norm(x)
        return x