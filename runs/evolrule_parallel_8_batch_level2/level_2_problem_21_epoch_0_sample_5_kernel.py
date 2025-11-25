import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for bias add, scaling, and sigmoid
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_operations_kernel(
    const float* input,
    const float* bias,
    const float* scale,
    float* output,
    int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) return;

    int c = (idx / (H * W)) % C;
    int spatial_idx = idx % (H * W);

    float x = input[idx];
    float b = bias[c];
    float s = scale[c];
    float scaled = (x + b) * s;
    float exp_val = expf(-scaled);
    output[idx] = 1.0f / (1.0f + exp_val);
}

torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale) {
    auto input_ = input.contiguous();
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    // Reshape bias and scale to 1D
    auto bias_ = bias.view({C});
    auto scale_ = scale.view({C});

    auto output = torch::empty_like(input_);

    const int block_size = 256;
    const int total_elements = N * C * H * W;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_operations_kernel<<<num_blocks, block_size>>>(
        input_.data_ptr<float>(),
        bias_.data_ptr<float>(),
        scale_.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W
    );

    return output;
}
"""

fused_ops_header = """
torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale);
"""

# Compile the fused operations kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_header,
    cuda_sources=fused_ops_source,
    functions=["fused_operations_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)  # includes bias
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.fused_ops = fused_ops  # Access to the fused CUDA function

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_operations_cuda(x, self.bias, self.scale)
        x = self.group_norm(x)
        return x