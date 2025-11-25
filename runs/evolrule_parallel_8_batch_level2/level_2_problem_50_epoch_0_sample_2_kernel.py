import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for bias addition and scaling
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_kernel(
    const float* input, const float* bias, float scale,
    float* output,
    int batch_size, int channels, int D, int H, int W) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * D * H * W)
        return;

    // Compute channel index
    int c = (idx % (channels * D * H * W)) / (D * H * W);

    // Get bias value
    float b = bias[c]; // since bias is (channels,1,1,1), so bias[c] is the value for channel c

    // The actual value from input
    float val = input[idx];
    output[idx] = (val + b) * scale;
}

torch::Tensor fused_bias_scale_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    int total_size = batch_size * channels * D * H * W;

    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;

    fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.item<float>(),
        output.data_ptr<float>(),
        batch_size, channels, D, H, W
    );

    return output;
}
"""

fused_kernel_cpp = """
torch::Tensor fused_bias_scale_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale);
"""

# Compile the fused kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_bias_scale_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1, dtype=torch.float32))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        self.scale2 = nn.Parameter(torch.tensor(scale2, dtype=torch.float32))
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x * self.scale1
        x = self.avg_pool(x)
        x = self.fused_op.fused_bias_scale_cuda(x, self.bias, self.scale2)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape]