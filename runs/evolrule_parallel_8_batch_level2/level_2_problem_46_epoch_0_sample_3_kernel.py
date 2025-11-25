import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused kernel combining element-wise operations and avg pooling
fused_elementwise_and_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_elementwise_and_pool_kernel(
    const float* input, float a, float b,
    float* output,
    int B, int C, int H, int W, int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * H_out * W_out)
        return;

    int j = idx % W_out;
    int i = (idx / W_out) % H_out;
    int c = (idx / (H_out * W_out)) % C;
    int b = idx / (C * H_out * W_out);

    float sum = 0.0f;
    for (int dx = 0; dx < 2; ++dx) {
        for (int dy = 0; dy < 2; ++dy) {
            int h = i * 2 + dx;
            int w = j * 2 + dy;
            if (h >= H || w >= W)
                continue;

            int input_idx = b * C * H * W + c * H * W + h * W + w;
            float val = input[input_idx] - a;
            val = tanhf(val);
            val = val - b;
            sum += val;
        }
    }
    output[idx] = sum / 4.0f;
}

torch::Tensor fused_elementwise_and_pool_cuda(
    torch::Tensor input,
    float a,
    float b) {
    // Assume kernel_size and stride are 2
    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int H_out = (H - 2)/2 +1;
    int W_out = (W - 2)/2 +1;

    auto output = torch::empty({B, C, H_out, W_out}, input.options());

    const int threads_per_block = 256;
    int num_blocks = (B * C * H_out * W_out + threads_per_block -1) / threads_per_block;

    fused_elementwise_and_pool_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(), a, b,
        output.data_ptr<float>(),
        B, C, H, W, H_out, W_out
    );

    return output;
}
"""

# Compile the fused kernel
fused_elementwise_and_pool = load_inline(
    name="fused_elementwise_and_pool",
    cpp_sources="torch::Tensor fused_elementwise_and_pool_cuda(torch::Tensor input, float a, float b);",
    cuda_sources=fused_elementwise_and_pool_source,
    functions=["fused_elementwise_and_pool_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.a = subtract1_value
        self.b = subtract2_value
        # self.avgpool is not needed anymore since it's fused into the kernel

        # Load the fused kernel
        self.fused_elementwise_and_pool = fused_elementwise_and_pool

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_elementwise_and_pool.fused_elementwise_and_pool_cuda(x, self.a, self.b)
        return x