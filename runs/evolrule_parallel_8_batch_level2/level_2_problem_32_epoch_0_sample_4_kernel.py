import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for scaling and min reduction
kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>  // for FLT_MAX

__global__ void channel_min_scaled_kernel(const float* input, float* output, int B, int C, int H, int W, float scale) {
    int idx = blockIdx.x;
    int b = idx / (H * W);
    int rem = idx % (H * W);
    int h = rem / W;
    int w = rem % W;

    int c = threadIdx.x;
    if (c >= C) return;

    float val = input[b * C * H * W + c * H * W + h * W + w] * scale;

    extern __shared__ float shared[];
    shared[threadIdx.x] = (threadIdx.x < C) ? val : FLT_MAX;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] = min(shared[threadIdx.x], shared[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[b * H * W + h * W + w] = shared[0];
    }
}

torch::Tensor channel_min_scaled_cuda(torch::Tensor input, float scale) {
    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::empty({B, 1, H, W}, torch::device(input.device()).dtype(input.dtype()));

    int num_blocks = B * H * W;
    int block_size = C;

    channel_min_scaled_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W,
        scale
    );

    return output;
}
"""

kernel_cpp = (
    "torch::Tensor channel_min_scaled_cuda(torch::Tensor input, float scale);"
)

# Compile the inline CUDA code
channel_min_scaled = load_inline(
    name="channel_min_scaled",
    cpp_sources=kernel_cpp,
    cuda_sources=kernel_source,
    functions=["channel_min_scaled_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        x = channel_min_scaled.channel_min_scaled_cuda(x, self.scale_factor)
        return x

# The following functions remain unchanged from the original
batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]