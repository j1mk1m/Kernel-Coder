import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for scalar multiplication and global average pooling
fused_multiply_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_multiply_mean_kernel(
    const float* input,
    float multiplier,
    float* output,
    int B, int C, int H, int W
) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    if (b >= B || c >= C) return;

    int tid = threadIdx.x;
    int spatial_size = H * W;
    float sum = 0.0f;

    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int h = i / W;
        int w = i % W;
        int in_idx = ((b * C + c) * H + h) * W + w;
        float val = input[in_idx] * multiplier;
        sum += val;
    }

    __shared__ float shared_sum[256];
    shared_sum[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = shared_sum[0] / (H * W);
        int out_idx = b * C + c;
        output[out_idx] = mean;
    }
}

torch::Tensor fused_multiply_mean_cuda(
    torch::Tensor input,
    float multiplier
) {
    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty({B, C, 1, 1}, input.options());

    const int threads = 256;
    dim3 blocks(B, C);
    dim3 threads_per_block(threads);

    fused_multiply_mean_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        multiplier,
        output.data_ptr<float>(),
        B, C, H, W
    );

    return output;
}
"""

fused_multiply_mean_header = """
torch::Tensor fused_multiply_mean_cuda(
    torch::Tensor input,
    float multiplier
);
"""

# Compile the fused CUDA kernel
fused_multiply_mean = load_inline(
    name="fused_multiply_mean",
    cuda_sources=fused_multiply_mean_source,
    cpp_sources=fused_multiply_mean_header,
    functions=["fused_multiply_mean_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.multiplier = multiplier
        self.fused_multiply_mean = fused_multiply_mean  # Reference to the CUDA module

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_multiply_mean.fused_multiply_mean_cuda(x, self.multiplier)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier]

# Constants as per the original architecture
batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier = 0.5