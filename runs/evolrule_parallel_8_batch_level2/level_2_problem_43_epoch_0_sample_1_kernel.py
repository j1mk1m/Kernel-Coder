import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for logsumexp and ReLU
fused_logsumexp_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_logsumexp_relu_kernel(
    const float* input, float* output,
    int batch_size, int channels,
    int depth, int height, int width
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * depth * height * width) return;

    // Compute indices for batch, depth, height, width
    int b = tid / (depth * height * width);
    int rem = tid % (depth * height * width);
    int d = rem / (height * width);
    rem %= (height * width);
    int h = rem / width;
    int w = rem % width;

    // Calculate the output index
    int out_idx = b * depth * height * width + d * height * width + h * width + w;

    float sum = 0.0f;
    for (int c = 0; c < channels; ++c) {
        // Input index calculation: input[b][c][d][h][w]
        int in_idx = b * channels * depth * height * width
                   + c * depth * height * width
                   + d * height * width
                   + h * width
                   + w;
        float val = input[in_idx];
        sum += expf(val);
    }

    float log_sum = logf(sum);
    float result = (log_sum > 0.0f) ? log_sum : 0.0f;
    output[out_idx] = result;
}

torch::Tensor fused_logsumexp_relu_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    auto output = torch::zeros({batch_size, 1, depth, height, width}, input.options());

    int total_elements = batch_size * depth * height * width;
    const int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_logsumexp_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, depth, height, width
    );

    return output;
}
"""

fused_logsumexp_relu_cpp = (
    "torch::Tensor fused_logsumexp_relu_cuda(torch::Tensor input);"
)

# Compile the fused CUDA kernel
fused_logsumexp_relu = load_inline(
    name="fused_logsumexp_relu",
    cpp_sources=[fused_logsumexp_relu_cpp],
    cuda_sources=[fused_logsumexp_relu_source],
    functions=["fused_logsumexp_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fused_logsumexp_relu = fused_logsumexp_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.fused_logsumexp_relu.fused_logsumexp_relu_cuda(x)
        return x

batch_size = 4
in_channels = 32
out_channels = 64
depth, height, width = 32, 128, 128
kernel_size = 3
stride = 1
padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]