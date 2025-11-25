import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_ops_kernel(
    const float* input,
    const float* multiplier,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int n = idx / (width * height * channels);

    int input_offset = n * channels * height * width + c * height * width + h * width + w;
    float val = input[input_offset];

    float scale = multiplier[c];
    val *= scale;

    float slope = 0.01f;
    val = (val > 0) ? val : val * slope;

    float x = val;
    float sqrt_2 = sqrt(2.0f);
    float erf_val = erf(x / sqrt_2);
    float gelu = 0.5f * x * (1.0f + erf_val);
    output[input_offset] = gelu;
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor multiplier) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto output = torch::empty_like(input);

    int num_elements = batch_size * channels * height * width;
    const int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_ops_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width
    );

    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor multiplier);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.fused_ops = fused_ops  # Import the fused kernel

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_ops_cuda(x, self.multiplier)
        return x