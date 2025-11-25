import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_operations_kernel(
    const float* input, const float* scaling_factor, const float* bias, float* output,
    int batch_size, int channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * depth * height * width) return;

    // Decompose linear index into 5D coordinates
    int w = idx % width;
    int rem = idx / width;
    int h = rem % height;
    rem = rem / height;
    int d = rem % depth;
    rem = rem / depth;
    int c = rem % channels;
    int n = rem / channels;

    // Compute the value using 5D indexing
    float val = input[ n * channels * depth * height * width +
                      c * depth * height * width +
                      d * height * width +
                      h * width +
                      w ];

    // Apply scaling
    val *= scaling_factor[c];
    // Apply tanh
    val = tanhf(val);
    // Multiply by bias
    val *= bias[c];
    // Apply sigmoid
    val = 1.0f / (1.0f + expf(-val));

    // Write result to output
    output[idx] = val;
}

torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor scaling_factor, torch::Tensor bias) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    auto output = torch::empty_like(input);

    int total_elements = batch_size * channels * depth * height * width;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    fused_operations_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        scaling_factor.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, depth, height, width
    );

    return output;
}
"""

fused_ops_cpp = "torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor scaling_factor, torch::Tensor bias);"

# Compile the fused operations kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp,
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

    def forward(self, x):
        x = self.conv(x)
        # Apply fused operations using the custom CUDA kernel
        x = fused_ops.fused_operations_cuda(x, self.scaling_factor, self.bias)
        return x