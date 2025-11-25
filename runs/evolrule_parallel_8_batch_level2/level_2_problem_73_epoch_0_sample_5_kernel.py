import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused batch norm and scaling
fused_batch_norm_and_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_batch_norm_and_scale_kernel(
    const float* input,
    const float* running_mean,
    const float* running_var,
    const float* weight,
    const float* bias,
    float scaling_factor,
    float* output,
    int N,
    int C,
    int H,
    int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (H * W)) % C;
    int n = idx / (C * H * W);

    float mean = running_mean[c];
    float var = running_var[c];
    float gamma = weight[c];
    float beta = bias[c];

    float eps = 1e-5;
    float denominator = sqrt(var + eps);

    float x_val = input[idx];
    float normalized = (x_val - mean) / denominator;
    float scaled = normalized * gamma + beta;
    float final = scaled * scaling_factor;

    output[idx] = final;
}

torch::Tensor fused_batch_norm_and_scale_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor
) {
    auto output = torch::empty_like(input);

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int size = N * C * H * W;

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    fused_batch_norm_and_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        output.data_ptr<float>(),
        N, C, H, W
    );

    return output;
}
"""

fused_batch_norm_and_scale_cpp = """
#include <torch/extension.h>

torch::Tensor fused_batch_norm_and_scale_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor
);
"""

# Compile the inline CUDA code
fused_batch_norm_and_scale = load_inline(
    name="fused_batch_norm_and_scale",
    cpp_sources=fused_batch_norm_and_scale_cpp,
    cuda_sources=fused_batch_norm_and_scale_source,
    functions=["fused_batch_norm_and_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.fused_batch_norm_and_scale = fused_batch_norm_and_scale  # Load the custom kernel

    def forward(self, x):
        x = self.conv(x)
        # Extract parameters from the batch norm layer
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        weight = self.bn.weight
        bias = self.bn.bias
        # Call the custom kernel
        x = self.fused_batch_norm_and_scale.fused_batch_norm_and_scale_cuda(
            x, running_mean, running_var, weight, bias, self.scaling_factor
        )
        return x

# Define the hyperparameters and input functions
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]