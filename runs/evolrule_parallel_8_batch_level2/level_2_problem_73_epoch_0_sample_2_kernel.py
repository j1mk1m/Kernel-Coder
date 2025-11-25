import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

        # Define the fused batch norm and scaling kernel
        fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_batch_norm_scale_kernel(
    const float* input,
    float* output,
    const float* running_mean,
    const float* running_var,
    const float* gamma,
    const float* beta,
    float scaling_factor,
    float eps,
    int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) return;

    int c = (idx / (H * W)) % C;
    float inv_std = 1.0f / sqrtf(running_var[c] + eps);
    float scaled_gamma = gamma[c] * inv_std;
    float bias_term = -running_mean[c] * inv_std * gamma[c] + beta[c];
    float val = input[idx] * scaled_gamma + bias_term;
    val *= scaling_factor;
    output[idx] = val;
}

torch::Tensor fused_batch_norm_scale_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta,
    float scaling_factor,
    float eps,
    int N, int C, int H, int W) {
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_elements = N * C * H * W;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_batch_norm_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        scaling_factor,
        eps,
        N, C, H, W
    );

    return output;
}
"""

        cpp_source = """
torch::Tensor fused_batch_norm_scale_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta,
    float scaling_factor,
    float eps,
    int N, int C, int H, int W
);
"""

        # Compile the fused kernel
        self.fused_bn_scale = load_inline(
            name="fused_bn_scale",
            cpp_sources=cpp_source,
            cuda_sources=fused_source,
            functions=["fused_batch_norm_scale_cuda"],
            verbose=True
        )

    def forward(self, x):
        x = self.conv(x).contiguous()
        N, C, H, W = x.size()
        x = self.fused_bn_scale.fused_batch_norm_scale_cuda(
            x,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.weight,
            self.bn.bias,
            self.scaling_factor,
            self.bn.eps,
            N, C, H, W
        )
        return x

# The get_inputs and get_init_inputs remain the same as original
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