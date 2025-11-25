import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for Mish + Add + Hardtanh + Scale
fused_mish_add_hardtanh_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_mish_add_hardtanh_scale_kernel(
    const float* input, float* output,
    float add_val, float scale_val,
    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Compute Mish: x * tanh(softplus(x))
        float softplus = logf(1.f + expf(x));
        float mish = x * tanhf(softplus);
        // Apply add
        mish += add_val;
        // Apply Hardtanh: clamp between -1 and 1
        mish = min(max(mish, -1.f), 1.f);
        // Scale
        mish *= scale_val;
        output[idx] = mish;
    }
}

torch::Tensor fused_mish_add_hardtanh_scale_cuda(
    torch::Tensor input,
    float add_val,
    float scale_val) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_mish_add_hardtanh_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        add_val,
        scale_val,
        size
    );

    return output;
}
"""

# Compile the fused CUDA kernel
fused_ops = load_inline(
    name="fused_mish_add_hardtanh_scale",
    cuda_sources=fused_mish_add_hardtanh_scale_source,
    functions=["fused_mish_add_hardtanh_scale_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride,
            padding, output_padding
        )
        self.add_value = add_value
        self.scale = scale
        self.fused_ops = fused_ops.fused_mish_add_hardtanh_scale_cuda

    def forward(self, x):
        x = self.conv_transpose(x)
        # Apply fused operations
        x = self.fused_ops(x, self.add_value, self.scale)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]