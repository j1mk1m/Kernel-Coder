import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for Mish, add, Hardtanh, and scaling
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void fused_operations_kernel(
    const float* input, 
    float* output,
    const float add_val,
    const float scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        
        // Compute Mish activation
        float exp_x = expf(x);
        float softplus = logf(1.0f + exp_x);
        float tanh_softplus = tanhf(softplus);
        float mish = x * tanh_softplus;

        // Apply add_value
        mish += add_val;

        // Apply Hardtanh (-1 to 1)
        if (mish < -1.0f) {
            mish = -1.0f;
        } else if (mish > 1.0f) {
            mish = 1.0f;
        }

        // Apply scaling
        mish *= scale;

        output[idx] = mish;
    }
}

torch::Tensor fused_operations_cuda(torch::Tensor input, float add_val, float scale) {
    auto output = torch::empty_like(input);
    const int size = input.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_operations_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), add_val, scale, size
    );

    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_operations_cuda(torch::Tensor input, float add_val, float scale);
"""

# Compile the fused CUDA kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_operations_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.add_value = add_value
        self.scale = scale
        self.fused_ops = fused_ops  # Load the fused CUDA operations

    def forward(self, x):
        x = self.conv_transpose(x)
        return self.fused_ops.fused_operations_cuda(x, self.add_value, self.scale)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]