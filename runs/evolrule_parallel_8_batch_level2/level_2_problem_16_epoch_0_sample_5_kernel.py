import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for Mish + add + Hardtanh + scaling
fused_ops_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void fused_ops_kernel(const float* input, float* output, int size, float add_val, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Compute Mish activation: x * tanh(ln(1 + exp(x)))
        float exp_x = expf(x);
        float softplus = logf(1.0f + exp_x);
        float tanh_soft = tanhf(softplus);
        float mish = x * tanh_soft;
        // Add the add_val
        mish += add_val;
        // Apply Hardtanh clamp between -1 and 1
        if (mish < -1.0f) {
            mish = -1.0f;
        } else if (mish > 1.0f) {
            mish = 1.0f;
        }
        // Multiply by scale
        mish *= scale;
        output[idx] = mish;
    }
}

torch::Tensor fused_elementwise_cuda(torch::Tensor input, float add_val, float scale) {
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (input.numel() + block_size - 1) / block_size;
    fused_ops_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), input.numel(), add_val, scale);
    return output;
}
"""

# Header for the C++ code
fused_ops_h_source = """
torch::Tensor fused_elementwise_cuda(torch::Tensor input, float add_val, float scale);
"""

# Load the fused CUDA operator
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_h_source,
    cuda_sources=fused_ops_source,
    functions=["fused_elementwise_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size, stride,
            padding, output_padding
        )
        self.add_value = add_value
        self.scale = scale
        self.fused_elementwise = fused_ops  # Reference to fused CUDA operator

    def forward(self, x):
        x = self.conv_transpose(x)
        # Apply fused element-wise operations in a single kernel call
        x = self.fused_elementwise.fused_elementwise_cuda(x, self.add_value, self.scale)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]