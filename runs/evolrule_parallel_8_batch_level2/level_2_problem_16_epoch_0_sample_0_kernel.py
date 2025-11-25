import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused post-processing CUDA kernel
fused_postprocess_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_postprocess_kernel(
    const float* input,
    float* output,
    int size,
    float add_value,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float abs_x = fabs(x);
        float softplus;
        if (x > 20.0f) {
            softplus = x;
        } else if (x < -20.0f) {
            softplus = 0.0f;
        } else {
            if (x >= 0.0f) {
                softplus = x + logf(1.0f + expf(-x));
            } else {
                softplus = logf(1.0f + expf(x));
            }
        }
        float tanh_softplus = tanhf(softplus);
        float mish = x * tanh_softplus;
        mish += add_value;
        if (mish < -1.0f) mish = -1.0f;
        else if (mish > 1.0f) mish = 1.0f;
        mish *= scale;
        output[idx] = mish;
    }
}

torch::Tensor fused_postprocess_cuda(torch::Tensor input, float add_value, float scale) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_postprocess_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        add_value,
        scale
    );

    return output;
}
"""

fused_postprocess_header = """
torch::Tensor fused_postprocess_cuda(torch::Tensor input, float add_value, float scale);
"""

# Compile the fused post-processing kernel
fused_postprocess = load_inline(
    name="fused_postprocess",
    cpp_sources=fused_postprocess_header,
    cuda_sources=fused_postprocess_source,
    functions=["fused_postprocess_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.add_value = add_value
        self.scale = scale
        self.fused_postprocess = fused_postprocess  # Load the fused CUDA kernel

    def forward(self, x):
        x = self.conv_transpose(x)
        # Apply fused post-processing kernel
        return self.fused_postprocess.fused_postprocess_cuda(
            x, self.add_value, self.scale
        )

# The get_inputs and get_init_inputs functions remain unchanged as per the original code
batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 128  
kernel_size  = 3
stride       = 2  
padding      = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]