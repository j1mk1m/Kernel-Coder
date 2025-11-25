import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused Swish and scaling kernel
fused_swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_swish_scale_kernel(
    const float* input,
    float* output,
    float scaling_factor,
    int num_elements
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_elements) {
        float x = input[tid];
        float sig = 1.0f / (1.0f + expf(-x));
        output[tid] = x * sig * scaling_factor;
    }
}

torch::Tensor fused_swish_scale_cuda(torch::Tensor input, float scaling_factor) {
    auto num_elements = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;

    fused_swish_scale_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        num_elements
    );

    return output;
}
"""

fused_swish_scale_cpp_header = """
torch::Tensor fused_swish_scale_cuda(torch::Tensor input, float scaling_factor);
"""

# Compile the fused kernel
fused_swish_scale = load_inline(
    name="fused_swish_scale",
    cpp_sources=fused_swish_scale_cpp_header,
    cuda_sources=fused_swish_scale_source,
    functions=["fused_swish_scale_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.fused_swish_scale = fused_swish_scale  # Reference to the loaded module

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_swish_scale.fused_swish_scale_cuda(x, self.scaling_factor)
        return x

def get_inputs():
    batch_size = 128
    in_features = 32768
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    in_features = 32768
    out_features = 32768
    scaling_factor = 2.0
    return [in_features, out_features, scaling_factor]