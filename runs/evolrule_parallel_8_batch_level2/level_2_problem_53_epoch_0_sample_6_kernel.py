import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for scaling, hardtanh, and GELU
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_ops_kernel(const float* input, float scaling_factor, 
                                float hardtanh_min, float hardtanh_max, 
                                float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float y = input[idx];
        float x_scaled = y * scaling_factor;
        float x_clamped = fmaxf(fminf(x_scaled, hardtanh_max), hardtanh_min);
        
        const float sqrt_2_over_pi = 0.7978845608f;
        const float poly_coeff = 0.044715f;
        float x_cubed = x_clamped * x_clamped * x_clamped;
        float inner = sqrt_2_over_pi * (x_clamped + poly_coeff * x_cubed);
        float tanh_inner = tanhf(inner);
        
        output[idx] = 0.5f * x_clamped * (1.0f + tanh_inner);
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, float scaling_factor, 
                             float hardtanh_min, float hardtanh_max) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_ops_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), scaling_factor, hardtanh_min, 
        hardtanh_max, output.data_ptr<float>(), size
    );
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, float scaling_factor, "
    "float hardtanh_min, float hardtanh_max);"
)

# Compile the fused CUDA operator
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, 
                 hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.fused_ops = fused_ops  # Load the fused CUDA operator

    def forward(self, x):
        x = self.gemm(x)
        return self.fused_ops.fused_ops_cuda(
            x, self.scaling_factor, self.hardtanh_min, self.hardtanh_max
        )

# Keep these functions as in the original code
batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]