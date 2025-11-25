import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

scale_clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void scale_clamp_kernel(const float* input, float scale, float min_val, float max_val, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] * scale;
        val = fmaxf(min_val, fminf(val, max_val));
        output[idx] = val;
    }
}

torch::Tensor scale_clamp_cuda(torch::Tensor input, float scale, float min_val, float max_val) {
    int64_t size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    scale_clamp_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), scale, min_val, max_val, output.data_ptr<float>(), size
    );

    return output;
}
"""

scale_clamp_cpp_source = "torch::Tensor scale_clamp_cuda(torch::Tensor input, float scale, float min_val, float max_val);"

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.gelu = nn.GELU()
        
        # Load the fused scale and clamp kernel
        self.scale_clamp = load_inline(
            name="scale_clamp",
            cpp_sources=scale_clamp_cpp_source,
            cuda_sources=scale_clamp_source,
            functions=["scale_clamp_cuda"],
            verbose=True,
        )

    def forward(self, x):
        x = self.gemm(x)
        x = self.scale_clamp.scale_clamp_cuda(
            x,
            self.scaling_factor,
            self.hardtanh_min,
            self.hardtanh_max,
        )
        x = self.gelu(x)
        return x

# Keep the same inputs for initialization and execution as original
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