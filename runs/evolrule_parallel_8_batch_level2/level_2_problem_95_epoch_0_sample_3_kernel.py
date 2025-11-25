import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_activation_kernel(const float* x, const float* add_value, float* out, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features)
        return;
    
    int f = idx % out_features;
    
    float val = x[idx] + add_value[f];
    
    // Swish
    float sigmoid_val = 1.0f / (1.0f + expf(-val));
    val *= sigmoid_val;
    
    // Tanh
    val = tanhf(val);
    
    // GELU approximation
    float inner = sqrtf(2.0f / 3.14159265358979323846f) * (val + 0.044715f * val * val * val);
    val *= 0.5f * (1.0f + tanhf(inner));
    
    // Hardtanh
    if (val < -1.0f) val = -1.0f;
    else if (val > 1.0f) val = 1.0f;
    
    out[idx] = val;
}

torch::Tensor fused_activation_cuda(torch::Tensor x, torch::Tensor add_value) {
    int batch_size = x.size(0);
    int out_features = x.size(1);
    if (add_value.size(0) != out_features) {
        throw std::runtime_error("add_value dimension mismatch");
    }
    
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    int num_elements = batch_size * out_features;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    
    fused_activation_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        add_value.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        out_features
    );
    
    return out;
}
"""

fused_activation_cpp_source = """
extern "C" {
    torch::Tensor fused_activation_cuda(torch::Tensor x, torch::Tensor add_value);
}
"""

fused_activation = load_inline(
    name="fused_activation",
    cuda_sources=fused_activation_source,
    cpp_sources=fused_activation_cpp_source,
    functions=["fused_activation_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape).cuda())
        self.fused_activation = fused_activation  # The loaded CUDA module

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_activation.fused_activation_cuda(x, self.add_value)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]