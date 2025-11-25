import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_fused_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void fused_activation_kernel(const float* input, float* output, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        
        // Compute Swish: x * sigmoid(x)
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float swish = x * sigmoid_x;
        
        // Divide by 2
        float after_div = swish / 2.0f;
        
        // Clamp between -1 and 1
        float clamped1 = fminf(fmaxf(after_div, -1.0f), 1.0f);
        
        // Compute tanh
        float tanh_out = tanhf(clamped1);
        
        // Final clamp (as per original code)
        output[idx] = fminf(fmaxf(tanh_out, -1.0f), 1.0f);
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int64_t size = input.numel();
    
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    fused_activation_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size
    );
    
    return output;
}
"""

elementwise_fused_cpp_source = (
    "torch::Tensor fused_activation_cuda(torch::Tensor input);"
)

# Compile the fused activation CUDA code
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=elementwise_fused_cpp_source,
    cuda_sources=elementwise_fused_source,
    functions=["fused_activation_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.fused_activation = fused_activation  # The loaded module

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_activation.fused_activation_cuda(x)
        return x

# Keep the same get_inputs and get_init_inputs as the original
batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]