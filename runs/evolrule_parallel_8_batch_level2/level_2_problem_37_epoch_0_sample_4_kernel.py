import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

swish_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_add_kernel(const float* x, const float* bias, float* out, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_features) {
        int c = idx % out_features;
        float val = x[idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        out[idx] = val * sigmoid_val + bias[c];
    }
}

torch::Tensor swish_add_cuda(torch::Tensor x, torch::Tensor bias) {
    int batch_size = x.size(0);
    int out_features = x.size(1);
    auto out = torch::empty_like(x);

    int num_elements = batch_size * out_features;
    const int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    swish_add_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), 
                                                 bias.data_ptr<float>(), 
                                                 out.data_ptr<float>(), 
                                                 batch_size, 
                                                 out_features);
    return out;
}
"""

swish_add_cpp_source = (
    "torch::Tensor swish_add_cuda(torch::Tensor x, torch::Tensor bias);"
)

# Compile the inline CUDA code
swish_add = load_inline(
    name="swish_add",
    cpp_sources=swish_add_cpp_source,
    cuda_sources=swish_add_source,
    functions=["swish_add_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.swish_add = swish_add  # Access the custom function

    def forward(self, x):
        x = self.matmul(x)
        x = self.swish_add.swish_add_cuda(x, self.bias)  # Call the custom kernel
        x = self.group_norm(x)
        return x

# Keep the original input and initialization functions
batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]