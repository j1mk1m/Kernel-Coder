import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Custom CUDA kernel for fused MatMul + Swish + Scaling
matmul_swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_matmul_swish_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float scaling_factor,
    int batch_size,
    int in_features,
    int out_features) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += input[row * in_features + k] * weight[k * out_features + col];
        }
        // Compute Swish activation: x * sigmoid(x)
        sum = sum / (1.0f + expf(-sum));
        // Apply scaling
        output[row * out_features + col] = sum * scaling_factor;
    }
}

torch::Tensor fused_matmul_swish_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    float scaling_factor,
    int batch_size,
    int in_features,
    int out_features) 
{
    const int threads_per_block = 32;
    dim3 blocks(
        (batch_size + threads_per_block - 1) / threads_per_block,
        (out_features + threads_per_block - 1) / threads_per_block
    );
    dim3 threads(threads_per_block, threads_per_block);

    auto output = torch::empty({batch_size, out_features}, 
        torch::dtype(input.dtype()).device(input.device()));

    fused_matmul_swish_scale_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

matmul_swish_scale_cpp_source = (
    "torch::Tensor fused_matmul_swish_scale_cuda(torch::Tensor input, torch::Tensor weight, float scaling_factor, int batch_size, int in_features, int out_features);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_matmul_swish_scale",
    sources=[],
    extra_cuda_cflags=['-lineinfo'],
    cpp_sources=matmul_swish_scale_cpp_source,
    cuda_sources=matmul_swish_scale_source,
    functions=["fused_matmul_swish_scale_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty((in_features, out_features)))
        self.scaling_factor = scaling_factor
        # Initialize weights similar to PyTorch's Linear layer
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        batch_size = x.size(0)
        in_features = self.weight.size(0)
        out_features = self.weight.size(1)
        return fused_op.fused_matmul_swish_scale_cuda(
            x, self.weight, self.scaling_factor,
            batch_size, in_features, out_features
        )

def get_inputs():
    batch_size = 128
    in_features = 32768
    return [torch.randn(batch_size, in_features, device='cuda')]

def get_init_inputs():
    return [32768, 32768, 2.0]