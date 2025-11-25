import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define fused linear + mish + mish CUDA kernel
fused_linear_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_linear_mish_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < out_features) {
        float sum = bias[col];
        for (int k = 0; k < in_features; ++k) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }

        // First Mish activation
        float z = sum;
        float exp_z = exp(z);
        float softplus = log(1 + exp_z);
        float tanh_sp = tanh(softplus);
        float mish1 = z * tanh_sp;

        // Second Mish activation
        z = mish1;
        exp_z = exp(z);
        softplus = log(1 + exp_z);
        tanh_sp = tanh(softplus);
        float mish2 = z * tanh_sp;

        output[row * out_features + col] = mish2;
    }
}

torch::Tensor fused_linear_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, 
                              torch::dtype(input.dtype()).device(input.device()));

    dim3 threads(32, 8);
    dim3 blocks(
        (out_features + threads.x - 1) / threads.x,
        (batch_size + threads.y - 1) / threads.y
    );

    fused_linear_mish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

# Define the fused kernel module
fused_linear_mish = load_inline(
    name="fused_linear_mish",
    cpp_sources="",
    cuda_sources=fused_linear_mish_source,
    functions=["fused_linear_mish_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.fused_op = fused_linear_mish
        
        # Initialize weights and bias like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.fused_op.fused_linear_mish_cuda(x, self.weight, self.bias)

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]