import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused kernel CUDA source code
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_kernel(const float* input, const float* bias, float* output, int batch_size, int out_features) {
    __shared__ float bias_sh[8192]; // Assuming out_features is fixed at 8192

    // Load bias into shared memory
    for (int i = threadIdx.x; i < 8192; i += blockDim.x) {
        bias_sh[i] = bias[i];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int n = idx / out_features;
    int c = idx % out_features;

    float val = input[idx] + bias_sh[c];

    // Apply Hardtanh
    if (val < -1.0f) val = -1.0f;
    else if (val > 1.0f) val = 1.0f;

    // Apply Mish using approximation
    float exp_val = expf(val);
    val = val * exp_val / (2.0f + exp_val);

    output[idx] = val;
}

torch::Tensor fused_op(torch::Tensor input, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto out_features = input.size(1);
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_elements = batch_size * out_features;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features
    );

    return output;
}
"""

# Corresponding C++ header for the fused operation
fused_kernel_cpp_source = (
    "torch::Tensor fused_op(torch::Tensor input, torch::Tensor bias);"
)

# Load the CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--use-fast-math"],
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.fused_op = fused_op  # Load the fused CUDA operator

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_op.fused_op(x, self.bias)  # Apply fused kernel
        x = self.groupnorm(x)
        return x