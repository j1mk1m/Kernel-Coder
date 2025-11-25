import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <math.h>

template <typename scalar_t>
__global__ void fused_batchnorm_gelu_relu_kernel(
    const scalar_t* input,
    const scalar_t* scale,
    const scalar_t* bias,
    scalar_t* output,
    int batch_size,
    int features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features) return;

    int j = idx % features;
    scalar_t x = input[idx];
    scalar_t scaled_x = x * scale[j] + bias[j];

    // Compute GELU approximation
    scalar_t temp = scaled_x + 0.044715f * scaled_x * scaled_x * scaled_x;
    scalar_t tanh_val = tanhf(sqrtf(2.0f / M_PI) * temp);
    scalar_t gelu_val = 0.5f * scaled_x * (1.0f + tanh_val);

    // Apply ReLU
    output[idx] = fmaxf(gelu_val, 0.0f);
}

at::Tensor fused_batchnorm_gelu_relu_cuda(
    const at::Tensor& input,
    const at::Tensor& scale,
    const at::Tensor& bias,
    int batch_size,
    int features
) {
    const int total_elements = batch_size * features;
    auto output = at::empty_like(input);

    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_batchnorm_gelu_relu_cuda", ([&] {
        fused_batchnorm_gelu_relu_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            scale.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            features
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }
    return output;
}
"""

cpp_source = """
#include <torch/extension.h>

at::Tensor fused_batchnorm_gelu_relu_cuda(
    const at::Tensor& input,
    const at::Tensor& scale,
    const at::Tensor& bias,
    int batch_size,
    int features
);
"""

# Load the fused CUDA operator
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_batchnorm_gelu_relu_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.gemm(x)
        # Compute batch norm parameters
        eps = self.batch_norm.eps
        scale = self.batch_norm.weight / torch.sqrt(self.batch_norm.running_var + eps)
        bias = self.batch_norm.bias - self.batch_norm.running_mean * scale
        # Ensure tensors are on the same device as input
        scale = scale.to(x.device)
        bias = bias.to(x.device)
        batch_size, features = x.size()
        x = fused_ops.fused_batchnorm_gelu_relu_cuda(x, scale, bias, batch_size, features)
        return x