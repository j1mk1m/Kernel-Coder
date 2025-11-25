import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused kernel code
fused_bngelu_relu_cuda = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_bngelu_relu_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float eps,
    float* output,
    int batch_size,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int j = idx % out_features;
    int i = idx / out_features;

    float y_j = input[idx];
    float mean = running_mean[j];
    float var = running_var[j];
    float inv_std = 1.0f / sqrtf(var + eps);
    float normalized = (y_j - mean) * inv_std;
    normalized = normalized * gamma[j] + beta[j];

    // Compute GELU using tanh approximation
    float x = normalized;
    float inner = sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x);
    float gelu_val = 0.5f * x * (1.0f + tanhf(inner));

    // Apply ReLU
    float relu_val = fmaxf(gelu_val, 0.0f);

    output[idx] = relu_val;
}

torch::Tensor fused_bngelu_relu(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
) {
    int batch_size = input.size(0);
    int out_features = input.size(1);
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_elements = batch_size * out_features;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_bngelu_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        output.data_ptr<float>(),
        batch_size,
        out_features
    );

    return output;
}
"""

fused_bngelu_relu_cpp = """
torch::Tensor fused_bngelu_relu(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
);
"""

# Compile the fused kernel
fused_mod = load_inline(
    name="fused_bngelu_relu",
    cpp_sources=fused_bngelu_relu_cpp,
    cuda_sources=fused_bngelu_relu_cuda,
    functions=["fused_bngelu_relu"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.fused_bngelu_relu = fused_mod.fused_bngelu_relu

    def forward(self, x):
        x = self.gemm(x)
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        running_mean = self.batch_norm.running_mean
        running_var = self.batch_norm.running_var
        eps = self.batch_norm.eps

        x = self.fused_bngelu_relu(x, gamma, beta, running_mean, running_var, eps)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]