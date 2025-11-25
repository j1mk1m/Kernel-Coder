import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batch_norm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float* output,
    int N, int C, int H, int W,
    float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W)
        return;

    int c = (idx % (C * H * W)) / (H * W);

    float mean = running_mean[c];
    float var = running_var[c];
    float inv_std = 1.0f / sqrtf(var + eps);
    float x = input[idx] - mean;
    float norm_x = x * inv_std;

    output[idx] = norm_x * gamma[c] + beta[c];
}

torch::Tensor batch_norm_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps) {
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t H = input.size(2);
    const int64_t W = input.size(3);

    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_elements = N * C * H * W;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    batch_norm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, eps);

    return output;
}
"""

batch_norm_cpp_source = """
torch::Tensor batch_norm_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps);
"""

batch_norm = load_inline(
    name="batch_norm",
    cpp_sources=batch_norm_cpp_source,
    cuda_sources=batch_norm_source,
    functions=["batch_norm_forward"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = 1e-5
        self.batch_norm_forward = batch_norm.batch_norm_forward

    def forward(self, x):
        x = x.contiguous()
        return self.batch_norm_forward(
            x, self.weight, self.bias, self.running_mean, self.running_var, self.eps
        )