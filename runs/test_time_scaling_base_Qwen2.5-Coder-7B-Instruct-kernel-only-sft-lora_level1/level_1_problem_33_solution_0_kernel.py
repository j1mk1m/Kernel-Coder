import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Batch Normalization
bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void bn_forward_kernel(const float* x, const float* mean, const float* var, const float* weight, const float* bias, float* y, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width) {
        return;
    }

    int c = idx / (height * width);
    int h = (idx / width) % height;
    int w = idx % width;
    int i = idx;

    float val = x[i];
    float norm_val = (val - mean[c]) / std::sqrt(var[c] + 1e-5f);
    y[i] = weight[c] * norm_val + bias[c];
}

torch::Tensor bn_forward_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    auto size = batch_size * channels * height * width;

    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    bn_forward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(), batch_size, channels, height, width);

    return y;
}
"""

bn_cpp_source = (
    "torch::Tensor bn_forward_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code for Batch Normalization
bn = load_inline(
    name="bn",
    cpp_sources=bn_cpp_source,
    cuda_sources=bn_source,
    functions=["bn_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA operators for Batch Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)
        self.register_buffer('mean', torch.zeros(num_features))
        self.register_buffer('var', torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        batch_size, channels, height, width = x.size()
        x_flat = x.view(batch_size, channels, -1)
        mean = torch.mean(x_flat, dim=-1, keepdim=True)
        var = torch.var(x_flat, dim=-1, keepdim=True, unbiased=False)

        weight = self.bn.weight.view(-1, 1, 1, 1)
        bias = self.bn.bias.view(-1, 1, 1, 1)

        y = bn.bn_forward_cuda(x.contiguous(), mean, var, weight, bias)
        return y