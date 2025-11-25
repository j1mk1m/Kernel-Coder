import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Instance Normalization
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void instance_norm_forward_kernel(const float* x, const int* N, const int* C, const int* H, const int* W, float* y, float* mean, float* var, float eps) {
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (n >= N[0] || c >= C[0]) return;

    float sum = 0.0f;
    for (int h = 0; h < H[0]; ++h) {
        for (int w = 0; w < W[0]; ++w) {
            sum += x[(n * C[0] + c) * H[0] * W[0] + h * W[0] + w];
        }
    }

    mean[n * C[0] + c] = sum / (H[0] * W[0]);

    float sum_sq = 0.0f;
    for (int h = 0; h < H[0]; ++h) {
        for (int w = 0; w < W[0]; ++w) {
            sum_sq += pow(x[(n * C[0] + c) * H[0] * W[0] + h * W[0] + w] - mean[n * C[0] + c], 2);
        }
    }

    var[n * C[0] + c] = sum_sq / (H[0] * W[0]);

    for (int h = 0; h < H[0]; ++h) {
        for (int w = 0; w < W[0]; ++w) {
            y[(n * C[0] + c) * H[0] * W[0] + h * W[0] + w] = (x[(n * C[0] + c) * H[0] * W[0] + h * W[0] + w] - mean[n * C[0] + c]) / sqrt(var[n * C[0] + c] + eps);
        }
    }
}

torch::Tensor instance_norm_forward_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, float eps) {
    auto N = x.sizes()[0];
    auto C = x.sizes()[1];
    auto H = x.sizes()[2];
    auto W = x.sizes()[3];

    auto y = torch::zeros_like(x);
    auto mean_out = torch::zeros(N * C, x.options().dtype(torch::kFloat32));
    auto var_out = torch::zeros(N * C, x.options().dtype(torch::kFloat32));

    const int block_size = 16;
    const int num_blocks_x = (C + block_size - 1) / block_size;
    const int num_blocks_y = (N + block_size - 1) / block_size;

    instance_norm_forward_kernel<<<num_blocks_y, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        &N,
        &C,
        &H,
        &W,
        y.data_ptr<float>(),
        mean_out.data_ptr<float>(),
        var_out.data_ptr<float>(),
        eps
    );

    return y;
}
"""

instance_norm_cpp_source = (
    "torch::Tensor instance_norm_forward_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, float eps);"
)

# Compile the inline CUDA code for Instance Normalization
instance_norm = load_inline(
    name="instance_norm",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Simple model that performs Instance Normalization using a custom CUDA kernel.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.mean = nn.Parameter(torch.zeros(num_features))
        self.var = nn.Parameter(torch.ones(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return instance_norm.instance_norm_forward_cuda(x, self.mean, self.var, self.eps)

# Example usage
if __name__ == "__main__":
    batch_size = 112  # heavier workload
    features = 64
    dim1 = 512
    dim2 = 512

    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    model = ModelNew(features).cuda()

    output = model(x)
    print(output.shape)  # Should be (batch_size, features, dim1, dim2)