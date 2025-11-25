import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom InstanceNorm2d CUDA kernel implementation
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void instance_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch_size,
    int channels,
    int height,
    int width,
    float eps) {

    int batch = blockIdx.x;
    int channel = blockIdx.y;

    // Compute mean and variance for current (batch, channel)
    int spatial_size = height * width;
    float mean = 0;
    float var = 0;
    for (int i = 0; i < spatial_size; ++i) {
        int idx = batch * channels * spatial_size + channel * spatial_size + i;
        mean += input[idx];
    }
    mean /= spatial_size;

    for (int i = 0; i < spatial_size; ++i) {
        int idx = batch * channels * spatial_size + channel * spatial_size + i;
        float val = input[idx] - mean;
        var += val * val;
    }
    var = var / spatial_size + eps;

    float inv_std = 1.0f / sqrtf(var);

    // Normalize and apply affine
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        int idx = batch * channels * spatial_size + channel * spatial_size + i;
        scalar_t val = input[idx];
        output[idx] = (val - mean) * inv_std * weight[channel] + bias[channel];
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> instance_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps) {

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int spatial_size = height * width;
    const int total_elements = batch_size * channels * spatial_size;

    auto output = torch::empty_like(input);
    auto mean = torch::empty({batch_size, channels}, input.options());
    auto var = torch::empty({batch_size, channels}, input.options());

    dim3 blocks(batch_size, channels);
    dim3 threads(256); // Use 256 threads per block

    AT_DISPATCH_FLOATING_TYPES(input.type(), "instance_norm_cuda", ([&] {
        instance_norm_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            batch_size,
            channels,
            height,
            width,
            eps);
    }));

    return std::make_tuple(output, mean, var);
}
"""

# Inline compilation
instance_norm_cpp = """
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> instance_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps);
"""

instance_norm = load_inline(
    name="instance_norm",
    cpp_sources=instance_norm_cpp,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_features):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = 1e-5  # Using PyTorch default epsilon
        self.instance_norm = instance_norm

    def forward(self, x):
        # Reshape for compatibility with kernel
        B, C, H, W = x.shape
        x_reshaped = x.reshape(B, C, H * W)  # Reshape to (B, C, H*W)
        output, _, _ = self.instance_norm.instance_norm_cuda(
            x_reshaped, self.weight, self.bias, self.eps
        )
        output = output.reshape(B, C, H, W)
        return output

# Ensure that the weight and bias are initialized similarly to PyTorch's InstanceNorm
def weights_init(m):
    if isinstance(m, ModelNew):
        torch.nn.init.constant_(m.weight, 1.0)
        torch.nn.init.constant_(m.bias, 0.0)

# Apply initialization
ModelNew.apply(weights_init)