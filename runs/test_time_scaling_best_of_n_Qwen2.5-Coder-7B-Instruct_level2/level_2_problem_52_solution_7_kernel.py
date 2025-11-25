import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batch normalization
batch_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batch_norm_kernel(const float* input, float* output, float* mean, float* var, float* gamma, float* beta, float eps, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float inv_stddev = rsqrt(var[idx] + eps);
        output[idx] = gamma[idx] * (input[idx] - mean[idx]) * inv_stddev + beta[idx];
    }
}

torch::Tensor batch_norm_cuda(torch::Tensor input, float* gamma, float* beta, float eps) {
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto output = torch::zeros_like(input);
    float* mean = new float[channels];
    float* var = new float[channels];

    batch_norm_kernel<<<...>>>(input.data_ptr<float>(), output.data_ptr<float>(), mean, var, gamma, beta, eps, channels, height, width);

    delete[] mean;
    delete[] var;

    return output;
}
"""

batch_norm_cpp_source = (
    "torch::Tensor batch_norm_cuda(torch::Tensor input, float* gamma, float* beta, float eps);"
)

# Compile the inline CUDA code for batch normalization
batch_norm = load_inline(
    name="batch_norm",
    cpp_sources=batch_norm_cpp_source,
    cuda_sources=batch_norm_source,
    functions=["batch_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.bn = batch_norm
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        x = self.conv(x)
        x = softplus_cuda(x)
        x = tanh_cuda(x)
        x = multiplication_cuda(x, x)
        x = self.bn(x, self.gamma, self.beta)
        return x