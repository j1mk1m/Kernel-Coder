import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Matmul, AvgPool, GELU, Scale, Max
matmul_avgpool_gelu_scale_max_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_avgpool_gelu_scale_max_kernel(const float* a, const float* b, float* out, int batch_size, int in_features, int out_features, int pool_kernel_size, float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int i = idx / out_features;
    int j = idx % out_features;

    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        sum += a[i * in_features + k] * b[k * out_features + j];
    }

    // Average Pooling
    int avg_idx = i * (in_features / pool_kernel_size);
    sum /= pool_kernel_size;

    // GELU activation
    float gelu_val = 0.5 * sum * (1.0 + tanh(sqrt(2.0 / M_PI) * (sum + 0.044715 * sum * sum * sum)));

    // Scale
    gelu_val *= scale_factor;

    // Max operation
    atomicMax(out + i * out_features + j, gelu_val);
}

torch::Tensor matmul_avgpool_gelu_scale_max_cuda(torch::Tensor a, torch::Tensor b, int pool_kernel_size, float scale_factor) {
    auto batch_size = a.size(0);
    auto in_features = a.size(1);
    auto out_features = b.size(1);
    auto out = torch::zeros({batch_size, out_features}, a.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_features + block_size - 1) / block_size;

    matmul_avgpool_gelu_scale_max_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), batch_size, in_features, out_features, pool_kernel_size, scale_factor);

    return out;
}
"""

matmul_avgpool_gelu_scale_max_cpp_source = (
    "torch::Tensor matmul_avgpool_gelu_scale_max_cuda(torch::Tensor a, torch::Tensor b, int pool_kernel_size, float scale_factor);"
)

# Compile the inline CUDA code for Matmul, AvgPool, GELU, Scale, Max
matmul_avgpool_gelu_scale_max = load_inline(
    name="matmul_avgpool_gelu_scale_max",
    cpp_sources=matmul_avgpool_gelu_scale_max_cpp_source,
    cuda_sources=matmul_avgpool_gelu_scale_max_source,
    functions=["matmul_avgpool_gelu_scale_max_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul_avgpool_gelu_scale_max = matmul_avgpool_gelu_scale_max

    def forward(self, x):
        return self.matmul_avgpool_gelu_scale_max.matmul_avgpool_gelu_scale_max_cuda(x, x, pool_kernel_size, scale_factor)


# Test the ModelNew class
batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0

model_new = ModelNew(in_features, out_features, pool_kernel_size, scale_factor)
inputs = get_inputs()[0].cuda()

output = model_new(inputs)
print(output.shape)  # Should print torch.Size([1024])