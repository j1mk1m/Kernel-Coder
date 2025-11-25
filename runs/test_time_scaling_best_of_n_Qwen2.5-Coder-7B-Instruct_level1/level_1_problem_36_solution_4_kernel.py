import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for RMS Normalization
rms_normalization_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void rms_normalization_kernel(const float* x, float* y, int batch_size, int num_features, int dim1, int dim2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_features * dim1 * dim2) {
        int i = idx / (dim1 * dim2);
        int j = idx % (dim1 * dim2);
        float mean_square = x[i * num_features * dim1 * dim2 + j];
        float rms = sqrt(mean_square + 0.000001f);
        y[idx] = x[idx] / rms;
    }
}

torch::Tensor rms_normalization_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto num_features = x.size(1);
    auto dim1 = x.size(2);
    auto dim2 = x.size(3);
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (batch_size * num_features * dim1 * dim2 + block_size - 1) / block_size;

    rms_normalization_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, num_features, dim1, dim2);

    return out;
}
"""

rms_normalization_cpp_source = (
    "torch::Tensor rms_normalization_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for RMS Normalization
rms_normalization = load_inline(
    name="rms_normalization",
    cpp_sources=rms_normalization_cpp_source,
    cuda_sources=rms_normalization_source,
    functions=["rms_normalization_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rms_normalization = rms_normalization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input tensor to perform RMS Normalization across all dimensions except the batch dimension
        x_flattened = x.view(x.size(0), -1)
        # Compute the mean square of each feature across the flattened dimensions
        mean_square = torch.mean(x_flattened ** 2, dim=1, keepdim=True)
        # Apply RMS Normalization using the custom CUDA kernel
        normalized_x = self.rms_normalization.rms_normalization_cuda(mean_square)
        # Reshape the normalized tensor back to the original shape
        normalized_x = normalized_x.view(x.shape)
        return x / normalized_x


batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]