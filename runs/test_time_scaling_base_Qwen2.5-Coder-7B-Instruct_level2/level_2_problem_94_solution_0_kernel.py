import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GEMM
gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void gemm_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream());
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, (const float*)1, b, n, a, k, (const float*)0, c, n);
    cublasDestroy(handle);
}
"""

gemm_cpp_source = (
    "void gemm_kernel(const float* a, const float* b, float* c, int m, int n, int k);"
)

# Compile the inline CUDA code for GEMM
gemm = load_inline(
    name="gemm",
    cpp_sources=gemm_cpp_source,
    cuda_sources=gemm_source,
    functions=["gemm_kernel"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        batch_size, in_features = x.shape
        out_features = self.bias.shape[0]

        y = torch.empty((batch_size, out_features)).cuda()
        gemm_kernel(x.data_ptr(), self.weight.data_ptr(), y.data_ptr(), batch_size, out_features, in_features)

        y = y + self.bias
        y = self.hardtanh(y)
        y = self.mish(y)
        y = self.groupnorm(y)
        return y


# Initialize weights for GEMM
model_new = ModelNew(in_features, out_features, bias_shape, num_groups)
model_new.weight = nn.Parameter(torch.randn(out_features, in_features))

# Get inputs
inputs = get_inputs()

# Forward pass
output = model_new(inputs[0])
print(output.shape)