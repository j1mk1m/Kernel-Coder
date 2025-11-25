import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GEMM, Swish, Divide, Clamp, Tanh, and Clamp
gemm_swish_divide_clamp_tanh_clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gemm_swish_divide_clamp_tanh_clamp_kernel(const float* a, const float* b, float* c, float* d, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
        d[row * n + col] = sum * (sum > 0 ? 1.0f : 0.0f);  // Swish activation
    }
}

torch::Tensor gemm_swish_divide_clamp_tanh_clamp_cuda(torch::Tensor a, torch::Tensor b) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);
    auto c = torch::zeros({m, n}, a.options());
    auto d = torch::zeros({m, n}, a.options());

    const int block_size = 32;
    const int num_blocks_x = (n + block_size - 1) / block_size;
    const int num_blocks_y = (m + block_size - 1) / block_size;

    gemm_swish_divide_clamp_tanh_clamp_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), d.data_ptr<float>(), m, n, k);

    return {c, d};
}
"""

gemm_swish_divide_clamp_tanh_clamp_cpp_source = (
    "std::tuple<torch::Tensor, torch::Tensor> gemm_swish_divide_clamp_tanh_clamp_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for GEMM, Swish, Divide, Clamp, Tanh, and Clamp
gemm_swish_divide_clamp_tanh_clamp = load_inline(
    name="gemm_swish_divide_clamp_tanh_clamp",
    cpp_sources=gemm_swish_divide_clamp_tanh_clamp_cpp_source,
    cuda_sources=gemm_swish_divide_clamp_tanh_clamp_source,
    functions=["gemm_swish_divide_clamp_tanh_clamp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.gemm_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.gemm_bias = nn.Parameter(torch.randn(out_features)) if bias else None

    def forward(self, x):
        x, swish_output = gemm_swish_divide_clamp_tanh_clamp.gemm_swish_divide_clamp_tanh_clamp_cuda(x, self.gemm_weight.t())
        if self.bias is not None:
            x = x + self.gemm_bias
        x = x / 2.0
        x = torch.clamp(x, min=-1.0, max=1.0)  # Clamp between -1 and 1
        x = torch.tanh(x)  # Tanh activation
        x = torch.clamp(x, min=-1.0, max=1.0)  # Clamp between -1 and 1
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]

model_new = ModelNew(in_features, out_features)
input_tensor = get_inputs()[0].cuda()
output_tensor = model_new(input_tensor.cuda()).cpu()

print(output_tensor.shape)