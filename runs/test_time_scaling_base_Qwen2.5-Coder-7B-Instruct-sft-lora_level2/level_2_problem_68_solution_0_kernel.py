import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication followed by min and subtraction
matrix_min_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_min_subtract_kernel(const float* x, const float* weight, const float* constant, float* out, int batch_size, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_features) {
        int row = idx / out_features;
        int col = idx % out_features;
        float val = x[row * in_features + col] * weight[col * out_features + row];
        if (val < constant[row]) {
            out[idx] = val - constant[row];
        } else {
            out[idx] = constant[row] - constant[row];
        }
    }
}

torch::Tensor matrix_min_subtract_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor constant) {
    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto out_features = weight.size(1);
    auto out = torch::zeros({batch_size, out_features}, device=x.device());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_features + block_size - 1) / block_size;

    matrix_min_subtract_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), weight.data_ptr<float>(), constant.data_ptr<float>(), out.data_ptr<float>(), batch_size, in_features, out_features);

    return out;
}
"""

matrix_min_subtract_cpp_source = (
    "torch::Tensor matrix_min_subtract_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor constant);"
)

# Compile the inline CUDA code for matrix multiplication followed by min and subtraction
matrix_min_subtract = load_inline(
    name="matrix_min_subtract",
    cpp_sources=matrix_min_subtract_cpp_source,
    cuda_sources=matrix_min_subtract_source,
    functions=["matrix_min_subtract_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        self.matrix_min_subtract = matrix_min_subtract

    def forward(self, x):
        x = self.linear(x)
        x = self.matrix_min_subtract.matrix_min_subtract_cuda(x, self.linear.weight.t(), self.constant)
        return x

batch_size = 128
in_features = 16384
out_features = 16384
constant = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, constant]