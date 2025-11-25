import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

custom_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_fused_kernel(const float* x, const float* weight, const float* bias,
                                   float subtract_value, float multiply_value,
                                   float* output,
                                   int batch_size, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int i = idx / out_features;
    int j = idx % out_features;

    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        sum += x[i * in_features + k] * weight[j * in_features + k];
    }

    sum += bias[j];
    sum = (sum - subtract_value) * multiply_value;
    output[i * out_features + j] = fmaxf(sum, 0.0f);
}

torch::Tensor custom_fused_op(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                             float subtract_value, float multiply_value) {
    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, x.options());

    int num_elements = batch_size * out_features;
    const int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    custom_fused_kernel<<<blocks_per_grid, threads_per_block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        subtract_value,
        multiply_value,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

custom_fused_cpp_source = """
torch::Tensor custom_fused_op(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                             float subtract_value, float multiply_value);
"""

custom_fused_op = load_inline(
    name="custom_fused_op",
    cpp_sources=custom_fused_cpp_source,
    cuda_sources=custom_fused_source,
    functions=["custom_fused_op"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        weight = self.linear.weight
        bias = self.linear.bias
        return custom_fused_op(x, weight, bias, self.subtract_value, self.multiply_value)

batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]