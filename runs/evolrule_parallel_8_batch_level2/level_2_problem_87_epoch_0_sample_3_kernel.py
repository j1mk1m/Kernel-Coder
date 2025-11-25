import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

custom_mish_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void custom_mish_subtract_kernel(const float* x, float v, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = x[idx] - v;
        float exp_temp = expf(temp);
        float softplus = log1pf(exp_temp);
        float tanh_soft = tanhf(softplus);
        out[idx] = temp * tanh_soft;
    }
}

torch::Tensor custom_mish_subtract_cuda(torch::Tensor x, float v) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    custom_mish_subtract_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), v, out.data_ptr<float>(), size);

    return out;
}
"""

custom_mish_subtract_cpp_source = """
torch::Tensor custom_mish_subtract_cuda(torch::Tensor x, float v);
"""

custom_mish_subtract = load_inline(
    name="custom_mish_subtract",
    cpp_sources=[custom_mish_subtract_cpp_source],
    cuda_sources=[custom_mish_subtract_source],
    functions=["custom_mish_subtract_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.custom_mish_subtract = custom_mish_subtract

    def forward(self, x):
        x = self.conv(x)
        v = self.subtract_value_1 + self.subtract_value_2
        return self.custom_mish_subtract.custom_mish_subtract_cuda(x, v)