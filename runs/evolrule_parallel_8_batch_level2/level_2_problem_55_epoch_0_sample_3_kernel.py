import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matmul_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int batch = idx / out_features;
    int out_f = idx % out_features;

    float sum = 0;
    for (int i = 0; i < in_features; ++i) {
        sum += input[batch * in_features + i] * weight[out_f * in_features + i];
    }
    output[batch * out_features + out_f] = sum + bias[out_f];
}

torch::Tensor matmul_cuda(torch::Tensor input,
                         torch::Tensor weight,
                         torch::Tensor bias) {
    const int threads = 256;
    const int elements = input.size(0) * weight.size(0);
    const dim3 blocks((elements + threads - 1) / threads, 1);

    auto output = torch::empty({input.size(0), weight.size(0)},
                              input.options());

    matmul_bias_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0),
        input.size(1),
        weight.size(0));

    cudaDeviceSynchronize();
    return output;
}
"""

matmul_cuda_ext = load_inline(
    name='matmul_cuda',
    cpp_sources='',
    cuda_sources=matmul_kernel_source,
    functions=['matmul_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = matmul_cuda_ext.matmul_cuda(x, self.matmul.weight, self.matmul.bias)
        x = self.max_pool(x.unsqueeze(1)).squeeze(1)
        x = torch.sum(x, dim=1) * self.scale_factor
        return x