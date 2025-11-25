import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D Average Pooling
avg_pool_1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool_1d_kernel(const float* input, float* output, int batch_size, int in_channels, int input_length, int kernel_size, int stride, int padding) {
    int b_idx = blockIdx.x / (in_channels * input_length);
    int c_idx = (blockIdx.x % (in_channels * input_length)) / input_length;
    int i_idx = blockIdx.x % input_length;

    int o_idx = (i_idx - padding) / stride;

    if (o_idx >= 0 && o_idx <= input_length - kernel_size) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            sum += input[b_idx * in_channels * input_length + c_idx * input_length + (i_idx - padding + k)];
        }
        output[b_idx * in_channels * input_length + c_idx * input_length + o_idx] = sum / kernel_size;
    } else {
        output[b_idx * in_channels * input_length + c_idx * input_length + o_idx] = 0.0f;
    }
}

torch::Tensor avg_pool_1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_length = input.size(2);
    auto output_length = (input_length - padding) / stride;

    auto output = torch::zeros({batch_size, in_channels, output_length}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * in_channels * output_length + block_size - 1) / block_size;

    avg_pool_1d_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, input_length, kernel_size, stride, padding);

    return output;
}
"""

avg_pool_1d_cpp_source = (
    "torch::Tensor avg_pool_1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"
)

# Compile the inline CUDA code for 1D Average Pooling
avg_pool_1d = load_inline(
    name="avg_pool_1d",
    cpp_sources=avg_pool_1d_cpp_source,
    cuda_sources=avg_pool_1d_source,
    functions=["avg_pool_1d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.avg_pool = avg_pool_1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool.avg_pool_1d_cuda(x, self.kernel_size, self.stride, self.padding)