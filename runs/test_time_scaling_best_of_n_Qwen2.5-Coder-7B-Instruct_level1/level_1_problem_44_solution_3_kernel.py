import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D Average Pooling
avg_pool_1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool_1d_kernel(const float* input, float* output, int batch_size, int channels, int input_length, int kernel_size, int stride, int padding) {
    int batch_idx = blockIdx.x / (channels * ((input_length + 2 * padding - kernel_size) / stride + 1));
    int channel_idx = (blockIdx.x % (channels * ((input_length + 2 * padding - kernel_size) / stride + 1))) / ((input_length + 2 * padding - kernel_size) / stride + 1);
    int output_idx = blockIdx.x % ((input_length + 2 * padding - kernel_size) / stride + 1);
    int input_start = batch_idx * channels * input_length + channel_idx * input_length + output_idx * stride - padding;
    int sum = 0;
    for (int i = 0; i < kernel_size; ++i) {
        int idx = input_start + i;
        if (idx >= 0 && idx < input_length) {
            sum += input[idx];
        }
    }
    output[blockIdx.x] = static_cast<float>(sum) / kernel_size;
}

torch::Tensor avg_pool_1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto input_length = input.size(2);
    auto output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    auto output = torch::zeros({batch_size, channels, output_length}, input.options());

    const int block_size = 256;
    const int num_blocks = batch_size * channels * output_length;

    avg_pool_1d_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, input_length, kernel_size, stride, padding);

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
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avg_pool = avg_pool_1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool.avg_pool_1d_cuda(x, self.kernel_size, self.stride, self.padding)