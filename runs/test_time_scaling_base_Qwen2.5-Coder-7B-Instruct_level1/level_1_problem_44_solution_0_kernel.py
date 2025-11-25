import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D Average Pooling
avg_pool_1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool_1d_kernel(const float* input, float* output, int batch_size, int in_channels, int input_length, int kernel_size, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * in_channels) {
        int b = idx / in_channels;
        int c = idx % in_channels;
        int start_idx = (idx * kernel_size) / in_channels - padding;
        int end_idx = start_idx + kernel_size;

        float sum = 0.0f;
        int count = 0;
        for (int i = start_idx; i < end_idx && i < input_length; ++i) {
            sum += input[b * in_channels * input_length + c * input_length + i];
            count++;
        }

        output[idx] = sum / count;
    }
}

torch::Tensor avg_pool_1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_length = input.size(2);
    auto output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, in_channels, output_length}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * in_channels + block_size - 1) / block_size;

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