import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel implementation
avg_pool_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>  // for std::max and std::min

__global__ void avg_pool1d_kernel(const float* input, float* output,
                                  int batch, int channels, int input_length,
                                  int kernel_size, int stride, int padding,
                                  int output_length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * channels * output_length)
        return;

    // Compute batch, channel, pos
    int b = index / (channels * output_length);
    int rem = index % (channels * output_length);
    int c = rem / output_length;
    int pos = rem % output_length;

    // Calculate start_padded and end_padded
    int start_padded = pos * stride;
    int end_padded = start_padded + kernel_size - 1;

    // Compute original indices range
    int original_start = start_padded - padding;
    int original_end = end_padded - padding;

    int valid_start = std::max(0, original_start);
    int valid_end = std::min(original_end, input_length - 1);

    float sum = 0.0f;
    if (valid_start <= valid_end) {
        for (int x = valid_start; x <= valid_end; ++x) {
            int input_offset = b * channels * input_length +
                               c * input_length + x;
            sum += input[input_offset];
        }
    }

    // Compute output offset
    int output_offset = b * channels * output_length +
                        c * output_length + pos;
    output[output_offset] = sum / kernel_size;
}

torch::Tensor avg_pool1d_cuda(torch::Tensor input,
                              int kernel_size, int stride, int padding) {
    // Ensure input is contiguous and on the same device
    input = input.contiguous();

    // Get input dimensions
    int batch = input.size(0);
    int channels = input.size(1);
    int input_length = input.size(2);

    // Compute output length
    int padded_length = input_length + 2 * padding;
    int output_length = (padded_length - kernel_size) / stride + 1;

    // Create output tensor
    auto output = torch::empty({batch, channels, output_length}, input.options());

    // Launch kernel
    const int threads_per_block = 256;
    int num_elements = batch * channels * output_length;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    avg_pool1d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, input_length,
        kernel_size, stride, padding,
        output_length
    );

    return output;
}
"""

avg_pool_cuda_cpp_source = (
    "torch::Tensor avg_pool1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"
)

avg_pool_cuda = load_inline(
    name="avg_pool_cuda",
    cpp_sources=avg_pool_cuda_cpp_source,
    cuda_sources=avg_pool_cuda_source,
    functions=["avg_pool1d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avg_pool_cuda = avg_pool_cuda  # the compiled module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool_cuda.avg_pool1d_cuda(
            x, self.kernel_size, self.stride, self.padding
        )