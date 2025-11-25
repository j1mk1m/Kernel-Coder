import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel and wrapper function
avg_pool1d_cpp_source = (
    "torch::Tensor avg_pool1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"
)

avg_pool1d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(const float* input, float* output,
                                 int batch_size, int channels, int input_length,
                                 int kernel_size, int stride, int padding, int output_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_length)
        return;

    int o = idx % output_length;
    int c = (idx / output_length) % channels;
    int b = idx / (output_length * channels);

    float sum = 0.0f;
    int start = o * stride;
    for (int i = 0; i < kernel_size; ++i) {
        int padded_pos = start + i;
        int original_pos = padded_pos - padding;
        if (original_pos >= 0 && original_pos < input_length) {
            sum += input[b * channels * input_length + c * input_length + original_pos];
        }
    }

    output[b * channels * output_length + c * output_length + o] = sum / kernel_size;
}

torch::Tensor avg_pool1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    int input_length = input.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({input.size(0), input.size(1), output_length}, input.options());

    const int threads_per_block = 256;
    int total_elements = input.size(0) * input.size(1) * output_length;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    avg_pool1d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0), input.size(1), input_length,
        kernel_size, stride, padding, output_length
    );

    return output;
}
"""

# Compile the CUDA code
avg_pool_cuda = load_inline(
    name="avg_pool_cuda",
    cpp_sources=avg_pool1d_cpp_source,
    cuda_sources=avg_pool1d_cuda_source,
    functions=["avg_pool1d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avg_pool_cuda.avg_pool1d_cuda(x, self.kernel_size, self.stride, self.padding)

def get_inputs():
    x = torch.rand(64, 128, 65536).cuda()
    return [x]

def get_init_inputs():
    return [8, 1, 4]