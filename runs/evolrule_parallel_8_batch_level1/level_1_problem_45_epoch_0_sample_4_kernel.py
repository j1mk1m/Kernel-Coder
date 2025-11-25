import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 16
channels = 64
height = 2048
width = 2048
kernel_size = 11

avg_pool_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool_2d_kernel(
    const float* input, float* output,
    int batch_size, int channels,
    int input_height, int input_width,
    int kernel_size, int stride, int padding,
    int output_height, int output_width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size * channels * output_height * output_width) return;

    // Compute indices
    int output_w = idx % output_width;
    int tmp = idx / output_width;
    int output_h = tmp % output_height;
    tmp /= output_height;
    int channel = tmp % channels;
    int batch = tmp / channels;

    // Compute input start positions considering padding
    int h_start = (output_h * stride) - padding;
    int w_start = (output_w * stride) - padding;

    // Accumulator
    float sum = 0.0;

    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int h = h_start + i;
            int w = w_start + j;
            // Check if within input bounds (consider padding)
            if (h < 0 || h >= input_height || w < 0 || w >= input_width) {
                continue;
            }
            // Compute index
            int in_idx = batch * channels * input_height * input_width +
                         channel * input_height * input_width +
                         h * input_width + w;
            sum += input[in_idx];
        }
    }

    // Compute output index
    int out_idx = batch * channels * output_height * output_width +
                  channel * output_height * output_width +
                  output_h * output_width + output_w;

    output[out_idx] = sum / (kernel_size * kernel_size);
}

torch::Tensor avg_pool_2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    // Check input is 4D
    if (input.dim() != 4) {
        throw std::runtime_error("Input tensor must be 4-dimensional");
    }

    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    // Compute output dimensions
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    // Create output tensor
    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Number of threads per block
    const int threads_per_block = 256;
    const int num_elements = batch_size * channels * output_height * output_width;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    avg_pool_2d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        input_height, input_width,
        kernel_size, stride, padding,
        output_height, output_width);

    // Synchronize to check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}
"""

avg_pool_cuda_header = """
torch::Tensor avg_pool_2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);
"""

# Compile the CUDA code
avg_pool_cuda = load_inline(
    name="avg_pool_cuda",
    cuda_sources=avg_pool_cuda_source,
    cpp_sources=avg_pool_cuda_header,
    functions=["avg_pool_2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avg_pool_cuda.avg_pool_2d_cuda(x, self.kernel_size, self.stride, self.padding)

def get_inputs():
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size]