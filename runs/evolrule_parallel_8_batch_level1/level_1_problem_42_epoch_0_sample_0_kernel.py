import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_pool_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

#define INFINITY std::numeric_limits<float>::infinity()

__global__ void max_pool_2d_cuda(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_h,
    int input_w,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int output_h,
    int output_w
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * channels * output_h * output_w) {
        return;
    }

    // Compute indices
    int w_out = tid % output_w;
    int h_out = (tid / output_w) % output_h;
    int c = (tid / (output_h * output_w)) % channels;
    int n = tid / (channels * output_h * output_w);

    // Starting position in the input (without padding)
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    float max_val = -INFINITY;

    // Iterate over the kernel window
    for (int kh = 0; kh < kernel_size; ++kh) {
        int h = h_start + kh * dilation;
        if (h < 0 || h >= input_h) continue;

        for (int kw = 0; kw < kernel_size; ++kw) {
            int w = w_start + kw * dilation;
            if (w < 0 || w >= input_w) continue;

            // Compute input offset
            int input_offset = n * (channels * input_h * input_w) +
                              c * (input_h * input_w) +
                              h * input_w +
                              w;

            float val = input[input_offset];
            if (val > max_val) {
                max_val = val;
            }
        }
    }

    // Write to output
    int output_offset = n * (channels * output_h * output_w) +
                        c * (output_h * output_w) +
                        h_out * output_w +
                        w_out;
    output[output_offset] = max_val;
}

torch::Tensor max_pool_2d_cuda_forward(torch::Tensor input, 
                                       int kernel_size, 
                                       int stride, 
                                       int padding, 
                                       int dilation) {
    // Check input dimensions
    if (input.dim() != 4) {
        AT_ERROR("Input tensor must be 4D");
    }

    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);

    // Compute output dimensions
    int output_h = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_w = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Create output tensor
    auto output = torch::empty({batch_size, channels, output_h, output_w}, input.options());

    // Launch kernel
    int block_size = 256;
    int total_threads = batch_size * channels * output_h * output_w;
    int num_blocks = (total_threads + block_size - 1) / block_size;

    dim3 blocks(num_blocks);
    dim3 threads(block_size);

    max_pool_2d_cuda<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_h,
        input_w,
        kernel_size,
        stride,
        padding,
        dilation,
        output_h,
        output_w
    );

    return output;
}
"""

# Compile the CUDA code
max_pool_cuda = load_inline(
    name="max_pool_cuda",
    cpp_sources="",
    cuda_sources=max_pool_cuda_source,
    functions=["max_pool_2d_cuda_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return max_pool_cuda.max_pool_2d_cuda_forward(
            x, self.kernel_size, self.stride, self.padding, self.dilation
        )