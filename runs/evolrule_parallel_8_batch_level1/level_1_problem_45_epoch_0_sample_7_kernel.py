import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool2d_kernel(const float* input, float* output, int batch_size, int channels, int input_height, int input_width, int output_height, int output_width, int kernel_size, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_height * output_width) return;

    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int c = (idx / (output_width * output_height)) % channels;
    int n = idx / (output_width * output_height * channels);

    int h_start = h_out * stride;
    int w_start = w_out * stride;

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        int h_padded = h_start + kh;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int w_padded = w_start + kw;

            if (h_padded < padding || h_padded >= padding + input_height) continue;
            if (w_padded < padding || w_padded >= padding + input_width) continue;

            int h = h_padded - padding;
            int w = w_padded - padding;

            int input_offset = n * channels * input_height * input_width + c * input_height * input_width + h * input_width + w;
            sum += input[input_offset];
        }
    }

    sum /= (kernel_size * kernel_size);

    int output_offset = n * channels * output_height * output_width + c * output_height * output_width + h_out * output_width + w_out;
    output[output_offset] = sum;
}

torch::Tensor avg_pool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, channels, output_height, output_width}, input.options());

    const int threads_per_block = 256;
    int num_elements = batch_size * channels * output_height * output_width;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    avg_pool2d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, input_height, input_width,
        output_height, output_width,
        kernel_size, stride, padding
    );

    return output;
}
"""

avg_pool_cpp_header = """
#include <torch/extension.h>
torch::Tensor avg_pool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);
"""

avg_pool = load_inline(
    name="avg_pool",
    cpp_sources=avg_pool_cpp_header,
    cuda_sources=avg_pool_source,
    functions=["avg_pool2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avg_pool.avg_pool2d_cuda(x, self.kernel_size, self.stride, self.padding)