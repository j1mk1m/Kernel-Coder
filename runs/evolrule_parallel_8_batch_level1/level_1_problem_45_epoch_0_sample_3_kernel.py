import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool_2d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int output_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_elements) return;

    int b = index / (channels * output_height * output_width);
    int remaining = index % (channels * output_height * output_width);
    int c = remaining / (output_height * output_width);
    remaining = remaining % (output_height * output_width);
    int ox = remaining / output_width;
    int oy = remaining % output_width;

    int start_x = ox * stride - padding;
    int start_y = oy * stride - padding;

    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int input_x = start_x + i;
            int input_y = start_y + j;
            if (input_x >= 0 && input_x < input_height && input_y >= 0 && input_y < input_width) {
                int input_offset = b * channels * input_height * input_width +
                                   c * input_height * input_width +
                                   input_x * input_width +
                                   input_y;
                sum += input[input_offset];
            }
        }
    }

    int output_offset = b * channels * output_height * output_width +
                        c * output_height * output_width +
                        ox * output_width +
                        oy;
    output[output_offset] = sum / (kernel_size * kernel_size);
}

torch::Tensor avg_pool_2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, channels, output_height, output_width}, input.options());

    int output_elements = batch_size * channels * output_height * output_width;
    int threads_per_block = 256;
    int blocks_per_grid = (output_elements + threads_per_block - 1) / threads_per_block;

    avg_pool_2d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        output_elements
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    return output;
}
"""

avg_pool_cpp_source = (
    "torch::Tensor avg_pool_2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"
)

avg_pool_cuda = load_inline(
    name="avg_pool_cuda",
    cuda_sources=avg_pool_source,
    cpp_sources=avg_pool_cpp_source,
    functions=["avg_pool_2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avg_pool_cuda.avg_pool_2d_cuda(x, self.kernel_size, self.stride, self.padding)