import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

maxpool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Compute output size for a given dimension
int compute_output_size(int input_size, int kernel_size, int stride, int padding, int dilation, bool ceil_mode) {
    int kernel_effective = dilation * (kernel_size - 1) + 1;
    int numerator = input_size + 2 * padding - kernel_effective - 1;
    int div = 0;
    if (ceil_mode) {
        div = (numerator + stride - 1) / stride;
    } else {
        div = numerator / stride;
    }
    return div + 1;
}

__global__ void max_pool3d_kernel(const float* input,
                                  float* output,
                                  int batch_size,
                                  int channels,
                                  int input_depth,
                                  int input_height,
                                  int input_width,
                                  int output_depth,
                                  int output_height,
                                  int output_width,
                                  int kernel_size,
                                  int stride,
                                  int padding,
                                  int dilation,
                                  bool ceil_mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_depth * output_height * output_width) return;

    // Compute indices
    int b = idx / (channels * output_depth * output_height * output_width);
    int remaining = idx % (channels * output_depth * output_height * output_width);
    int c = remaining / (output_depth * output_height * output_width);
    remaining %= (output_depth * output_height * output_width);
    int od = remaining / (output_height * output_width);
    int oh = (remaining % (output_height * output_width)) / output_width;
    int ow = remaining % output_width;

    // Compute input start positions
    int input_d_start = od * stride - padding;
    int input_h_start = oh * stride - padding;
    int input_w_start = ow * stride - padding;

    float max_val = -FLT_MAX;

    for (int kd = 0; kd < kernel_size; ++kd) {
        int input_d = input_d_start + kd * dilation;
        if (input_d < 0 || input_d >= input_depth) continue;
        for (int kh = 0; kh < kernel_size; ++kh) {
            int input_h = input_h_start + kh * dilation;
            if (input_h < 0 || input_h >= input_height) continue;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int input_w = input_w_start + kw * dilation;
                if (input_w < 0 || input_w >= input_width) continue;

                int in_idx = b * channels * input_depth * input_height * input_width +
                             c * input_depth * input_height * input_width +
                             input_d * input_height * input_width +
                             input_h * input_width +
                             input_w;
                float val = input[in_idx];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }

    // Compute output index
    int out_idx = b * channels * output_depth * output_height * output_width +
                  c * output_depth * output_height * output_width +
                  od * output_height * output_width +
                  oh * output_width +
                  ow;
    output[out_idx] = max_val;
}

torch::Tensor max_pool3d_cuda(torch::Tensor input,
                              int kernel_size,
                              int stride,
                              int padding,
                              int dilation,
                              bool return_indices,
                              bool ceil_mode) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);

    int output_depth = compute_output_size(input_depth, kernel_size, stride, padding, dilation, ceil_mode);
    int output_height = compute_output_size(input_height, kernel_size, stride, padding, dilation, ceil_mode);
    int output_width = compute_output_size(input_width, kernel_size, stride, padding, dilation, ceil_mode);

    auto output = torch::empty({batch_size, channels, output_depth, output_height, output_width}, input.options());

    const int threads_per_block = 256;
    int num_elements = batch_size * channels * output_depth * output_height * output_width;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    max_pool3d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}
"""

maxpool3d_cpp_source = "torch::Tensor max_pool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation, bool return_indices, bool ceil_mode);"

max_pool3d = load_inline(
    name="max_pool3d",
    cpp_sources=maxpool3d_cpp_source,
    cuda_sources=maxpool3d_source,
    functions=["max_pool3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.max_pool3d_cuda = max_pool3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.max_pool3d_cuda.max_pool3d_cuda(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices,
            self.ceil_mode
        )
        return output