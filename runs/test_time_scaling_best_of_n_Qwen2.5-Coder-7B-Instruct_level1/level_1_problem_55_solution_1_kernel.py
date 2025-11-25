import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D convolution
convolution_2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_2d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int height, int width, int out_channels, int kernel_size) {
    int n = blockIdx.x; // batch index
    int c_out = blockIdx.y; // output channel index
    int h_out = blockIdx.z; // output height index
    int w_out = blockIdx.w; // output width index

    float sum = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_out * 2 - kh;
                int w_in = w_out * 2 - kw;
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    sum += input[n * in_channels * height * width + c_in * height * width + h_in * width + w_in] *
                           weight[c_out * in_channels * kernel_size * kernel_size + c_in * kernel_size * kernel_size + kh * kernel_size + kw];
                }
            }
        }
    }
    output[n * out_channels * height * width + c_out * height * width + h_out * width + w_out] = sum;
}

torch::Tensor convolution_2d_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    dim3 blocks_per_grid((width + 7) / 8, (height + 7) / 8, (out_channels + 7) / 8, (batch_size + 7) / 8);
    dim3 threads_per_block(8, 8, 1);

    convolution_2d_kernel<<<blocks_per_grid, threads_per_block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, height, width, out_channels, kernel_size);

    return output;
}
"""

convolution_2d_cpp_source = (
    "torch::Tensor convolution_2d_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for 2D convolution
convolution_2d = load_inline(
    name="convolution_2d",
    cpp_sources=convolution_2d_cpp_source,
    cuda_sources=convolution_2d_source,
    functions=["convolution_2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = convolution_2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d.convolution_2d_cuda(x, self.weight)