import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 2D convolution
transposed_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void transposed_conv2d_kernel(float* x, float* weight, float* bias, float* out, int batch_size, int in_channels, int out_channels, int kernel_size, int stride, int padding, int output_padding, int groups) {
    int batch_id = blockIdx.z;
    int out_ch_id = blockIdx.y * blockDim.y + threadIdx.y;
    int in_ch_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_ch_id >= out_channels || in_ch_id >= in_channels) {
        return;
    }

    out[batch_id * out_channels * (output_height + output_padding) * (output_width + output_padding) + out_ch_id * (output_height + output_padding) * (output_width + output_padding)] = 0;

    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int in_idx = (batch_id * in_channels * height * width +
                          (in_ch_id / groups) * (height * width) +
                          ((i + padding) * width + (j + padding)));
            int out_idx = (batch_id * out_channels * (output_height + output_padding) * (output_width + output_padding) +
                           out_ch_id * (output_height + output_padding) * (output_width + output_padding) +
                           ((i + padding - stride) * (output_width + output_padding) + (j + padding - stride)));

            if (in_idx >= 0 && in_idx < batch_size * in_channels * height * width &&
                out_idx >= 0 && out_idx < batch_size * out_channels * (output_height + output_padding) * (output_width + output_padding)) {
                out[out_idx] += x[in_idx] * weight[out_ch_id * kernel_size * kernel_size + in_ch_id * kernel_size * kernel_size + i * kernel_size + j];
            }
        }
    }

    if (bias != nullptr) {
        out[batch_id * out_channels * (output_height + output_padding) * (output_width + output_padding) + out_ch_id * (output_height + output_padding) * (output_width + output_padding)] += bias[out_ch_id];
    }
}

torch::Tensor transposed_conv2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int output_padding, int groups) {
    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    auto height = x.size(2);
    auto width = x.size(3);
    auto output_height = (height - 1) * stride - 2 * padding + kernel_size + output_padding;
    auto output_width = (width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto out = torch::zeros({batch_size, out_channels, output_height, output_width}, x.options());

    const int block_size = 32;
    const int grid_x = (in_channels + block_size - 1) / block_size;
    const int grid_y = (out_channels + block_size - 1) / block_size;
    const int grid_z = batch_size;

    transposed_conv2d_kernel<<<grid_z, dim3(grid_x, grid_y), 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(),
        batch_size, in_channels, out_channels, kernel_size, stride, padding, output_padding, groups);

    return out;
}
"""

transposed_conv2d_cpp_source = (
    "torch::Tensor transposed_conv2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int output_padding, int groups);"
)

# Compile the inline CUDA code for transposed 2D convolution
transposed_conv2d = load_inline(
    name="transposed_conv2d",
    cpp_sources=transposed_conv2d_cpp_source,
    cuda_sources=transposed_conv2d_source,
    functions=["transposed_conv2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return transposed_conv2d_cuda(x, self.weight, self.bias, stride=self.stride, padding=self.padding, output_padding=self.output_padding, groups=self.groups)


# Example usage
model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias)
inputs = get_inputs()[0]
outputs = model_new(inputs)
print(outputs.shape)