import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

custom_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_conv2d(
    const float* input, const float* weights, float* output,
    int batch_size, int in_channels, int out_channels,
    int kernel_h, int kernel_w, int oh, int ow,
    int input_height, int input_width) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= batch_size * out_channels * oh * ow)
        return;

    // Unpack indices
    int b = tid / (out_channels * oh * ow);
    int rem = tid % (out_channels * oh * ow);
    int oc = rem / (oh * ow);
    rem %= (oh * ow);
    int y = rem / ow;
    int x = rem % ow;

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int ky = 0; ky < kernel_h; ++ky) {
            for (int kx = 0; kx < kernel_w; ++kx) {
                int iy = y + ky;
                int ix = x + kx;

                // Calculate input index
                int input_idx = b * in_channels * input_height * input_width +
                                ic * input_height * input_width +
                                iy * input_width + ix;

                // Calculate weight index
                int weight_idx = oc * in_channels * kernel_h * kernel_w +
                                 ic * kernel_h * kernel_w +
                                 ky * kernel_w + kx;

                sum += input[input_idx] * weights[weight_idx];
            }
        }
    }

    // Calculate output index
    int output_idx = b * out_channels * oh * ow +
                     oc * oh * ow +
                     y * ow + x;
    output[output_idx] = sum;
}

torch::Tensor custom_conv2d_cuda(torch::Tensor input, torch::Tensor weights) {
    // Check dimensions
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);

    auto out_channels = weights.size(0);
    auto kernel_h = weights.size(2);
    auto kernel_w = weights.size(3);

    // Compute output dimensions
    auto oh = input_height - kernel_h + 1;
    auto ow = input_width - kernel_w + 1;

    // Output tensor
    auto output = torch::zeros({batch_size, out_channels, oh, ow}, input.options());

    // Launch kernel
    int total_threads = batch_size * out_channels * oh * ow;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;

    custom_conv2d<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        kernel_h, kernel_w, oh, ow,
        input_height, input_width
    );

    return output;
}
"""

custom_conv2d_cpp_source = "torch::Tensor custom_conv2d_cuda(torch::Tensor input, torch::Tensor weights);"

# Compile the inline CUDA code for the custom convolution
custom_conv = load_inline(
    name="custom_conv",
    cpp_sources=custom_conv2d_cpp_source,
    cuda_sources=custom_conv2d_source,
    functions=["custom_conv2d_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, **kwargs):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.custom_conv = custom_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.conv2d.weight
        return self.custom_conv.custom_conv2d_cuda(x, weights)