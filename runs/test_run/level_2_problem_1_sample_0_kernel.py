import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + ReLU + Bias
fused_conv_relu_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void fused_conv_relu_bias_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size
) {
    int out_channel = blockIdx.x;
    int batch = blockIdx.y;
    int out_y = threadIdx.x;
    int out_x = threadIdx.y;

    if (out_channel < out_channels && batch < batch_size && out_y < in_height - kernel_size + 1 && out_x < in_width - kernel_size + 1) {
        float sum = 0.0f;
        for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
            for (int kernel_y = 0; kernel_y < kernel_size; ++kernel_y) {
                for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x) {
                    int input_y = out_y + kernel_y;
                    int input_x = out_x + kernel_x;
                    int input_index = batch * in_channels * in_height * in_width +
                                      in_channel * in_height * in_width +
                                      input_y * in_width +
                                      input_x;
                    int weight_index = out_channel * in_channels * kernel_size * kernel_size +
                                       in_channel * kernel_size * kernel_size +
                                       kernel_y * kernel_size +
                                       kernel_x;
                    sum += input[input_index] * weight[weight_index];
                }
            }
        }

        float biased_sum = sum + bias[out_channel];
        output[batch * out_channels * (in_height - kernel_size + 1) * (in_width - kernel_size + 1) +
               out_channel * (in_height - kernel_size + 1) * (in_width - kernel_size + 1) +
               out_y * (in_width - kernel_size + 1) +
               out_x] = (biased_sum > 0.0f) ? biased_sum : 0.0f; // ReLU
    }
}

torch::Tensor fused_conv_relu_bias_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size
) {
    auto output = torch::zeros({batch_size, out_channels, in_height - kernel_size + 1, in_width - kernel_size + 1},
                               torch::kFloat).cuda();

    dim3 threads(in_height - kernel_size + 1, in_width - kernel_size + 1);
    dim3 blocks(out_channels, batch_size);

    fused_conv_relu_bias_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_size
    );

    return output;
}
"""

fused_conv_relu_bias_cpp_source = """
torch::Tensor fused_conv_relu_bias_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size
);
"""

fused_conv_relu_bias = load_inline(
    name="fused_conv_relu_bias",
    cpp_sources=fused_conv_relu_bias_cpp_source,
    cuda_sources=fused_conv_relu_bias_source,
    functions=["fused_conv_relu_bias_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        batch_size = x.size(0)
        in_height = x.size(2)
        in_width = x.size(3)
        return fused_conv_relu_bias.fused_conv_relu_bias_cuda(
            x,
            self.weight,
            self.bias,
            batch_size,
            self.in_channels,
            self.out_channels,
            in_height,
            in_width,
            self.kernel_size
        )