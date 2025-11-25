import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom fused kernel: Conv2d + element-wise multiplication with scalar + LeakyReLU + GELU
fused_conv_mul_lrelu_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void fused_conv_mul_lrelu_gelu_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const T* __restrict__ multiplier,
    T* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size,
    const float negative_slope,
    const int padding = 1
) {
    const int H_out = height - kernel_size + 1 + 2 * padding;
    const int W_out = width - kernel_size + 1 + 2 * padding;
    const int output_size = batch_size * out_channels * H_out * W_out;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= output_size) return;

    const int w = index % W_out;
    const int h = (index / W_out) % H_out;
    const int c_out = (index / (H_out * W_out)) % out_channels;
    const int n = index / (out_channels * H_out * W_out);

    T sum = static_cast<T>(0);
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int h_in = h + kh - padding;
                const int w_in = w + kw - padding;
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    sum += weight[c_out * in_channels * kernel_size * kernel_size +
                                  c_in * kernel_size * kernel_size + kh * kernel_size + kw] *
                           input[n * in_channels * height * width +
                                 c_in * height * width +
                                 h_in * width + w_in];
                }
            }
        }
    }
    if (bias) sum += bias[c_out];
    sum *= multiplier[c_out];
    T val = sum > static_cast<T>(0) ? sum : sum * static_cast<T>(negative_slope);
    val = val > static_cast<T>(0) ? val : val * (static_cast<T>(0.5) * (1 + torch::erf(val / sqrt(static_cast<T>(2)))));
    output[index] = val;
}

torch::Tensor fused_conv_mul_lrelu_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiplier,
    float negative_slope,
    int kernel_size,
    int padding
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int height = input.size(2);
    const int width = input.size(3);
    const int output_size = batch_size * out_channels * (height - kernel_size + 1 + 2 * padding) * (width - kernel_size + 1 + 2 * padding);

    auto output = torch::empty({batch_size, out_channels, height - kernel_size + 1 + 2 * padding, width - kernel_size + 1 + 2 * padding}, input.options());

    const int threads = 256;
    const int blocks = (output_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_conv_mul_lrelu_gelu_cuda", ([&] {
        fused_conv_mul_lrelu_gelu_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            bias ? bias.data<scalar_t>() : nullptr,
            multiplier.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, in_channels, out_channels, height, width, kernel_size, negative_slope, padding
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

fused_conv_mul_lrelu_gelu_header = (
    "torch::Tensor fused_conv_mul_lrelu_gelu_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "torch::Tensor multiplier, float negative_slope, int kernel_size, int padding);"
)

fused_conv_mul_lrelu_gelu = load_inline(
    name="fused_conv_mul_lrelu_gelu",
    cpp_sources=fused_conv_mul_lrelu_gelu_header,
    cuda_sources=fused_conv_mul_lrelu_gelu_source,
    functions=["fused_conv_mul_lrelu_gelu_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-gencode=arch=compute_80,code=sm_80"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu_negative_slope = nn.Parameter(torch.tensor(0.01), requires_grad=False)  # Default LeakyReLU slope
        self.fused_conv_mul_lrelu_gelu = fused_conv_mul_lrelu_gelu

    def forward(self, x):
        # Extract parameters for kernel fusion
        weight = self.conv.weight
        bias = self.conv.bias
        multiplier = self.multiplier
        kernel_size = self.conv.kernel_size[0]
        negative_slope = self.leaky_relu_negative_slope.item()
        padding = self.conv.padding[0]

        # Execute fused kernel
        return self.fused_conv_mul_lrelu_gelu_cuda(
            x, weight, bias, multiplier, negative_slope, kernel_size, padding
        )

    # Workaround to make state_dict compatible with original model
    def fused_conv_mul_lrelu_gelu_cuda(self, *args):
        return fused_conv_mul_lrelu_gelu.fused_conv_mul_lrelu_gelu_cuda(
            *args
        )