import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused transposed convolution, bias addition, clamping, and scaling
fused_conv_transpose_bias_clamp_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_bias_clamp_scale_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float scaling_factor
) {
    int batch_idx = blockIdx.x / (out_height * out_width);
    int out_h_idx = (blockIdx.x % (out_height * out_width)) / out_width;
    int out_w_idx = blockIdx.x % out_width;
    int out_channel_idx = blockIdx.y;
    int in_channel_idx = threadIdx.x;

    if (batch_idx >= batch_size || out_h_idx >= out_height || out_w_idx >= out_width || out_channel_idx >= out_channels || in_channel_idx >= in_channels) {
        return;
    }
    
    float sum = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int in_h = out_h_idx * stride - padding + kh;
            int in_w = out_w_idx * stride - padding + kw;

            if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                int input_idx = batch_idx * in_channels * in_height * in_width +
                                in_channel_idx * in_height * in_width +
                                in_h * in_width +
                                in_w;

                int weight_idx = out_channel_idx * in_channels * kernel_size * kernel_size +
                                in_channel_idx * kernel_size * kernel_size +
                                kh * kernel_size +
                                kw;

                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    // Add bias
    sum += bias[out_channel_idx];

    // Clamp, Scale, Clamp, Divide
    sum = fmaxf(0.0f, fminf(1.0f, sum));
    sum *= scaling_factor;
    sum = fmaxf(0.0f, fminf(1.0f, sum));
    sum /= scaling_factor;

    int output_idx = batch_idx * out_channels * out_height * out_width +
                        out_channel_idx * out_height * out_width +
                        out_h_idx * out_width +
                        out_w_idx;

    output[output_idx] = sum;
}

torch::Tensor fused_conv_transpose_bias_clamp_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int output_padding,
    float scaling_factor
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    dim3 blocks(batch_size * out_height * out_width, out_channels);
    dim3 threads(in_channels);

    fused_conv_transpose_bias_clamp_scale_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        scaling_factor
    );

    return output;
}
"""

fused_conv_transpose_bias_clamp_scale_cpp_source = """
torch::Tensor fused_conv_transpose_bias_clamp_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int output_padding,
    float scaling_factor
);
"""

fused_conv_transpose_bias_clamp_scale = load_inline(
    name="fused_conv_transpose_bias_clamp_scale",
    cpp_sources=fused_conv_transpose_bias_clamp_scale_cpp_source,
    cuda_sources=fused_conv_transpose_bias_clamp_scale_source,
    functions=["fused_conv_transpose_bias_clamp_scale_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scaling_factor = scaling_factor

    def forward(self, x):
        out_height = (x.size(2) - 1) * self.stride - 2 * self.padding + self.weight.size(2) + self.output_padding
        out_width = (x.size(3) - 1) * self.stride - 2 * self.padding + self.weight.size(3) + self.output_padding
        return fused_conv_transpose_bias_clamp_scale.fused_conv_transpose_bias_clamp_scale_cuda(
            x, 
            self.weight, 
            self.bias, 
            out_height, 
            out_width,
            self.stride,
            self.padding,
            self.output_padding,
            self.scaling_factor
        )