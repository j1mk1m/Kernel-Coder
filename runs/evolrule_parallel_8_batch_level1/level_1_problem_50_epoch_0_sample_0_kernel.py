import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

custom_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)

__global__ void custom_conv(
    const float* __restrict__ input_padded,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int C_in,
    int H_padded,
    int W_padded,
    int K_h,
    int K_w,
    int stride,
    int out_channels,
    int H_out,
    int W_out
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= batch_size * out_channels * H_out * W_out) return;

    int b = index / (out_channels * H_out * W_out);
    int remainder = index % (out_channels * H_out * W_out);
    int o = remainder / (H_out * W_out);
    int pos = remainder % (H_out * W_out);
    int oh = pos / W_out;
    int ow = pos % W_out;

    float sum = 0.0f;

    for (int kh = 0; kh < K_h; ++kh) {
        for (int kw = 0; kw < K_w; ++kw) {
            for (int ic = 0; ic < C_in; ++ic) {
                int ih = oh * stride + kh;
                int iw = ow * stride + kw;

                int input_offset = b * C_in * H_padded * W_padded
                                  + ic * H_padded * W_padded
                                  + ih * W_padded + iw;
                float in_val = input_padded[input_offset];

                int weight_offset = o * C_in * K_h * K_w
                                   + ic * K_h * K_w
                                   + kh * K_w + kw;
                float weight_val = weights[weight_offset];

                sum += in_val * weight_val;
            }
        }
    }

    sum += bias[o];

    int output_offset = b * out_channels * H_out * W_out
                      + o * H_out * W_out
                      + oh * W_out + ow;
    output[output_offset] = sum;
}

torch::Tensor custom_conv_cuda(
    torch::Tensor input_padded,
    torch::Tensor weights,
    torch::Tensor bias
) {
    int batch_size = input_padded.size(0);
    int C_in = input_padded.size(1);
    int H_padded = input_padded.size(2);
    int W_padded = input_padded.size(3);

    int K_h = weights.size(2);
    int K_w = weights.size(3);
    int out_channels = weights.size(0);
    int stride = 4;
    int H_out = (H_padded - K_h) / stride + 1;
    int W_out = (W_padded - K_w) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, H_out, W_out}, input_padded.options());

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid( (output.numel() + threadsPerBlock.x -1) / threadsPerBlock.x );

    custom_conv<<<blocksPerGrid, threadsPerBlock>>>(
        input_padded.data_ptr<float>(),
        weights.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        C_in,
        H_padded,
        W_padded,
        K_h,
        K_w,
        stride,
        out_channels,
        H_out,
        W_out
    );

    return output;
}
"""

custom_conv_cpp_source = """
torch::Tensor custom_conv_cuda(
    torch::Tensor input_padded,
    torch::Tensor weights,
    torch::Tensor bias
);
"""

custom_conv = load_inline(
    name="custom_conv",
    cpp_sources=custom_conv_cpp_source,
    cuda_sources=custom_conv_source,
    functions=["custom_conv_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(96, 3, 11, 11))
        self.bias = nn.Parameter(torch.empty(96))
        # Initialize weights and bias similarly to PyTorch's default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.custom_conv = custom_conv

    def forward(self, x):
        padded_x = F.pad(x, (2, 2, 2, 2), 'constant', 0)
        return self.custom_conv.custom_conv_cuda(padded_x, self.weight, self.bias)