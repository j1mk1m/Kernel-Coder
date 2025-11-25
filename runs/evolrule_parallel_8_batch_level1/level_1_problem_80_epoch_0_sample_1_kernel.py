import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * H_out * W_out) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int n = idx / (W_out * H_out * C_out);

    float acc = 0.0;
    for (int kh = 0; kh < kH; ++kh) {
        for (int kw = 0; kw < kW; ++kw) {
            for (int c_in = 0; c_in < C_in; ++c_in) {
                int h_in = h_out * stride_h - padding_h + kh * dilation_h;
                int w_in = w_out * stride_w - padding_w + kw * dilation_w;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    acc += input[ n * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in ] *
                           weight[ c_out * C_in * kH * kW + c_in * kH * kW + kh * kW + kw ];
                }
            }
        }
    }
    if (bias != nullptr) {
        acc += bias[c_out];
    }
    output[idx] = acc;
}

torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::tuple<int, int> stride,
    std::tuple<int, int> padding,
    std::tuple<int, int> dilation,
    int H_out,
    int W_out
) {
    int stride_h = std::get<0>(stride);
    int stride_w = std::get<1>(stride);
    int padding_h = std::get<0>(padding);
    int padding_w = std::get<1>(padding);
    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(0);
    int kH = weight.size(2);
    int kW = weight.size(3);

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty({N, C_out, H_out, W_out}, options);

    int num_elements = N * C_out * H_out * W_out;
    const int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, kH, kW,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        H_out, W_out
    );

    return output;
}
"""

conv2d_cpp_source = (
    "torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, std::tuple<int, int> stride, std::tuple<int, int> padding, std::tuple<int, int> dilation, int H_out, int W_out);"
)

conv2d = load_inline(
    name="conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias:
            self.bias_param = nn.Parameter(torch.Tensor(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)
        else:
            self.bias_param = None

        # Assign the compiled CUDA function
        self.conv2d_cuda = conv2d.conv2d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move input to the same device as the model's parameters
        x = x.to(self.weight.device)

        # Extract dimensions
        N, C_in, H_in, W_in = x.shape
        kH, kW = self.kernel_size
        stride_h, stride_w = (self.stride, self.stride)
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation

        # Calculate output dimensions
        H_out = int((H_in + 2 * padding_h - dilation_h * (kH - 1) - 1) // stride_h + 1)
        W_out = int((W_in + 2 * padding_w - dilation_w * (kW - 1) - 1) // stride_w + 1)

        # Prepare the bias tensor (empty if no bias)
        bias_tensor = self.bias_param if self.bias else torch.empty(0, device=x.device)

        # Call the CUDA kernel
        output = self.conv2d_cuda(
            x,
            self.weight,
            bias_tensor,
            (stride_h, stride_w),
            (padding_h, padding_w),
            (dilation_h, dilation_w),
            H_out,
            W_out
        )

        return output