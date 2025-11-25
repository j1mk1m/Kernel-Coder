import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 64
in_channels = 8
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0
dilation = 1

depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_cuda_kernel(
    const scalar_t* input, const scalar_t* weight, const scalar_t* bias, 
    scalar_t* output, 
    int N, int C, int H, int W, 
    int H_out, int W_out,
    int kernel_size, int stride, int padding, int dilation,
    bool has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H_out * W_out) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c = (idx / (H_out * W_out)) % C;
    int n = idx / (C * H_out * W_out);

    scalar_t sum = 0.0;

    for (int k = 0; k < kernel_size; ++k) {
        int h_in = h_out * stride - padding + k * dilation;
        int w_in = w_out * stride - padding;

        if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) {
            continue;
        }

        int weight_idx = c * kernel_size + k;
        scalar_t w_val = weight[weight_idx];

        int input_offset = n * C * H * W + c * H * W + h_in * W + w_in;
        scalar_t in_val = input[input_offset];

        sum += in_val * w_val;
    }

    if (has_bias) {
        sum += bias[c];
    }

    int output_offset = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out;
    output[output_offset] = sum;
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size, int stride, int padding, int dilation, bool has_bias) {
    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);

    // Compute output dimensions
    int numerator_h = H + 2 * padding - dilation * (kernel_size -1) -1;
    int H_out = (numerator_h / stride) + 1;
    int numerator_w = W + 2 * padding -1;
    int W_out = (numerator_w / stride) +1;

    auto output = torch::zeros({N, C, H_out, W_out}, input.options());

    const int threads_per_block = 256;
    int num_elements = N * C * H_out * W_out;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    depthwise_conv2d_cuda_kernel<float><<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C, H, W,
        H_out, W_out,
        kernel_size, stride, padding, dilation,
        has_bias
    );

    cudaDeviceSynchronize();
    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size, int stride, int padding, int dilation, bool has_bias);
"""

depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(kernel_size, 1),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias
        )
        self.depthwise_conv2d = depthwise_conv2d

    def forward(self, x):
        weight = self.conv2d.weight.contiguous()
        bias = self.conv2d.bias.contiguous() if self.conv2d.bias is not None else None
        has_bias = self.conv2d.bias is not None

        return self.depthwise_conv2d.depthwise_conv2d_cuda(
            x.contiguous(),
            weight,
            bias,
            kernel_size=self.conv2d.kernel_size[0],
            stride=self.conv2d.stride[0],
            padding=self.conv2d.padding[0],
            dilation=self.conv2d.dilation[0],
            has_bias=has_bias
        )

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding, dilation]