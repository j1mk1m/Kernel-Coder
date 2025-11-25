import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_forward_kernel(
    const at::PackedTensorAccessor32<scalar_t,4> input,
    const at::PackedTensorAccessor32<scalar_t,4> weight,
    const at::PackedTensorAccessor32<scalar_t,1> bias,
    at::PackedTensorAccessor32<scalar_t,4> output,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    bool has_bias
) {
    int batch = blockIdx.z / out_channels;
    int out_c = blockIdx.z % out_channels;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (h_out >= output.size(2) || w_out >= output.size(3) || batch >= output.size(0) || out_c >= output.size(1))
        return;

    int in_c = (out_c * in_channels) / out_channels;
    scalar_t sum = 0;

    int input_h_start = h_out * stride - padding;
    int input_w_start = w_out * stride - padding;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int input_h = input_h_start + kh;
            int input_w = input_w_start + kw;

            if (input_h >= 0 && input_h < input.size(2) && input_w >= 0 && input_w < input.size(3)) {
                scalar_t in_val = input[batch][in_c][input_h][input_w];
                scalar_t weight_val = weight[out_c][0][kh][kw];
                sum += in_val * weight_val;
            }
        }
    }

    if (has_bias)
        sum += bias[out_c];

    output[batch][out_c][h_out][w_out] = sum;
}

void depthwise_conv2d_forward_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor output,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    bool has_bias
) {
    AT_ASSERT(input.dim() == 4);
    AT_ASSERT(weight.dim() == 4);
    if (has_bias)
        AT_ASSERT(bias.dim() == 1);

    int output_height = output.size(2);
    int output_width = output.size(3);

    dim3 block(16, 16); 
    dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        input.size(0) * out_channels 
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "depthwise_conv2d_forward_cuda", ([&] {
            depthwise_conv2d_forward_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                input.packed_accessor32<scalar_t,4>(),
                weight.packed_accessor32<scalar_t,4>(),
                bias.packed_accessor32<scalar_t,1>(),
                output.packed_accessor32<scalar_t,4>(),
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                has_bias
            );
        })
    );
}
"""

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(out_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.depthwise_conv2d_forward = load_inline(
            name="depthwise_conv2d_forward",
            cuda_sources=cuda_source,
            functions=["depthwise_conv2d_forward_cuda"],
            verbose=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, input_height, input_width = x.size()
        output_height = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        output = torch.empty(
            batch_size, self.out_channels, output_height, output_width,
            device=x.device, dtype=x.dtype
        )

        has_bias = self.bias is not None

        self.depthwise_conv2d_forward.depthwise_conv2d_forward_cuda(
            x, self.weight, self.bias if has_bias else torch.empty(0, device=x.device),
            output,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            has_bias
        )

        return output