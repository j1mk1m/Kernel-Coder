import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void conv_transpose3d_forward_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    T* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int out_depth,
    int out_height,
    int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * out_depth * out_height * out_width) return;

    int b = idx / (out_channels * out_depth * out_height * out_width);
    int rem = idx % (out_channels * out_depth * out_height * out_width);
    int oc = rem / (out_depth * out_height * out_width);
    rem %= (out_depth * out_height * out_width);
    int od = rem / (out_height * out_width);
    rem %= (out_height * out_width);
    int oh = rem / out_width;
    int ow = rem % out_width;

    T value = 0;
    if (bias != nullptr) {
        value = bias[oc];
    }

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int id = od + padding - kd * dilation;
                    int ih = oh + padding - kh * dilation;
                    int iw = ow + padding - kw * dilation;

                    if (id < 0 || id >= in_depth) continue;
                    if (ih < 0 || ih >= in_height) continue;
                    if (iw < 0 || iw >= in_width) continue;

                    int w_offset = ic * out_channels * kernel_size * kernel_size * kernel_size +
                                   oc * kernel_size * kernel_size * kernel_size +
                                   kd * kernel_size * kernel_size +
                                   kh * kernel_size +
                                   kw;

                    T w = weight[w_offset];

                    int in_offset = b * in_channels * in_depth * in_height * in_width +
                                    ic * in_depth * in_height * in_width +
                                    id * in_height * in_width +
                                    ih * in_width +
                                    iw;

                    value += w * input[in_offset];
                }
            }
        }
    }

    int out_offset = b * out_channels * out_depth * out_height * out_width +
                     oc * out_depth * out_height * out_width +
                     od * out_height * out_width +
                     oh * out_width +
                     ow;

    output[out_offset] = value;
}

torch::Tensor conv_transpose3d_forward(torch::Tensor input,
                                      torch::Tensor weight,
                                      torch::Tensor bias,
                                      int stride,
                                      int padding,
                                      int dilation) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    int kernel_size = weight.size(2);
    int out_channels = weight.size(1);

    int out_depth = (in_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    dim3 threads(256);
    dim3 blocks((output.numel() + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_forward", ([&] {
        conv_transpose3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            bias.defined() ? bias.data<scalar_t>() : nullptr,
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            in_depth,
            in_height,
            in_width,
            kernel_size,
            stride,
            padding,
            dilation,
            out_depth,
            out_height,
            out_width
        );
    }));

    return output;
}
"""

conv_transpose3d_h = """
torch::Tensor conv_transpose3d_forward(torch::Tensor input,
                                      torch::Tensor weight,
                                      torch::Tensor bias,
                                      int stride,
                                      int padding,
                                      int dilation);
"""

# Compile the inline CUDA code
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_h,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.bias is not None else torch.empty(0, device=x.device)
        return conv_transpose3d.conv_transpose3d_forward(x, self.weight, bias, self.stride, self.padding, self.dilation)