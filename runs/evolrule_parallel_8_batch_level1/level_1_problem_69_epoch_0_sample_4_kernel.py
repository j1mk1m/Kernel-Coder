import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* kernel,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int dilation_h,
    int dilation_w,
    int input_h,
    int input_w,
    int output_h,
    int output_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_h * output_w)
        return;

    int ow = idx % output_w;
    int oh = (idx / output_w) % output_h;
    int oc = (idx / (output_h * output_w)) % out_channels;
    int b = idx / (out_channels * output_h * output_w);

    float output_val = 0.0;

    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int ih = (oh + 2 * padding_h - kh * dilation_h - output_padding_h) / stride_h;
                int iw = (ow + 2 * padding_w - kw * dilation_w - output_padding_w) / stride_w;

                if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                    int kernel_offset = oc * in_channels * kernel_h * kernel_w +
                                       ic * kernel_h * kernel_w +
                                       kh * kernel_w + kw;
                    float k_val = kernel[kernel_offset];

                    int input_offset = b * in_channels * input_h * input_w +
                                      ic * input_h * input_w +
                                      ih * input_w + iw;
                    float in_val = input[input_offset];

                    output_val += in_val * k_val;
                }
            }
        }
    }

    int output_offset = b * out_channels * output_h * output_w +
                       oc * output_h * output_w +
                       oh * output_w + ow;
    output[output_offset] = output_val;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor kernel,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int dilation_h,
    int dilation_w
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int out_channels = kernel.size(0);
    int kernel_h = kernel.size(2);
    int kernel_w = kernel.size(3);

    int output_h = (input_h - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int output_w = (input_w - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    const int block_size = 256;
    int num_threads = batch_size * out_channels * output_h * output_w;
    int num_blocks = (num_threads + block_size - 1) / block_size;

    conv_transpose2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        dilation_h,
        dilation_w,
        input_h,
        input_w,
        output_h,
        output_w
    );

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose2d_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor kernel, int stride_h, int stride_w, int padding_h, int padding_w, int output_padding_h, int output_padding_w, int dilation_h, int dilation_w);"
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # Initialize weights and bias like PyTorch's ConvTranspose2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Compile the CUDA kernel
        self.conv_transpose2d = load_inline(
            name="conv_transpose2d",
            cpp_sources=conv_transpose2d_cpp_source,
            cuda_sources=conv_transpose2d_source,
            functions=["conv_transpose2d_cuda"],
            verbose=True
        )

    def forward(self, x):
        batch_size, _, input_h, input_w = x.size()
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        output_padding_h, output_padding_w = self.output_padding
        dilation_h, dilation_w = self.dilation

        # Call the CUDA kernel
        output = self.conv_transpose2d.conv_transpose2d_cuda(
            x,
            self.weight,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_padding_h,
            output_padding_w,
            dilation_h,
            dilation_w
        )

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        return output