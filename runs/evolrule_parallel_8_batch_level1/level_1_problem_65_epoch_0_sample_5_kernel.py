import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding,
    int output_padding,
    int input_height,
    int input_width,
    int output_height,
    int output_width) {

    int n = blockIdx.z / out_channels;
    int c_out = blockIdx.z % out_channels;
    int h_out_start = blockIdx.y * blockDim.y;
    int w_out_start = blockIdx.x * blockDim.x;

    for (int ty = 0; ty < blockDim.y; ++ty) {
        int h_out = h_out_start + ty;
        if (h_out >= output_height) continue;

        for (int tx = 0; tx < blockDim.x; ++tx) {
            int w_out = w_out_start + tx;
            if (w_out >= output_width) continue;

            scalar_t acc = 0.0;

            for (int c_in = 0; c_in < in_channels; ++c_in) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int h_in = (h_out - kh + 2 * padding - output_padding) / stride;
                        int w_in = (w_out - kw + 2 * padding - output_padding) / stride;

                        if (h_in < 0 || h_in >= input_height || w_in < 0 || w_in >= input_width) {
                            continue;
                        }

                        int weight_idx = c_in * out_channels * kernel_h * kernel_w +
                                         c_out * kernel_h * kernel_w +
                                         kh * kernel_w + kw;

                        scalar_t w_val = weight[weight_idx];

                        int input_offset = n * in_channels * input_height * input_width +
                                           c_in * input_height * input_width +
                                           h_in * input_width + w_in;
                        scalar_t in_val = input[input_offset];

                        acc += in_val * w_val;
                    }
                }
            }

            int output_offset = n * out_channels * output_height * output_width +
                                c_out * output_height * output_width +
                                h_out * output_width + w_out;

            output[output_offset] = acc;
        }
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int input_height = input.size(2);
    int input_width = input.size(3);

    int output_height = (input_height - 1) * stride - 2 * padding + kernel_h + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_w + output_padding;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 block(16, 16);
    dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        batch_size * out_channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_cuda", ([&] {
        using scalar_t = scalar_t;
        conv_transpose2d_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
            output_padding,
            input_height,
            input_width,
            output_height,
            output_width
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorName(err)));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose2d_cuda", &conv_transpose2d_cuda, "Custom conv_transpose2d CUDA kernel");
}
"""

conv_transpose2d_cuda = load_inline(
    name="conv_transpose2d_cuda",
    cpp_sources="",
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        kernel_h, kernel_w = kernel_size
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_h, kernel_w))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        output = conv_transpose2d_cuda(x, self.weight, self.stride, self.padding, self.output_padding)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output