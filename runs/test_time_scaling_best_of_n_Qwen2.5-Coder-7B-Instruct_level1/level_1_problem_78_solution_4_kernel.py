import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D transposed convolution
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height_in, int width_in, int height_out, int width_out, int kernel_height, int kernel_width, int stride_h, int stride_w, int pad_h, int pad_w) {
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int h_out = blockIdx.x / (width_out / stride_w);
    int w_out = blockIdx.x % (width_out / stride_w);

    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;

    float sum = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                int h_idx = h_in + kh;
                int w_idx = w_in + kw;
                if (h_idx >= 0 && h_idx < height_in && w_idx >= 0 && w_idx < width_in) {
                    int input_idx = n * in_channels * height_in * width_in + c_in * height_in * width_in + h_idx * width_in + w_idx;
                    int weight_idx = c_out * in_channels * kernel_height * kernel_width + c_in * kernel_height * kernel_width + kh * kernel_width + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = n * out_channels * height_out * width_out + c_out * height_out * width_out + h_out * width_out + w_out;
    output[output_idx] = sum;
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int pad_h, int pad_w) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height_in = input.size(2);
    auto width_in = input.size(3);
    auto height_out = (height_in - 1) * stride_h + 1 - 2 * pad_h;
    auto width_out = (width_in - 1) * stride_w + 1 - 2 * pad_w;
    auto kernel_height = weight.size(2);
    auto kernel_width = weight.size(3);

    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, input.options());

    dim3 grid_size(height_out * width_out, out_channels, batch_size);
    dim3 block_size(1);

    conv_transpose2d_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height_in, width_in, height_out, width_out, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w);

    return output;
}
"""

conv_transpose2d_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int pad_h, int pad_w);"
)

# Compile the inline CUDA code for 2D transposed convolution
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0)):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose2d_cuda(x, self.weight, self.stride[0], self.stride[1], self.padding[0], self.padding[1])


# Test code
batch_size = 8
in_channels = 32
out_channels = 32
kernel_size = (3, 7)
height = 512
width = 1024
stride = (1, 1)
padding = (1, 3)

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]