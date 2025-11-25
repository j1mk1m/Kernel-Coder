import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2d
conv_transpose_2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(const scalar_t* __restrict__ input,
                                       const scalar_t* __restrict__ weight,
                                       scalar_t* __restrict__ output,
                                       int batch_size, int in_channels, int out_channels,
                                       int kernel_h, int kernel_w,
                                       int input_h, int input_w,
                                       int output_h, int output_w,
                                       int stride, int padding,
                                       int output_padding) {

    int batch = blockIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.z * blockDim.x + threadIdx.x;

    if (out_y >= output_h || out_x >= output_w) return;

    for (int out_ch = threadIdx.y; out_ch < out_channels; out_ch += blockDim.y) {
        scalar_t sum = 0;
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int input_y = (out_y - kh + padding - output_padding) / stride;
                    int input_x = (out_x - kw + padding - output_padding) / stride;

                    if ((out_y - kh + padding - output_padding) % stride != 0 ||
                        (out_x - kw + padding - output_padding) % stride != 0) {
                        continue;
                    }

                    if (input_y < 0 || input_y >= input_h || input_x < 0 || input_x >= input_w) {
                        continue;
                    }

                    int weight_idx = (in_ch * out_channels + out_ch) * kernel_h * kernel_w + kh * kernel_w + kw;
                    int input_idx = batch * in_channels * input_h * input_w + in_ch * input_h * input_w + input_y * input_w + input_x;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
        output[batch * out_channels * output_h * output_w + out_ch * output_h * output_w + out_y * output_w + out_x] = sum;
    }
}

std::tuple<torch::Tensor> conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight,
                                               int stride, int padding, int output_padding) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0) / (in_channels * kernel_h * kernel_w);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int input_h = input.size(2);
    const int input_w = input.size(3);

    int output_h = (input_h - 1) * stride - 2 * padding + kernel_h + output_padding;
    int output_w = (input_w - 1) * stride - 2 * padding + kernel_w + output_padding;

    torch::Tensor output = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options());

    const int threads = 256;
    dim3 blocks(batch_size, 1, 1);
    dim3 threadsPerBlock(32, 8);

    conv_transpose2d_kernel<float><<<blocks, threadsPerBlock>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        kernel_h, kernel_w,
        input_h, input_w,
        output_h, output_w,
        stride, padding, output_padding);

    return output;
}
"""

# Compile the CUDA kernel
conv_transpose_2d = load_inline(
    name='conv_transpose_2d',
    cpp_sources='',
    cuda_sources=conv_transpose_2d_source,
    functions=['conv_transpose2d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        output = conv_transpose_2d.conv_transpose2d_cuda(x, self.weight, self.stride, self.padding, self.output_padding)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output

def get_inputs():
    x = torch.rand(batch_size, in_channels, height_in, width_in).cuda()
    return [x.cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization