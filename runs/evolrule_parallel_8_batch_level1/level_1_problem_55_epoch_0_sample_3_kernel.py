import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 2D convolution optimized for 3x3 kernel, small batch, and asymmetric input dimensions
custom_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define TILE_HEIGHT 16

__global__ void custom_conv2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int output_h,
    int output_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w
) {
    extern __shared__ float shared_input[];

    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int batch = blockIdx.z;

    int output_x_start = block_x * TILE_WIDTH;
    int output_y_start = block_y * TILE_HEIGHT;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int output_x = output_x_start + tx;
    int output_y = output_y_start + ty;

    if (output_x >= output_w || output_y >= output_h) {
        return;
    }

    const int input_tile_h = TILE_HEIGHT + kernel_h - 1;
    const int input_tile_w = TILE_WIDTH + kernel_w - 1;

    int in_tile_x = tx;
    int in_tile_y = ty;

    if (in_tile_x >= input_tile_w || in_tile_y >= input_tile_h) {
        return;
    }

    int input_x_start = output_x_start - padding_w;
    int input_y_start = output_y_start - padding_h;

    int in_x = input_x_start + in_tile_x;
    int in_y = input_y_start + in_tile_y;

    if (in_x < 0 || in_x >= input_w || in_y < 0 || in_y >= input_h) {
        return;
    }

    for (int c = 0; c < in_channels; c++) {
        int input_offset = batch * in_channels * input_h * input_w +
                           c * input_h * input_w +
                           in_y * input_w + in_x;

        shared_input[ (in_tile_y * input_tile_w + in_tile_x) * in_channels + c ] = input[input_offset];
    }

    __syncthreads();

    for (int c_out = 0; c_out < out_channels; c_out++) {
        float acc = 0.0f;
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int input_ih = output_y - output_y_start + kh;
                    int input_iw = output_x - output_x_start + kw;
                    int shared_offset = (input_ih * input_tile_w + input_iw) * in_channels + c_in;
                    float input_val = shared_input[shared_offset];

                    int weight_offset = c_out * in_channels * kernel_h * kernel_w +
                                        c_in * kernel_h * kernel_w +
                                        kh * kernel_w + kw;
                    float weight_val = weight[weight_offset];

                    acc += input_val * weight_val;
                }
            }
        }

        int output_offset = batch * out_channels * output_h * output_w +
                            c_out * output_h * output_w +
                            output_y * output_w + output_x;

        output[output_offset] = acc;
    }
}

torch::Tensor custom_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);

    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels = weight.size(0);

    int output_h = (input_h + 2 * padding_h - (kernel_h - 1) - 1) / stride_h + 1;
    int output_w = (input_w + 2 * padding_w - (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    const int tile_h = TILE_HEIGHT + kernel_h - 1;
    const int tile_w = TILE_WIDTH + kernel_w - 1;

    dim3 blockDim(tile_w, tile_h);
    dim3 gridDim(
        (output_w + TILE_WIDTH - 1) / TILE_WIDTH,
        (output_h + TILE_HEIGHT - 1) / TILE_HEIGHT,
        batch_size
    );

    int shared_size = tile_h * tile_w * in_channels * sizeof(float);

    custom_conv2d_kernel<<<gridDim, blockDim, shared_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_h,
        input_w,
        output_h,
        output_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w
    );

    return output;
}
"""

custom_conv_cpp_source = """
torch::Tensor custom_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w
);
"""

custom_conv = load_inline(
    name="custom_conv",
    cpp_sources=custom_conv_cpp_source,
    cuda_sources=custom_conv_source,
    functions=["custom_conv2d"],
    verbose=True,
)

class CustomConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding):
        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        ctx.padding = padding
        return custom_conv.custom_conv2d(input, weight, stride, stride, padding, padding)

    @staticmethod
    def backward(ctx, grad_output):
        # Placeholder for backward implementation (requires custom kernels)
        input, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        return None, None, None, None  # Actual implementation required

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x):
        out = CustomConv2dFunction.apply(x, self.weight, self.stride, self.padding)
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        return out