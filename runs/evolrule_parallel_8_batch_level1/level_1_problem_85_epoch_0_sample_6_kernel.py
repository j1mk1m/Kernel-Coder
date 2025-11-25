import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# CUDA kernel source code
depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* kernel_weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    bool has_bias
) {
    int batch = blockIdx.x;
    int channel = blockIdx.y;
    int tile_idx = blockIdx.z;

    const int block_h = blockDim.x;
    const int block_w = blockDim.y;

    const int num_tiles_h = (H_out + block_h - 1) / block_h;
    const int num_tiles_w = (W_out + block_w - 1) / block_w;

    const int tile_h = tile_idx / num_tiles_w;
    const int tile_w = tile_idx % num_tiles_w;

    const int x_start = tile_h * block_h;
    const int y_start = tile_w * block_w;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = x_start + tx;
    const int y = y_start + ty;

    if (x < H_out && y < W_out) {
        float acc = 0.0;

        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int input_x = x * stride_h - padding_h + kh * dilation_h;
                const int input_y = y * stride_w - padding_w + kw * dilation_w;

                if (input_x >= 0 && input_x < H_in &&
                    input_y >= 0 && input_y < W_in) {
                    const int input_offset = 
                        batch * in_channels * H_in * W_in +
                        channel * H_in * W_in +
                        input_x * W_in +
                        input_y;

                    const float input_val = input[input_offset];

                    const int kernel_offset = 
                        channel * kernel_h * kernel_w + 
                        kh * kernel_w + kw;

                    const float weight_val = kernel_weights[kernel_offset];

                    acc += input_val * weight_val;
                }
            }
        }

        if (has_bias) {
            acc += bias[channel];
        }

        const int output_offset = 
            batch * in_channels * H_out * W_out +
            channel * H_out * W_out +
            x * W_out +
            y;

        output[output_offset] = acc;
    }
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor kernel_weights,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    bool has_bias
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    const int kernel_h = kernel_weights.size(2);
    const int kernel_w = kernel_weights.size(3);

    const int H_out = (H_in + 2 * padding_h - dilation_h * (kernel_h - 1)) / stride_h + 1;
    const int W_out = (W_in + 2 * padding_w - dilation_w * (kernel_w - 1)) / stride_w + 1;

    auto output = torch::empty({batch_size, in_channels, H_out, W_out}, input.options());

    const int block_h = 16;
    const int block_w = 16;
    dim3 block(block_h, block_w);

    const int num_tiles_h = (H_out + block_h - 1) / block_h;
    const int num_tiles_w = (W_out + block_w - 1) / block_w;
    const int num_tiles = num_tiles_h * num_tiles_w;

    dim3 grid(batch_size, in_channels, num_tiles);

    depthwise_conv2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        kernel_weights.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        H_in,
        W_in,
        H_out,
        W_out,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        has_bias
    );

    return output;
}
"""

depthwise_conv_cpp = """
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor kernel_weights,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    bool has_bias
);
"""

depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=depthwise_conv_cpp,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, 
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = in_channels  # Force groups to in_channels for depthwise
        self.bias = bias

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(
            out_channels, 1, kernel_size_h, kernel_size_w
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        has_bias = self.bias is not None
        bias_tensor = self.bias if has_bias else torch.empty(0, device=x.device)
        return depthwise_conv.depthwise_conv2d_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            bias_tensor.contiguous(),
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w,
            has_bias
        )

# Test code (not part of the submission)
def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups]