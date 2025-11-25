import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code
kernel_code = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv_kernel(
    const float* input, const float* kernel_weights, float* output,
    int batch_size, int in_channels,
    int input_height, int input_width,
    int kernel_h, int kernel_w,
    int output_height, int output_width,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
) {
    int batch_idx = blockIdx.x;
    int channel = blockIdx.y;
    int tile_idx = blockIdx.z;

    // Determine tile's row and column
    int tiles_h = (output_height + blockDim.y - 1) / blockDim.y;
    int tiles_w = (output_width + blockDim.x - 1) / blockDim.x;

    int tile_row = tile_idx / tiles_w;
    int tile_col = tile_idx % tiles_w;

    int h_out_start = tile_row * blockDim.y;
    int w_out_start = tile_col * blockDim.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    __shared__ float s_input[18][22]; // 3 rows and 7 cols kernel, block size 16x16

    // Load input into shared memory
    int input_row = h_out_start + ty;
    int input_col = w_out_start + tx;

    int input_offset = batch_idx * in_channels * input_height * input_width
        + channel * input_height * input_width
        + input_row * input_width + input_col;

    if (input_row < input_height && input_col < input_width) {
        s_input[ty][tx] = input[input_offset];
    } else {
        s_input[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Compute output position
    int h_out = h_out_start + ty;
    int w_out = w_out_start + tx;

    if (h_out >= output_height || w_out >= output_width)
        return;

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int s_row = ty + kh;
            int s_col = tx + kw;
            sum += s_input[s_row][s_col] * kernel_weights[channel * kernel_h * kernel_w + kh * kernel_w + kw];
        }
    }

    // Write output
    int output_offset = batch_idx * in_channels * output_height * output_width
        + channel * output_height * output_width
        + h_out * output_width + w_out;

    output[output_offset] = sum;
}
"""

# Compile the kernel
depthwise_conv = load_inline(
    name="depthwise_conv",
    cuda_sources=kernel_code,
    functions=["depthwise_conv_kernel"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Use the same Conv2d layer to hold weights and bias
        self.conv2d = nn.Conv2d(
            in_channels, in_channels,  # depthwise: out_channels = in_channels
            (kernel_size_h, kernel_size_w),
            stride=(stride_h, stride_w),
            padding=(padding_h, padding_w),
            dilation=(dilation_h, dilation_w),
            groups=in_channels,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, input_height, input_width = x.size()
        kernel_h = self.conv2d.kernel_size[0]
        kernel_w = self.conv2d.kernel_size[1]
        stride_h = self.conv2d.stride[0]
        stride_w = self.conv2d.stride[1]
        padding_h = self.conv2d.padding[0]
        padding_w = self.conv2d.padding[1]
        dilation_h = self.conv2d.dilation[0]
        dilation_w = self.conv2d.dilation[1]
        
        # Calculate output dimensions
        output_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        output_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

        output = torch.empty(
            (batch_size, in_channels, output_height, output_width),
            dtype=x.dtype,
            device=x.device
        )

        # Define block and grid dimensions
        block_size = (16, 16)  # blockDim.x (width), blockDim.y (height)
        blockDim = block_size

        # Calculate number of tiles per channel and batch
        tiles_h = (output_height + blockDim[1] - 1) // blockDim[1]
        tiles_w = (output_width + blockDim[0] - 1) // blockDim[0]
        num_tiles_per_channel = tiles_h * tiles_w

        gridDim = (
            batch_size,          # blockIdx.x: batch index
            in_channels,         # blockIdx.y: channel index
            num_tiles_per_channel  # blockIdx.z: tile index
        )

        # Launch the kernel
        depthwise_conv.depthwise_conv_kernel[
            gridDim,
            blockDim
        ](x.contiguous().data_ptr(),
          self.conv2d.weight.contiguous().data_ptr(),
          output.data_ptr(),
          batch_size,
          in_channels,
          input_height,
          input_width,
          kernel_h,
          kernel_w,
          output_height,
          output_width,
          stride_h,
          stride_w,
          padding_h,
          padding_w,
          dilation_h,
          dilation_w)

        # Apply bias if present
        if self.conv2d.bias is not None:
            bias = self.conv2d.bias.view(1, -1, 1, 1)
            output += bias

        return output

def get_inputs():
    # Same as original
    x = torch.rand(32, 128, 128, 256).cuda()
    return [x]

def get_init_inputs():
    # Same as original
    return [32, 128, 128, 256, 3, 7, 1, 1, 0, 0, 1, 1, 128, False]