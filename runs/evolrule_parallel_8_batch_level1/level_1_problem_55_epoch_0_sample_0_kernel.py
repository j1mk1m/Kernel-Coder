import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
conv2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_H 16
#define TILE_W 16
#define CHANNELS_PER_BLOCK 4

template<int KERNEL_SIZE>
__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int stride,
    int padding,
    int dilation,
    int output_height,
    int output_width
) {
    const int shared_h = TILE_H + KERNEL_SIZE - 1;
    const int shared_w = TILE_W + KERNEL_SIZE - 1;
    extern __shared__ float s_input[];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int output_x_start = bx * TILE_W;
    int output_y_start = by * TILE_H;

    int input_x_start = output_x_start * stride - padding;
    int input_y_start = output_y_start * stride - padding;

    int batch_per_block = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    int batch = bz / batch_per_block;
    int channel_block = bz % batch_per_block;
    int channel_start = channel_block * CHANNELS_PER_BLOCK;
    int channel = channel_start + tz;

    if (channel >= out_channels) return;

    int output_x = output_x_start + tx;
    int output_y = output_y_start + ty;

    if (output_x >= output_width || output_y >= output_height) return;

    // Load input into shared memory
    for (int c = tz; c < in_channels; c += CHANNELS_PER_BLOCK) {
        for (int s_y = ty; s_y < shared_h; s_y += blockDim.y) {
            for (int s_x = tx; s_x < shared_w; s_x += blockDim.x) {
                int in_y = input_y_start + s_y;
                int in_x = input_x_start + s_x;

                if (in_y < 0 || in_y >= input_height || in_x < 0 || in_x >= input_width)
                    continue;

                int s_offset = (s_y * shared_w + s_x) * in_channels + c;
                int in_offset = batch * in_channels * input_height * input_width +
                                c * input_height * input_width +
                                in_y * input_width + in_x;

                s_input[s_offset] = input[in_offset];
            }
        }
    }
    __syncthreads();

    float sum = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
            for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                int s_y = ty + ky - 1;
                int s_x = tx + kx - 1;

                if (s_y < 0 || s_y >= shared_h || s_x < 0 || s_x >= shared_w)
                    continue;

                int s_offset = (s_y * shared_w + s_x) * in_channels + c_in;
                float in_val = s_input[s_offset];

                int w_offset = channel * (in_channels * KERNEL_SIZE * KERNEL_SIZE) +
                               c_in * (KERNEL_SIZE * KERNEL_SIZE) +
                               ky * KERNEL_SIZE + kx;
                float w_val = weight[w_offset];

                sum += in_val * w_val;
            }
        }
    }

    int out_offset = batch * out_channels * output_height * output_width +
                     channel * output_height * output_width +
                     output_y * output_width + output_x;
    output[out_offset] = sum;
}

torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation,
    int kernel_size
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);

    int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 blockDim(TILE_W, TILE_H, CHANNELS_PER_BLOCK);
    dim3 grid(
        (output_width + TILE_W - 1) / TILE_W,
        (output_height + TILE_H - 1) / TILE_H,
        batch_size * ((out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK)
    );

    int shared_size = (TILE_H + kernel_size -1) * (TILE_W + kernel_size -1) * in_channels * sizeof(float);

    if (kernel_size == 3) {
        conv2d_kernel<3><<<grid, blockDim, shared_size>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, in_channels, out_channels,
            input_height, input_width,
            stride, padding, dilation,
            output_height, output_width
        );
    }

    return output;
}
"""

conv2d_cpp_source = """
#include <torch/extension.h>
torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation, int kernel_size);
"""

# Load the CUDA extension
conv_cuda = load_inline(
    name="conv_cuda",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        out = conv_cuda.conv2d_cuda(
            x, self.weight, self.stride, self.padding, self.dilation, self.kernel_size
        )
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        return out