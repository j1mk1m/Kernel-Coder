import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Initialize weights similar to PyTorch's Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

        # Define CUDA kernel source
        conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define TILE_HEIGHT 16

template <int K>
__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int N,
    int C_in,
    int H,
    int W,
    int C_out,
    int H_out,
    int W_out,
    int stride,
    int padding,
    int dilation
) {
    __shared__ float shared_input[TILE_WIDTH + K -1][TILE_HEIGHT + K -1];

    int batch = blockIdx.z / C_out;
    int c_out = blockIdx.z % C_out;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int output_x = bx * TILE_WIDTH + tx;
    int output_y = by * TILE_HEIGHT + ty;

    if (output_x >= H_out || output_y >= W_out) {
        return;
    }

    float sum = 0.0f;

    // Compute the starting position of the input block for this output tile
    int input_block_start_x = (bx * TILE_WIDTH) * stride - padding;
    int input_block_start_y = (by * TILE_HEIGHT) * stride - padding;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        // Load input into shared memory for this channel
        int in_row = input_block_start_x + tx;
        int in_col = input_block_start_y + ty;

        if (in_row < 0 || in_row >= H || in_col < 0 || in_col >= W) {
            shared_input[tx][ty] = 0.0f;
        } else {
            int input_idx = batch * C_in * H * W + c_in * H * W + in_row * W + in_col;
            shared_input[tx][ty] = input[input_idx];
        }

        __syncthreads();

        // Compute the contribution of this channel and kernel elements
        for (int r = 0; r < K; ++r) {
            for (int s = 0; s < K; ++s) {
                int sm_row = tx + r;
                int sm_col = ty + s;
                if (sm_row < (TILE_WIDTH + K -1) && sm_col < (TILE_HEIGHT + K -1)) {
                    int weight_offset = c_out * C_in * K * K + c_in * K * K + r * K + s;
                    float w = weight[weight_offset];
                    sum += w * shared_input[sm_row][sm_col];
                }
            }
        }

        __syncthreads();
    }

    // Write the output
    int output_offset = batch * C_out * H_out * W_out + c_out * H_out * W_out + output_x * W_out + output_y;
    output[output_offset] = sum;
}

// Host function to launch the kernel
torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
) {
    const int C_in = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int C_out = weight.size(0);
    const int kernel_size = weight.size(2);
    const int N = input.size(0);

    const int H_out = (H - kernel_size + 2 * padding) / stride + 1;
    const int W_out = (W - kernel_size + 2 * padding) / stride + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    dim3 threads(TILE_WIDTH, TILE_HEIGHT);
    dim3 blocks(
        (H_out + TILE_WIDTH - 1) / TILE_WIDTH,
        (W_out + TILE_HEIGHT - 1) / TILE_HEIGHT,
        N * C_out
    );

    if (kernel_size == 3) {
        conv2d_kernel<3><<<blocks, threads>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            N, C_in, H, W, C_out, H_out, W_out, stride, padding, dilation
        );
    } else {
        AT_ERROR("Kernel size not supported");
    }

    return output;
}
        """

        # Compile the kernel
        self.conv2d = load_inline(
            name="conv2d",
            cpp_sources=[conv2d_source],
            functions=["conv2d_forward"],
            verbose=True
        )

    def forward(self, x):
        output = self.conv2d.conv2d_forward(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.dilation
        )
        return output