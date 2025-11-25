import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

im2col_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void im2col(
    const float* input,
    float* col,
    int N, int C_in, int H_in, int W_in,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int H_out, int W_out) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C_in * kernel_h * kernel_w * N * H_out * W_out) return;

    int row = idx / (N * H_out * W_out);
    int col_idx = idx % (N * H_out * W_out);

    int kw = row % kernel_w;
    row /= kernel_w;
    int kh = row % kernel_h;
    int c_in = row / kernel_h;

    int w_out = col_idx % W_out;
    col_idx /= W_out;
    int h_out = col_idx % H_out;
    int n = col_idx / H_out;

    int h_in = h_out * stride_h - padding_h + kh * dilation_h;
    int w_in = w_out * stride_w - padding_w + kw * dilation_w;

    int input_offset = n * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in;
    float val = input[input_offset];

    int out_offset = row * (N * H_out * W_out) + col_idx;
    col[out_offset] = val;
}
"""

im2col_cpp_source = """
#include <torch/extension.h>

extern "C" {

void im2col_cuda(
    torch::Tensor input,
    torch::Tensor col,
    int N, int C_in, int H_in, int W_in,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int H_out, int W_out) {

    int64_t total = C_in * kernel_h * kernel_w * N * H_out * W_out;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    im2col<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        col.data_ptr<float>(),
        N, C_in, H_in, W_in,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        H_out, W_out
    );
}
}
"""

im2col = load_inline(
    name="im2col",
    cuda_sources=im2col_source,
    cpp_sources=im2col_cpp_source,
    functions=["im2col_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        assert groups == 1, "Only groups=1 supported"
        assert bias is False, "Bias not supported"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                kernel_size[0],
                kernel_size[1]
            )
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C_in, H_in, W_in = x.size()
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation

        # Compute output dimensions
        H_out = (
            (H_in + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h
            + 1
        )
        W_out = (
            (W_in + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w
            + 1
        )

        # Compute the number of columns in the im2col matrix
        cols = N * H_out * W_out
        rows = C_in * kernel_h * kernel_w

        # Allocate the im2col matrix
        col = torch.empty((rows, cols), dtype=x.dtype, device=x.device)

        # Launch the im2col kernel
        im2col.im2col_cuda(
            x,
            col,
            N, C_in, H_in, W_in,
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            H_out, W_out
        )

        # Reshape weights to (out_channels, rows)
        weight_matrix = self.weight.view(self.out_channels, rows)

        # Perform matrix multiply
        output_matrix = torch.mm(weight_matrix, col)

        # Reshape to desired output shape
        output = output_matrix.view(N, self.out_channels, H_out, W_out)

        return output

def get_init_inputs():
    # Provide parameters needed to initialize the model
    in_channels = 64
    out_channels = 128
    kernel_size = (5, 7)
    return [in_channels, out_channels, kernel_size]