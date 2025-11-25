import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void im2col_kernel(const float* input, float* cols,
                             int N, int C, int D, int H, int W,
                             int KD, int KH, int KW,
                             int stride, int padding, int dilation,
                             int D_out, int H_out, int W_out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= C * KD * KH * KW * D_out * H_out * W_out * N)
        return;

    int total_columns = D_out * H_out * W_out;
    int column = (index / (C * KD * KH * KW)) % total_columns;
    int element_in_col = index % (C * KD * KH * KW);
    int n = index / (total_columns * C * KD * KH * KW);

    int kd = element_in_col / (KH * KW * C);
    int remainder = element_in_col % (KH * KW * C);
    int c = remainder / (KH * KW);
    int kh_kw = remainder % (KH * KW);
    int kh = kh_kw / KW;
    int kw = kh_kw % KW;

    int w_out = column % W_out;
    int rem = column / W_out;
    int h_out = rem % H_out;
    int d_out = rem / H_out;

    int d_in = d_out * stride + kd * dilation - padding;
    int h_in = h_out * stride + kh * dilation - padding;
    int w_in = w_out * stride + kw * dilation - padding;

    if (d_in < 0 || d_in >= D ||
        h_in < 0 || h_in >= H ||
        w_in < 0 || w_in >= W) {
        cols[index] = 0.0f;
        return;
    }

    int input_offset = n * C * D * H * W +
                       c * D * H * W +
                       d_in * H * W +
                       h_in * W +
                       w_in;
    cols[index] = input[input_offset];
}

torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight,
                                  torch::Tensor bias,
                                  int stride, int padding,
                                  int dilation, int groups) {
    if (groups != 1) {
        TORCH_CHECK(groups == 1, "Only groups=1 supported for now.");
    }

    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int out_channels = weight.size(0);
    int KD = weight.size(2);
    int KH = weight.size(3);
    int KW = weight.size(4);

    int D_out = (D + 2 * padding - dilation * (KD - 1) - 1) / stride + 1;
    int H_out = (H + 2 * padding - dilation * (KH - 1) - 1) / stride + 1;
    int W_out = (W + 2 * padding - dilation * (KW - 1) - 1) / stride + 1;

    int cols_height = C * KD * KH * KW;
    int cols_width = D_out * H_out * W_out;

    auto cols = torch::empty({N, cols_height, cols_width}, input.options());

    int total_cols_elements = N * cols_height * cols_width;
    int threads = 256;
    int blocks = (total_cols_elements + threads - 1) / threads;

    im2col_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), cols.data_ptr<float>(),
        N, C, D, H, W,
        KD, KH, KW,
        stride, padding, dilation,
        D_out, H_out, W_out
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;

    auto weight_reshaped = weight.view({out_channels, cols_height});
    auto cols_transposed = cols.permute({1, 2, 0}).contiguous().view({cols_height, N * cols_width});
    auto output_flat = torch::empty({out_channels, N * cols_width}, input.options());

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                out_channels, N * cols_width, cols_height,
                &alpha,
                weight_reshaped.data_ptr<float>(), out_channels,
                cols_transposed.data_ptr<float>(), cols_height,
                &beta,
                output_flat.data_ptr<float>(), out_channels);

    cublasDestroy(handle);

    auto output = output_flat.view({N, out_channels, D_out, H_out, W_out});

    if (bias.defined()) {
        output += bias.view({1, -1, 1, 1, 1});
    }

    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                  int stride, int padding, int dilation, int groups);
"""

conv3d_module = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv3d_module.conv3d_forward_cuda(
            x, self.weight,
            self.bias if self.bias is not None else torch.empty(0),
            self.stride, self.padding, self.dilation, self.groups
        )