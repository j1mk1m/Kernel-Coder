import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return custom_conv2d.custom_conv2d_cuda(x, self.weight, self.stride, self.padding, self.dilation)

from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void im2col_kernel(
    const float* data_im,
    float* data_col,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w) {

    int col_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (col_index >= channels * kernel_h * kernel_w * batch_size * height * width) {
        return;
    }

    int total_per_channel = kernel_h * kernel_w * batch_size * height * width;
    int channel = col_index / total_per_channel;
    int remainder = col_index % total_per_channel;

    int kernel_index = remainder / (batch_size * height * width);
    int kernel_y = kernel_index / kernel_w;
    int kernel_x = kernel_index % kernel_w;

    remainder = remainder % (batch_size * height * width);

    int n = remainder / (height * width);
    int input_y = (remainder / width) % height;
    int input_x = remainder % width;

    int output_y = (input_y + pad_h - kernel_y * dilation_h) / stride_h;
    int output_x = (input_x + pad_w - kernel_x * dilation_w) / stride_w;

    if ((input_y + pad_h - kernel_y * dilation_h) % stride_h != 0 ||
        (input_x + pad_w - kernel_x * dilation_w) % stride_w != 0) {
        return;
    }

    int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    if (output_y < 0 || output_y >= output_h) return;
    if (output_x < 0 || output_x >= output_w) return;

    int output_pos = output_y * output_w + output_x;

    int col_offset = channel * kernel_h * kernel_w * output_h * output_w * batch_size 
                    + kernel_y * kernel_w * output_h * output_w * batch_size 
                    + kernel_x * output_h * output_w * batch_size 
                    + n * output_h * output_w 
                    + output_pos;

    data_col[col_offset] = data_im[ n * channels * height * width + channel * height * width + input_y * width + input_x ];
}

torch::Tensor custom_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation) {

    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(weight.device().is_cuda(), "Weight must be on CUDA");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D tensor");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D tensor");

    int batch_size = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int C_out = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int output_h = (H + 2 * padding - (dilation * (kernel_h - 1) + 1)) / stride + 1;
    int output_w = (W + 2 * padding - (dilation * (kernel_w - 1) + 1)) / stride + 1;

    int col_size = C_in * kernel_h * kernel_w * output_h * output_w * batch_size;
    auto col = torch::zeros({col_size}, input.options());

    int threadsPerBlock = 256;
    int blocksPerGrid = (col_size + threadsPerBlock - 1) / threadsPerBlock;

    im2col_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input.data_ptr<float>(),
        col.data_ptr<float>(),
        batch_size, C_in, H, W,
        kernel_h, kernel_w,
        padding, padding,
        stride, stride,
        dilation, dilation);

    cublasHandle_t handle;
    cublasCreate(&handle);

    int m = C_out;
    int n = output_h * output_w * batch_size;
    int k = C_in * kernel_h * kernel_w;

    auto weight_matrix = weight.view({m, k});
    auto col_matrix = col.view({k, n});

    auto output_col = torch::empty({m, n}, input.options());

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        weight_matrix.data_ptr<float>(), m,
        col_matrix.data_ptr<float>(), k,
        &beta,
        output_col.data_ptr<float>(), m);

    cublasDestroy(handle);

    return output_col.view({batch_size, C_out, output_h, output_w});
}
"""

cuda_header = """
torch::Tensor custom_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation);
"""

custom_conv2d = load_inline(
    name="custom_conv2d",
    cuda_sources=cuda_source,
    cuda_headers=cuda_header,
    functions=["custom_conv2d_cuda"],
    verbose=True
)