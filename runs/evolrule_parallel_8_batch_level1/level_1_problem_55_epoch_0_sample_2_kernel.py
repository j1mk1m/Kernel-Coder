import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

custom_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

template <typename scalar_t>
__global__ void im2col_gpu_kernel(
    const scalar_t* data_im, 
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
    const int dilation_w,
    const int batch_size,
    scalar_t* data_col) 
{
    const int output_height = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_width = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int output_size = output_height * output_width;
    const int kernel_size_total = kernel_h * kernel_w * channels;

    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_index >= batch_size * kernel_size_total * output_size) return;

    int batch = global_index / (kernel_size_total * output_size);
    int col_index = global_index % (kernel_size_total * output_size);

    int kh = (col_index / (kernel_w * channels * output_size)) % kernel_h;
    int kw = (col_index / (channels * output_size)) % kernel_w;
    int c = (col_index / output_size) % channels;
    int h_out = (col_index % output_size) / output_width;
    int w_out = col_index % output_width;

    int h_in = h_out * stride_h - pad_h + kh * dilation_h;
    int w_in = w_out * stride_w - pad_w + kw * dilation_w;

    int input_offset = batch * channels * height * width + c * height * width;
    int data_col_offset = batch * kernel_size_total * output_size + col_index;

    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
        data_col[data_col_offset] = data_im[input_offset + h_in * width + w_in];
    } else {
        data_col[data_col_offset] = 0;
    }
}

torch::Tensor custom_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation,
    int groups) 
{
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int kernel_size = weight.size(2);
    const int out_channels = weight.size(0);

    const int kernel_h = kernel_size;
    const int kernel_w = kernel_size;
    const int stride_h = stride;
    const int stride_w = stride;
    const int pad_h = padding;
    const int pad_w = padding;
    const int dilation_h = dilation;
    const int dilation_w = dilation;

    const int output_height = (input_height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_width = (input_width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int output_size = output_height * output_width;
    const int kernel_size_total = kernel_h * kernel_w * in_channels;

    auto options = input.options();
    auto col = torch::empty({batch_size, kernel_size_total, output_size}, options);

    const int num_threads = batch_size * kernel_size_total * output_size;
    const int threads_per_block = 1024;
    const int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

    im2col_gpu_kernel<float><<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        in_channels,
        input_height,
        input_width,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        batch_size,
        col.data_ptr<float>());

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, options);
    auto output_reshaped = output.view({batch_size, out_channels, -1});

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    auto weight_reshaped = weight.view({out_channels, kernel_size_total});

    int m = out_channels;
    int n = output_size;
    int k = kernel_size_total;

    int lda = k;
    int strideA = 0;
    int ldb = k;
    int strideB = k * n;
    int ldc = m;
    int strideC = m * n;

    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha,
        weight_reshaped.data_ptr<float>(), CUDA_R_32F, lda, strideA,
        col.data_ptr<float>(), CUDA_R_32F, ldb, strideB,
        &beta,
        output_reshaped.data_ptr<float>(), CUDA_R_32F, ldc, strideC,
        batch_size,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CUBLAS_CHECK(cublasDestroy(handle));

    return output;
}
"""

custom_conv_cpp_source = (
    "torch::Tensor custom_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation, int groups);"
)

custom_conv = load_inline(
    name="custom_conv",
    cpp_sources=custom_conv_cpp_source,
    cuda_sources=custom_conv_source,
    functions=["custom_conv2d_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_conv.custom_conv2d_cuda(x, self.weight, self.stride, self.padding, self.dilation, self.groups)