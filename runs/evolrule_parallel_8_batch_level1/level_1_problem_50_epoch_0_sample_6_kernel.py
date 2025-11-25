import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Custom CUDA kernel for convolution using F.unfold and cuBLAS
custom_conv2d_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

at::Tensor custom_conv2d_cuda(
    const at::Tensor &input,
    const at::Tensor &weight,
    const at::Tensor &bias,
    int stride,
    int padding
) {
    // Pad the input
    at::Tensor padded_input = F::pad(input, {padding, padding, padding, padding});

    // Compute im2col using F.unfold
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    at::Tensor cols = F::unfold(padded_input, {kernel_h, kernel_w}, stride=stride);

    // Reshape weight to (out_channels, in_channels * kernel_h * kernel_w)
    int out_channels = weight.size(0);
    int in_channels = weight.size(1);
    int kernel_size = kernel_h * kernel_w;
    at::Tensor weight_matrix = weight.view({out_channels, in_channels * kernel_size});

    // Prepare output tensor
    int batch_size = input.size(0);
    int OH = (padded_input.size(2) - kernel_h) / stride + 1;
    int OW = (padded_input.size(3) - kernel_w) / stride + 1;
    at::Tensor output = at::empty({batch_size, out_channels, OH, OW}, input.options());

    // Perform matrix multiply using cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream().stream());

    int M = out_channels;
    int N = cols.size(1); // cols has size (C_in*KH*KW, N*OH*OW)
    int K = weight_matrix.size(1); // in_channels*KH*KW

    at::Tensor output_col = output.view({M, -1});

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        M,
        N,
        K,
        &alpha,
        weight_matrix.data_ptr<float>(),
        M,
        cols.data_ptr<float>(),
        K,
        &beta,
        output_col.data_ptr<float>(),
        M
    );

    cublasDestroy(handle);

    // Add bias
    output_col.add_(bias.view({-1, 1}));

    return output;
}
"""

custom_conv2d_cpp_source = "at::Tensor custom_conv2d_cuda(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias, int stride, int padding);"

custom_conv2d = load_inline(
    name="custom_conv2d",
    cpp_sources=custom_conv2d_cpp_source,
    cuda_sources=custom_conv2d_source,
    functions=["custom_conv2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(96, 3, 11, 11))
        self.bias = nn.Parameter(torch.empty(96))
        # Initialize weights and bias similar to the original Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.stride = 4
        self.padding = 2

    def forward(self, x):
        return custom_conv2d.custom_conv2d_cuda(x, self.weight, self.bias, self.stride, self.padding)