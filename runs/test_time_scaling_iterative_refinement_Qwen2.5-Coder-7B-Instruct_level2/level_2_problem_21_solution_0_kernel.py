import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, const float* bias, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size, int stride, int padding, int dilation) {
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    if (n >= batch_size || c >= out_channels) return;

    int h_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int w_out = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    int h_in_start = n * in_channels * height * width + c * height * width;
    int w_in_start = h_in_start;
    int h_out_start = n * out_channels * h_out * w_out + c * h_out * w_out;
    int w_out_start = h_out_start;

    for (int i = 0;