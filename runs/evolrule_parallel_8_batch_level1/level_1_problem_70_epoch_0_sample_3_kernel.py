import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

conv3d_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<int K>
__global__ void conv3d_transpose(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int B, int C_in, int D_in, int H_in, int W_in,
    int C_out, int stride, int padding,
    int D_out, int H_out, int W_out) {
    
    int b = blockIdx.x;
    int f = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    // Compute spatial indices using block and thread indices
    int d = tx + blockIdx.z * blockDim.x;
    int h = ty + blockIdx.z * blockDim.y;
    int w = tz + blockIdx.z * blockDim.z;
    
    if (d >= D_out || h >= H_out || w >= W_out) return;

    float acc = 0.0f;
    
    for (int c = 0; c < C_in; ++c) {
        for (int kd = 0; kd < K; ++kd) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int d_in = (d - kd - padding) / stride;
                    int h_in = (h - kh - padding) / stride;
                    int w_in = (w - kw - padding) / stride;
                    
                    if (d_in >= 0 && d_in < D_in &&
                        h_in >= 0 && h_in < H_in &&
                        w_in >= 0 && w_in < W_in) {
                        int input_offset = ((b * C_in + c) * D_in + d_in) * H_in * W_in +
                                          h_in * W_in + w_in;
                        float in_val = input[input_offset];
                        
                        int weight_offset = ((f * C_in + c) * K + kd) * K * K +
                                           kh * K + kw;
                        float wgt_val = weight[weight_offset];
                        
                        acc += in_val * wgt_val;
                    }
                }
            }
        }
    }
    
    int output_offset = ((b * C_out + f) * D_out + d) * H_out * W_out +
                        h * W_out + w;
    output[output_offset] = acc;
}

torch::Tensor conv3d_transpose_cuda(torch::Tensor input, torch::Tensor weight,
                                   int stride, int padding, int kernel_size) {
    const int B = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);
    
    const int C_out = weight.size(0);
    const int K = kernel_size;
    
    // Compute output dimensions
    int D_out = (D_in - 1) * stride + K - 2 * padding;
    int H_out = (H_in - 1) * stride + K - 2 * padding;
    int W_out = (W_in - 1) * stride + K - 2 * padding;
    
    auto output = torch::empty({B, C_out, D_out, H_out, W_out}, input.options());
    
    dim3 threadsPerBlock(8, 8, 8);
    int spatial_blocks = (D_out + threadsPerBlock.x - 1) / threadsPerBlock.x;
    spatial_blocks = (spatial_blocks + threadsPerBlock.y - 1) / threadsPerBlock.y;
    spatial_blocks = (spatial_blocks + threadsPerBlock.z - 1) / threadsPerBlock.z;
    dim3 blocksPerGrid(B, C_out, spatial_blocks);
    
    switch (K) {
        case 3:
            conv3d_transpose<3><<<blocksPerGrid, threadsPerBlock>>>(
                input.data_ptr<float>(),
                weight.data_ptr<float>(),
                output.data_ptr<float>(),
                B, C_in, D_in, H_in, W_in,
                C_out, stride, padding,
                D_out, H_out, W_out);
            break;
        default:
            AT_ERROR("Kernel size not supported");
    }
    
    return output;
}
"""

conv3d_transpose = load_inline(
    name="conv3d_transpose",
    cpp_sources="",
    cuda_sources=conv3d_transpose_source,
    functions=["conv3d_transpose_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 output_padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        assert dilation == 1, "Only dilation=1 supported"
        assert groups == 1, "Only groups=1 supported"
        assert output_padding == 0, "Only output_padding=0 supported"
        assert isinstance(stride, int), "Stride must be an integer"
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size, kernel_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        output = conv3d_transpose_cuda(x, self.weight, self.stride, self.padding, self.kernel_size)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output