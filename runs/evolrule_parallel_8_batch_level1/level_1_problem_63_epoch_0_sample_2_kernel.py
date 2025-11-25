import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H, int W,
    int C_out, int H_out, int W_out,
    int padding, int stride) {

    const int K = 3;
    const int tile_size = 16;
    const int tile_h = tile_size + K - 1;
    const int tile_w = tile_size + K - 1;

    int n = blockIdx.x;
    int tile_x = blockIdx.y;
    int tile_y = blockIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int input_x_start = tile_x * tile_size;
    int input_y_start = tile_y * tile_size;

    extern __shared__ float shared_input[];

    // Load input into shared memory
    for (int c = 0; c < C_in; c += blockDim.x * blockDim.y) {
        int c_offset = c + tx + ty * blockDim.x;
        if (c_offset < C_in) {
            for (int y = ty; y < tile_h; y += blockDim.y) {
                for (int x = tx; x < tile_w; x += blockDim.x) {
                    int input_y = input_y_start + y - (K-1)/2;
                    int input_x = input_x_start + x - (K-1)/2;
                    if (input_y < 0 || input_y >= H || input_x < 0 || input_x >= W) {
                        shared_input[ c * tile_h * tile_w + y * tile_w + x ] = 0.0f;
                    } else {
                        int idx_in = n * C_in * H * W + c * H * W + input_y * W + input_x;
                        shared_input[ c * tile_h * tile_w + y * tile_w + x ] = input[idx_in];
                    }
                }
            }
        }
    }
    __syncthreads();

    int out_y_start = input_y_start;
    int out_x_start = input_x_start;
    int out_y_end = input_y_start + tile_size;
    int out_x_end = input_x_start + tile_size;

    out_y_end = min(out_y_end, H_out);
    out_x_end = min(out_x_end, W_out);

    int out_x = out_x_start + tx;
    int out_y = out_y_start + ty;

    if (out_x >= out_x_end || out_y >= out_y_end) {
        return;
    }

    for (int c_out = 0; c_out < C_out; ++c_out) {
        float sum = 0.0f;
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int tile_y_in_tile = kh + (out_y - input_y_start);
                    int tile_x_in_tile = kw + (out_x - input_x_start);
                    float in_val = shared_input[ c_in * tile_h * tile_w + tile_y_in_tile * tile_w + tile_x_in_tile ];
                    float weight_val = weights[ c_out * C_in * K * K + c_in * K * K + kh * K + kw ];
                    sum += in_val * weight_val;
                }
            }
        }
        if (bias != nullptr) {
            sum += bias[c_out];
        }
        int out_offset = n * C_out * H_out * W_out + c_out * H_out * W_out + out_y * W_out + out_x;
        output[out_offset] = sum;
    }
}

torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int padding, int stride) {

    int N = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int C_out = weight.size(0);
    int K = 3;

    int H_out = (H - K + 2 * padding) / stride + 1;
    int W_out = (W - K + 2 * padding) / stride + 1;

    torch::Tensor output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const int tile_size = 16;
    dim3 blockDim(16, 16);
    int num_blocks_x = (W / tile_size) + (W % tile_size != 0);
    int num_blocks_y = (H / tile_size) + (H % tile_size != 0);
    dim3 gridDim(N, num_blocks_x, num_blocks_y);

    int tile_h = tile_size + K - 1;
    int tile_w = tile_size + K - 1;
    size_t smem_size = C_in * tile_h * tile_w * sizeof(float);

    conv2d_kernel<<<gridDim, blockDim, smem_size, torch::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H, W,
        C_out, H_out, W_out,
        padding, stride);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

conv2d_cuda = load_inline(
    name="conv2d_cuda",
    cpp_sources=[""],
    cuda_sources=conv2d_cuda_source,
    functions=["conv2d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        assert dilation == 1, "Dilation must be 1"
        assert groups == 1, "Groups must be 1"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weights using Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            bias_tensor = self.bias
        else:
            bias_tensor = torch.empty(0, device=x.device)
        return conv2d_cuda(x, self.weight, bias_tensor, self.padding, self.stride)