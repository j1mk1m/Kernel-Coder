import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the CUDA kernel code with optimizations
conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define H_TILE 16
#define W_TILE 16

__global__ void conv2d_kernel(float* input, float* weight, float* output,
                             int batch, int c_out,
                             int C_in, int K_h, int K_w,
                             int H_in, int W_in,
                             int H_out, int W_out) {
    extern __shared__ float s_input[];
    int block_y = blockIdx.y;
    int block_x = blockIdx.x;
    int start_h = block_y * H_TILE;
    int start_w = block_x * W_TILE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int h_out = start_h + ty;
    int w_out = start_w + tx;
    if (h_out >= H_out || w_out >= W_out) return;

    const int H_patch = H_TILE + K_h - 1;
    const int W_patch = W_TILE + K_w - 1;
    const int elements_per_channel = H_patch * W_patch;
    const int threads_per_block = blockDim.x * blockDim.y;
    const int elements_per_thread = (elements_per_channel + threads_per_block - 1) / threads_per_block;

    // Load input into shared memory with coalesced access
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int i = 0; i < elements_per_thread; ++i) {
            int pos = (threadIdx.y * blockDim.x + threadIdx.x) * elements_per_thread + i;
            if (pos < elements_per_channel) {
                int h_patch = pos / W_patch;
                int w_patch = pos % W_patch;
                int h_in = start_h + h_patch;
                int w_in = start_w + w_patch;
                float in_val = 0.0f;
                if (h_in < H_in && w_in < W_in) {
                    int offset = c_in * H_in * W_in + h_in * W_in + w_in;
                    in_val = input[batch * C_in * H_in * W_in + offset];
                }
                s_input[c_in * elements_per_channel + pos] = in_val;
            }
        }
    }
    __syncthreads();

    float sum = 0.0f;
    // Reordered loops for better cache locality
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                int h_patch = ty + kh;
                int w_patch = tx + kw;
                if (h_patch < H_patch && w_patch < W_patch) {
                    int pos_in_patch = h_patch * W_patch + w_patch;
                    float in_val = s_input[c_in * elements_per_channel + pos_in_patch];
                    int weight_offset = c_out * C_in * K_h * K_w + c_in * K_h * K_w + kh * K_w + kw;
                    sum += in_val * weight[weight_offset];
                }
            }
        }
    }
    int out_offset = batch * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out;
    output[out_offset] = sum;
}

torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight,
                            int K_h, int K_w) {
    int batch_size = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(0);
    int H_out = H_in - K_h + 1;
    int W_out = W_in - K_w + 1;

    auto output = torch::empty({batch_size, C_out, H_out, W_out}, input.options());

    dim3 threads(H_TILE, W_TILE);
    int H_patch = H_TILE + K_h - 1;
    int W_patch = W_TILE + K_w - 1;
    int shared_size = C_in * H_patch * W_patch * sizeof(float);

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int c_out = 0; c_out < C_out; ++c_out) {
            int grid_x = (W_out + W_TILE - 1) / W_TILE;
            int grid_y = (H_out + H_TILE - 1) / H_TILE;
            dim3 blocks(grid_x, grid_y);
            conv2d_kernel<<<blocks, threads, shared_size>>>(
                input.data_ptr<float>(),
                weight.data_ptr<float>(),
                output.data_ptr<float>(),
                batch, c_out,
                C_in, K_h, K_w,
                H_in, W_in,
                H_out, W_out
            );
        }
    }

    return output;
}
"""

# Compile the CUDA code
conv2d = load_inline(
    name="conv2d",
    cpp_sources="",
    cuda_sources=conv2d_cuda_source,
    functions=["conv2d_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        # Initialize weights using the same method as PyTorch's Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K_h, K_w = self.weight.shape[2], self.weight.shape[3]
        return conv2d.conv2d_forward(x, self.weight, K_h, K_w)