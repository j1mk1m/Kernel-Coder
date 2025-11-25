import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for scaling and min operation
fused_scale_min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_scale_min_kernel(
    const float* input,
    float scale_factor,
    float* output,
    int batch_size,
    int out_channels,
    int height_out,
    int width_out
) {
    int index = blockIdx.x;
    int batch = index / (height_out * width_out);
    int rem = index % (height_out * width_out);
    int h = rem / width_out;
    int w = rem % width_out;

    int c = threadIdx.x;

    extern __shared__ float shared_data[];

    float scaled_val = 0.0f;
    if (c < out_channels) {
        int input_offset = batch * out_channels * height_out * width_out +
                           c * height_out * width_out +
                           h * width_out + w;
        scaled_val = input[input_offset] * scale_factor;
    }

    shared_data[threadIdx.x] = scaled_val;
    __syncthreads();

    int tid = threadIdx.x;
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_data[tid] > shared_data[tid + s]) {
                shared_data[tid] = shared_data[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int output_offset = batch * height_out * width_out + h * width_out + w;
        output[output_offset] = shared_data[0];
    }
}

torch::Tensor fused_scale_min_cuda(torch::Tensor input, float scale_factor) {
    int batch_size = input.size(0);
    int out_channels = input.size(1);
    int height_out = input.size(2);
    int width_out = input.size(3);

    auto output = torch::empty({batch_size, 1, height_out, width_out}, 
                              torch::dtype(input.dtype()).device(input.device()));

    int block_size = out_channels;
    int grid_size = batch_size * height_out * width_out;

    size_t shared_mem_size = block_size * sizeof(float);

    fused_scale_min_kernel<<<grid_size, block_size, shared_mem_size, 
                             torch::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        scale_factor,
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        height_out,
        width_out
    );

    return output;
}
"""

fused_scale_min_cpp_source = """
torch::Tensor fused_scale_min_cuda(torch::Tensor input, float scale_factor);
"""

# Compile the inline CUDA code
fused_scale_min = load_inline(
    name="fused_scale_min",
    cpp_sources=fused_scale_min_cpp_source,
    cuda_sources=fused_scale_min_source,
    functions=["fused_scale_min_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.fused_scale_min = fused_scale_min

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_scale_min.fused_scale_min_cuda(x, self.scale_factor)
        return x

# The get_inputs and get_init_inputs remain unchanged as per the problem statement
batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]