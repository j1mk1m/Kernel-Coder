import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused GELU and average pooling kernel
fused_gelu_avg_pool_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_gelu_avg_pool(
    const float* input,
    float* output,
    int B, int C, int H, int W) {

    int b = blockIdx.x / C;
    int c = blockIdx.x % C;

    extern __shared__ float shared[];
    int tid = threadIdx.x;
    float sum = 0.0f;

    const int spatial_size = H * W;
    for (int idx = tid; idx < spatial_size; idx += blockDim.x) {
        int h = idx / W;
        int w = idx % W;
        float x = input[b * C * H * W + c * H * W + h * W + w];
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        float tanh_val = tanhf(inner);
        float y = x * 0.5f * (1.0f + tanh_val);
        sum += y;
    }

    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[b * C + c] = shared[0] / (H * W);
    }
}

torch::Tensor fused_gelu_avg_pool_cuda(torch::Tensor input) {
    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty({B, C}, input.options());

    const int block_size = 256;
    const int grid_size = B * C;

    const size_t shared_mem_size = block_size * sizeof(float);

    fused_gelu_avg_pool<<<grid_size, block_size, shared_mem_size, torch::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W
    );

    return output;
}
"""

fused_gelu_avg_pool_cpp_source = """
torch::Tensor fused_gelu_avg_pool_cuda(torch::Tensor input);
"""

# Compile the fused kernel
fused_gelu_avg_pool = load_inline(
    name="fused_gelu_avg_pool",
    cpp_sources=fused_gelu_avg_pool_cpp_source,
    cuda_sources=fused_gelu_avg_pool_source,
    functions=["fused_gelu_avg_pool_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_gelu_avg_pool = fused_gelu_avg_pool

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_gelu_avg_pool.fused_gelu_avg_pool_cuda(x)
        return x