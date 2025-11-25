import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused GELU + adaptive_avg_pool2d + squeeze kernel
fused_gelu_avgpool_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <math.h>

#define THREADS_PER_BLOCK 256

__global__ void fused_gelu_avgpool_kernel(
    const float* input, float* output,
    int B, int C, int H, int W) {

    int block_idx = blockIdx.x;
    int b = block_idx / C;
    int c = block_idx % C;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];

    float sum = 0.0f;

    for (int idx = tid; idx < H * W; idx += blockDim.x) {
        int h = idx / W;
        int w = idx % W;
        int offset = b * C * H * W + c * H * W + h * W + w;
        float x = input[offset];

        // Compute GELU
        const float sqrt_2_over_pi = 0.7978845608f;
        const float approximation_term = 0.044715f;
        float inner = sqrt_2_over_pi * (x + approximation_term * x * x * x);
        float tanh_val = tanhf(inner);
        float gelu_val = x * 0.5f * (1.0f + tanh_val);

        sum += gelu_val;
    }

    sdata[tid] = sum;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
            total += sdata[i];
        }
        int output_offset = b * C + c;
        output[output_offset] = total / (H * W);
    }
}

torch::Tensor fused_gelu_avgpool_cuda(torch::Tensor input) {
    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::empty({B, C}, input.options());

    const dim3 grid(B * C);
    const dim3 block(THREADS_PER_BLOCK);
    size_t shared_size = block.x * sizeof(float);

    fused_gelu_avgpool_kernel<<<grid, block, shared_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W);

    return output;
}
"""

fused_gelu_avgpool_cpp = """
torch::Tensor fused_gelu_avgpool_cuda(torch::Tensor input);
"""

# Compile the fused kernel
fused_gelu_avgpool = load_inline(
    name="fused_gelu_avgpool",
    cpp_sources=fused_gelu_avgpool_cpp,
    cuda_sources=fused_gelu_avgpool_source,
    functions=["fused_gelu_avgpool_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_gelu_avgpool = fused_gelu_avgpool

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_gelu_avgpool.fused_gelu_avgpool_cuda(x)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]