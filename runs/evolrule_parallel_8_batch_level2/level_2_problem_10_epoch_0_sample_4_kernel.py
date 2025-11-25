import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for hardtanh, mean, and tanh
fused_hmt_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__device__ inline int get_offset(int b, int c, int h, int w, int B, int C, int H, int W) {
    return ((b * C + c) * H + h) * W + w;
}

__global__ void fused_hmt_kernel(
    const float* input,
    float* output,
    int B,
    int C,
    int H,
    int W,
    float min_val,
    float max_val) {

    int block_idx = blockIdx.x;
    int b = block_idx / C;
    int c = block_idx % C;

    extern __shared__ float shared_sums[];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < H * W; i += blockDim.x) {
        int h = i / W;
        int w = i % W;
        float val = input[get_offset(b, c, h, w, B, C, H, W)];
        float clamped_val = fmaxf(min_val, fminf(max_val, val));
        local_sum += clamped_val;
    }

    shared_sums[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sums[threadIdx.x] += shared_sums[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float total_sum = shared_sums[0];
        float mean = total_sum / (H * W);
        float result = tanhf(mean);
        output[get_offset(b, c, 0, 0, B, C, 1, 1)] = result;
    }
}

torch::Tensor fused_hmt_cuda(torch::Tensor input, float min_val, float max_val) {
    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty({B, C, 1, 1}, input.options());

    const int num_blocks = B * C;
    const int threads_per_block = THREADS_PER_BLOCK;

    fused_hmt_kernel<<<num_blocks, threads_per_block, threads_per_block * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W,
        min_val, max_val
    );

    return output;
}
"""

fused_hmt_cpp_source = (
    "torch::Tensor fused_hmt_cuda(torch::Tensor input, float min_val, float max_val);"
)

# Compile the fused kernel
fused_hmt = load_inline(
    name="fused_hmt",
    cpp_sources=fused_hmt_cpp_source,
    cuda_sources=fused_hmt_source,
    functions=["fused_hmt_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.fused_hmt = fused_hmt
        self.min_val = hardtanh_min
        self.max_val = hardtanh_max

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x).contiguous()
        x = self.fused_hmt.fused_hmt_cuda(x, self.min_val, self.max_val)
        return x

batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 256  
kernel_size  = 3
stride = 1
padding = 1
maxpool_kernel_size = 2
maxpool_stride = 2
hardtanh_min = -1
hardtanh_max = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max]