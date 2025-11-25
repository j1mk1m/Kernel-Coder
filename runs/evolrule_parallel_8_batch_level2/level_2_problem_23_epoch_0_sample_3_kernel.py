import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the mean operation
mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_kernel(const float* input, float* output, int batch_size, int channels, int D, int H, int W) {
    int batch = blockIdx.x;
    if (batch >= batch_size) return;

    int elements = channels * D * H * W;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    float sum = 0.0f;

    for (int i = tid; i < elements; i += stride) {
        int offset = batch * elements + i;
        sum += input[offset];
    }

    __shared__ float shared[256];
    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch] = shared[0] / static_cast<float>(elements);
    }
}

torch::Tensor mean_cuda(torch::Tensor input) {
    input = input.contiguous();
    int batch_size = input.size(0);
    int channels = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    auto output = torch::empty({batch_size}, input.options());

    dim3 threads(256);
    dim3 blocks(batch_size);

    mean_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        D,
        H,
        W
    );

    return output;
}
"""

mean_cpp = "torch::Tensor mean_cuda(torch::Tensor input);"

# Compile the inline CUDA code for the mean operation
mean_cuda_module = load_inline(
    name="mean_cuda",
    cuda_sources=mean_source,
    cpp_sources=mean_cpp,
    functions=["mean_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size).cuda()
        self.group_norm = nn.GroupNorm(num_groups, out_channels).cuda()
        self.mean_cuda = mean_cuda_module

    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        x = self.mean_cuda.mean_cuda(x)
        return x

def get_inputs():
    batch_size = 128
    in_channels = 3
    D, H, W = 24, 32, 32
    return [torch.rand(batch_size, in_channels, D, H, W).cuda()]

def get_init_inputs():
    in_channels = 3
    out_channels = 24
    kernel_size = 3
    num_groups = 8
    return [in_channels, out_channels, kernel_size, num_groups]