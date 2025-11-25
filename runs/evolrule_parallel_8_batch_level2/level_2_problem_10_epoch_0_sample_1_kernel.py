import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for hardtanh, mean, and tanh
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_hardtanh_mean_tanh_kernel(
    const float* input, float* output,
    int B, int C, int H, int W) {

    int b = blockIdx.x / C;
    int c = blockIdx.x % C;

    extern __shared__ float shared[];
    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int idx = tid; idx < H * W; idx += blockDim.x) {
        int h = idx / W;
        int w = idx % W;
        float val = input[ b * C * H * W + c * H * W + h * W + w ];
        val = __fmaxf(__fminf(val, 1.0f), -1.0f); // Apply hardtanh
        sum += val;
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
        float mean = shared[0] / (H * W);
        output[ b * C + c ] = __tanhf(mean); // Apply tanh
    }
}

torch::Tensor fused_operation_cuda(
    torch::Tensor input, // shape (B, C, H, W)
    int H, int W) {

    int B = input.size(0);
    int C = input.size(1);

    auto output = torch::zeros({B, C, 1, 1}, input.options());

    int block_size = 256;
    int num_blocks = B * C;

    fused_hardtanh_mean_tanh_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W
    );

    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_operation_cuda(torch::Tensor input, int H, int W);
"""

# Load the fused CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_operation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=maxpool_kernel_size, stride=maxpool_stride
        )
        self.fused_op = fused_op  # Reference to the fused CUDA kernel

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = x.contiguous()  # Ensure contiguous memory for CUDA kernel
        H = x.size(2)
        W = x.size(3)
        x = self.fused_op.fused_operation_cuda(x, H, W)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding,
            maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max]