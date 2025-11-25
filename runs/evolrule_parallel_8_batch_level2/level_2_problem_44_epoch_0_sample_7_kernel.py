import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for scalar multiplication and global average pooling
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void fused_multiply_and_mean_kernel(
    const T* __restrict__ input,
    T scalar,
    T* output,
    int N, int C, int H, int W,
    int output_size) {

    // Each block handles a (n, c) pair
    int n = blockIdx.x;
    int c = blockIdx.y;

    // Thread index within the block
    int tid = threadIdx.x;
    T sum = static_cast<T>(0);

    // Iterate over all spatial elements (H * W)
    for (int idx = tid; idx < H * W; idx += blockDim.x) {
        int h = idx / W;
        int w = idx % W;

        // Compute input index
        int input_offset = n * C * H * W + c * H * W + h * W + w;
        T val = input[input_offset] * scalar;
        sum += val;
    }

    // Use shared memory for reduction
    extern __shared__ T sdata[];
    sdata[tid] = sum;
    __syncthreads();

    // Block-wise reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Output index for (n, c, 0, 0)
        int output_offset = n * C + c;
        output[output_offset] = sdata[0] / static_cast<T>(H * W);
    }
}

at::Tensor fused_multiply_and_mean_cuda(at::Tensor input, float multiplier) {
    AT_ASSERT(input.dim() == 4);
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Output shape: (N, C, 1, 1)
    at::Tensor output = at::zeros({N, C, 1, 1}, input.options());

    const int threads_per_block = 256;
    dim3 blocks(N, C);
    dim3 threads(threads_per_block);

    // Shared memory size for reduction
    int shared_size = threads_per_block * sizeof(float);

    // Launch kernel
    fused_multiply_and_mean_kernel<float><<<blocks, threads, shared_size, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        multiplier,
        output.data_ptr<float>(),
        N, C, H, W,
        output.numel()
    );

    AT_CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

# Header for the CUDA function
fused_kernel_header = """
#include <torch/extension.h>
at::Tensor fused_multiply_and_mean_cuda(at::Tensor input, float multiplier);
"""

# Load the fused kernel
fused_multiply_and_mean = load_inline(
    name="fused_multiply_and_mean",
    cpp_sources=fused_kernel_header,
    cuda_sources=fused_kernel_source,
    functions=["fused_multiply_and_mean_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = multiplier
        self.fused_op = fused_multiply_and_mean  # Load the fused CUDA kernel

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_multiply_and_mean_cuda(x, self.multiplier)
        return x

# The get_inputs and get_init_inputs remain unchanged from the original problem's setup
batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier]