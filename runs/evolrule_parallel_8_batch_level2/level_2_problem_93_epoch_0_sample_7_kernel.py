import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

        # Define the fused CUDA kernel for element-wise operations after conv_transpose
        fused_operations_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256

__global__ void fused_operations_kernel(
    const float* input,
    float* output,
    int size,
    float add_val,
    float multiply_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] + add_val;
        x = min(x, 0.0f); // Clamp to maximum 0.0
        // Compute GELU approximation
        const float sqrt_2_over_pi = 0.7978845608f;
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        float tanh_val = tanhf(inner);
        float gelu = x * 0.5f * (1.0f + tanh_val);
        output[idx] = gelu * multiply_val;
    }
}

torch::Tensor fused_operations_cuda(torch::Tensor input, float add_val, float multiply_val) {
    input = input.contiguous();
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = THREADS_PER_BLOCK;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_operations_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        add_val,
        multiply_val
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}
"""

        fused_operations_cpp_source = """
torch::Tensor fused_operations_cuda(torch::Tensor input, float add_val, float multiply_val);
"""

        # Compile the fused CUDA kernel
        self.fused_ops = load_inline(
            name="fused_operations",
            cpp_sources=fused_operations_cpp_source,
            cuda_sources=fused_operations_source,
            functions=["fused_operations_cuda"],
            verbose=True,
            extra_cflags=[""],
            extra_ldflags=[""],
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        return self.fused_ops.fused_operations_cuda(x, self.add_value, self.multiply_value)

# The following functions are kept as per the original architecture
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]