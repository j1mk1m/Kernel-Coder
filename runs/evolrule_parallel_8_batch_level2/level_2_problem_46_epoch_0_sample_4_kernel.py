import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused element-wise CUDA kernel
fused_elementwise_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void fused_elementwise_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const float subtract1,
    const float subtract2,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t val = input[idx] - subtract1;
        val = tanh(val);
        output[idx] = val - subtract2;
    }
}

torch::Tensor fused_elementwise_cuda(torch::Tensor input,
                                    float subtract1,
                                    float subtract2) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_elementwise_cuda", ([&] {
        fused_elementwise_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            subtract1,
            subtract2,
            size
        );
    }));

    return output;
}
"""

# Compile the fused element-wise kernel
fused_elementwise = load_inline(
    name="fused_elementwise",
    cpp_sources="",
    cuda_sources=fused_elementwise_source,
    functions=["fused_elementwise_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)
        self.fused_elementwise = fused_elementwise  # Reference to the CUDA function

    def forward(self, x):
        x = self.conv(x)
        # Apply fused element-wise operations in a single kernel
        x = self.fused_elementwise.fused_elementwise_cuda(
            x,
            self.subtract1_value,
            self.subtract2_value
        )
        x = self.avgpool(x)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]