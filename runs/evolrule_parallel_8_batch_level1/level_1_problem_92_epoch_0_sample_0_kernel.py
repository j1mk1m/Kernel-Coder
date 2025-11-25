import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for shifting inclusive cumsum to exclusive
shift_exclusive_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void shift_exclusive_cumsum_kernel(
    const float* input,
    float* output,
    int dim_size,
    int row_stride
) {
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (tid >= dim_size) return;

    int row_offset = row_idx * row_stride;
    int input_offset = row_offset + tid;
    int output_offset = row_offset + tid;

    if (tid == 0) {
        output[output_offset] = 0.0f;
    } else {
        output[output_offset] = input[input_offset - 1];
    }
}

torch::Tensor shift_exclusive_cumsum_cuda(torch::Tensor input) {
    auto size = input.sizes();
    int dim = input.dim() - 1;  // Assuming the last dimension
    int dim_size = size[dim];
    int row_stride = dim_size;  // stride for the last dimension
    int row_count = input.numel() / dim_size;

    auto output = torch::zeros_like(input);

    dim3 blocks(row_count);
    dim3 threads(dim_size);

    shift_exclusive_cumsum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dim_size,
        row_stride
    );

    return output;
}
"""

shift_exclusive_cumsum_cpp_source = (
    "torch::Tensor shift_exclusive_cumsum_cuda(torch::Tensor input);"
)

# Compile the CUDA kernel
shift_exclusive_cumsum = load_inline(
    name="shift_exclusive_cumsum",
    cpp_sources=shift_exclusive_cumsum_cpp_source,
    cuda_sources=shift_exclusive_cumsum_source,
    functions=["shift_exclusive_cumsum_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.shift = shift_exclusive_cumsum

    def forward(self, x):
        # Compute inclusive cumsum using PyTorch's optimized implementation
        inclusive = torch.cumsum(x, dim=self.dim)
        # Shift to get exclusive cumsum using the custom CUDA kernel
        return self.shift.shift_exclusive_cumsum_cuda(inclusive)

def get_inputs():
    batch_size = 32768
    input_shape = (32768,)
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [1]  # Example uses dim=1