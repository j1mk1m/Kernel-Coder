import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code for argmax
argmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

__global__ void argmax_kernel(const float* input, int64_t* output, int dim, int B, int D1, int D2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int output_size;
    if (dim == 0) {
        output_size = D1 * D2;
    } else if (dim == 1) {
        output_size = B * D2;
    } else {
        output_size = B * D1;
    }

    if (idx >= output_size)
        return;

    int step, base, dim_size;

    if (dim == 0) {
        dim_size = B;
        step = D1 * D2;
        int d1 = idx / D2;
        int d2 = idx % D2;
        base = d1 * D2 + d2;
    } else if (dim == 1) {
        dim_size = D1;
        step = D2;
        int b = idx / D2;
        int d2 = idx % D2;
        base = b * D1 * D2 + d2;
    } else {
        dim_size = D2;
        step = 1;
        int b = idx / D1;
        int d1 = idx % D1;
        base = b * D1 * D2 + d1 * D2;
    }

    float max_val = -std::numeric_limits<float>::infinity();
    int max_idx = -1;

    for (int i = 0; i < dim_size; ++i) {
        int offset = base + i * step;
        float current_val = input[offset];
        if (current_val > max_val) {
            max_val = current_val;
            max_idx = i;
        }
    }

    output[idx] = max_idx;
}

void argmax_cuda(const torch::Tensor& input, torch::Tensor& output, int dim, int B, int D1, int D2) {
    int output_size;
    if (dim == 0) {
        output_size = D1 * D2;
    } else if (dim == 1) {
        output_size = B * D2;
    } else {
        output_size = B * D1;
    }

    const int block_size = 256;
    const int grid_size = (output_size + block_size - 1) / block_size;

    argmax_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        dim,
        B,
        D1,
        D2
    );
}
"""

argmax_cpp_source = """
void argmax_cuda(const torch::Tensor& input, torch::Tensor& output, int dim, int B, int D1, int D2);
"""

# Load the CUDA extension
argmax = load_inline(
    name="argmax_cuda",
    cpp_sources=argmax_cpp_source,
    cuda_sources=argmax_source,
    functions=["argmax_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return argmax_cuda(x, self.dim)

def argmax_cuda(input: torch.Tensor, dim: int) -> torch.Tensor:
    B = input.size(0)
    D1 = input.size(1)
    D2 = input.size(2)

    if dim == 0:
        output_shape = (D1, D2)
    elif dim == 1:
        output_shape = (B, D2)
    else:
        output_shape = (B, D1)

    output_size = B * D2 if dim == 1 else (D1 * D2 if dim == 0 else B * D1)

    output = torch.empty(output_size, dtype=torch.int64, device=input.device)

    # Call the CUDA function
    argmax.argmax_cuda(input, output, dim, B, D1, D2)

    return output.view(output_shape)

# The get_inputs and get_init_inputs functions remain unchanged
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]