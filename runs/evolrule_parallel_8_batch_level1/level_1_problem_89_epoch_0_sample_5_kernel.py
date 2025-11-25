import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumsum_kernel(const float* input, float* output, int row_length) {
    int row = blockIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < row_length; ++i) {
        sum += input[row * row_length + i];
        output[row * row_length + i] = sum;
    }
}

void cumsum_kernelLauncher(const float* input, float* output, int rows, int row_length) {
    dim3 blocks(rows);
    dim3 threads(1);
    cumsum_kernel<<<blocks, threads>>>(input, output, row_length);
    cudaDeviceSynchronize();
}
"""

cumsum_cpp_source = """
void cumsum_kernelLauncher(const float* input, float* output, int rows, int row_length);
"""

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cumsum_kernel = load_inline(
            name="cumsum",
            cpp_sources=cumsum_cpp_source,
            cuda_sources=cumsum_source,
            functions=["cumsum_kernelLauncher"],
            verbose=True,
        )

    def forward(self, x):
        rows = x.size(0)
        row_length = x.size(1)
        output = torch.empty_like(x)
        self.cumsum_kernel.cumsum_kernelLauncher(
            x.data_ptr(), output.data_ptr(), rows, row_length
        )
        return output

def get_inputs():
    return [torch.rand(32768, 32768, device='cuda')]

def get_init_inputs():
    return [1]  # dim=1