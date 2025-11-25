import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

reverse_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <int BlockSize>
__global__ void reverse_cumsum_kernel(float* input, float* output, int L) {
    extern __shared__ float s_data[];
    int tid = threadIdx.x;
    if (tid >= L) return;
    s_data[tid] = input[blockIdx.x * L + tid];
    __syncthreads();

    // Compute inclusive scan in reverse
    for (int d = 1; d < L; d <<= 1) {
        int ai = tid + d;
        if (ai < L) {
            s_data[tid] += s_data[ai];
        }
        __syncthreads();
    }

    output[blockIdx.x * L + tid] = s_data[tid];
}

torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int L) {
    const int block_size = 1024;
    const int grid_size = input.size(0); // One block per batch element
    auto output = torch::empty_like(input);
    int smem_size = L * sizeof(float);

    reverse_cumsum_kernel<block_size><<<grid_size, block_size, smem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        L
    );

    return output;
}
"""

reverse_cumsum_cpp_source = (
    "torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int L);"
)

reverse_cumsum = load_inline(
    name="reverse_cumsum",
    cpp_sources=reverse_cumsum_cpp_source,
    cuda_sources=reverse_cumsum_source,
    functions=["reverse_cumsum_cuda"],
    verbose=True,
    extra_cflags=["-D_FORCE_INLINES"],
    extra_cuda_cflags=["-lineinfo"],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reverse_cumsum = reverse_cumsum

    def forward(self, x):
        L = x.size(self.dim)
        return self.reverse_cumsum.reverse_cumsum_cuda(x, L)