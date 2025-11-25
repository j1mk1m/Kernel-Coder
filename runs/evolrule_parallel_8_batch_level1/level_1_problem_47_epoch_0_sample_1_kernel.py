import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

reduce_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void reduce_sum_kernel(
    const float* in_ptr,
    float* out_ptr,
    int B,
    int D1,
    int D2
) {
    int out_idx = blockIdx.x;
    int i = out_idx / D2;
    int j = out_idx % D2;

    int tid = threadIdx.x;
    int total_elements = D1;
    int per_thread = (total_elements + blockDim.x - 1) / blockDim.x;

    float sum = 0.0f;
    for (int k = tid; k < total_elements; k += blockDim.x) {
        int in_offset = i * D1 * D2 + k * D2 + j;
        sum += in_ptr[in_offset];
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
        out_ptr[out_idx] = shared[0];
    }
}

torch::Tensor reduce_sum_cuda(torch::Tensor x) {
    int B = x.size(0);
    int D1 = x.size(1);
    int D2 = x.size(2);
    auto out = torch::zeros({B, 1, D2}, x.options());

    const int block_size = 256;
    const int grid_size = B * D2;

    reduce_sum_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        D1,
        D2
    );

    return out;
}
"""

reduce_sum_cpp_source = (
    "torch::Tensor reduce_sum_cuda(torch::Tensor x);"
)

reduce_sum = load_inline(
    name="reduce_sum",
    cuda_sources=reduce_sum_source,
    cpp_sources=reduce_sum_cpp_source,
    functions=["reduce_sum_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduce_sum = reduce_sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduce_sum.reduce_sum_cuda(x)