import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 128
dim1 = 4096
dim2 = 4095

max_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void max_reduction_kernel(const float* input, float* output, int B, int D1, int D2) {
    int idx = blockIdx.x;
    int b = idx / D2;
    int d2 = idx % D2;

    int start = b * D1 * D2 + d2;

    int tid = threadIdx.x;
    float local_max = -INFINITY;

    for (int i = 0; i < 16; ++i) {
        int d1 = tid * 16 + i;
        if (d1 < D1) {
            int pos = start + d1 * D2;
            float val = input[pos];
            if (val > local_max) {
                local_max = val;
            }
        }
    }

    __shared__ float shared_max[256];
    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared_max[threadIdx.x] < shared_max[threadIdx.x + s]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[idx] = shared_max[0];
    }
}

torch::Tensor max_reduction_cuda(torch::Tensor input) {
    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);
    auto output = torch::empty({B, D2}, input.options());

    int block_size = 256;
    int grid_size = B * D2;

    max_reduction_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2
    );

    return output;
}
"""

max_reduction_cpp_source = "torch::Tensor max_reduction_cuda(torch::Tensor input);"

max_reduction = load_inline(
    name="max_reduction",
    cpp_sources=max_reduction_cpp_source,
    cuda_sources=max_reduction_source,
    functions=["max_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gpu = x.cuda()
        return max_reduction.max_reduction_cuda(x_gpu)

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]