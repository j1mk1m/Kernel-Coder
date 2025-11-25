import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mean_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_kernel(const float* input, float* output, int B, int D, int D2) {
    int block_idx = blockIdx.x;
    int i = block_idx / D2;
    int j = block_idx % D2;

    float sum = 0.0f;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int k = tid; k < D; k += stride) {
        int input_idx = i * D * D2 + k * D2 + j;
        sum += input[input_idx];
    }

    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[block_idx] = shared_sum[0] / D;
    }
}

torch::Tensor mean_cuda(torch::Tensor input, int dim) {
    // Compute output shape
    auto output_shape = input.sizes().vec();
    output_shape.erase(output_shape.begin() + dim);
    auto output = torch::empty(output_shape, input.options());

    int B = input.size(0);
    int D = input.size(1);
    int D2 = input.size(2);

    int grid_size = B * D2;
    int block_size = 256;

    mean_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D, D2
    );

    return output;
}
"""

mean_cpp_source = """
torch::Tensor mean_cuda(torch::Tensor input, int dim);
"""

mean_cuda_module = load_inline(
    name="mean_cuda",
    cpp_sources=mean_cpp_source,
    cuda_sources=mean_kernel_source,
    functions=["mean_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mean_cuda_module.mean_cuda(x, self.dim)