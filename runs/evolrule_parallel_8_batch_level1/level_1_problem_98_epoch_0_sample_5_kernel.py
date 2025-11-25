import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

kl_div_cpp_source = """
#include <torch/extension.h>

torch::Tensor kl_div_cuda(torch::Tensor predictions, torch::Tensor targets, int batch_size);
"""

kl_div_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void kl_div_kernel(const float* predictions, const float* targets, float* result, int num_elements) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    float sum = 0.0f;

    for (int i = tid; i < num_elements; i += blockDim.x * gridDim.x) {
        float p = predictions[i];
        float t = targets[i];
        sum += t * (logf(t) - logf(p));
    }

    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

torch::Tensor kl_div_cuda(torch::Tensor predictions, torch::Tensor targets, int batch_size) {
    auto num_elements = predictions.numel();
    auto result = torch::zeros(1, torch::dtype(torch::kFloat32).device(predictions.device()));

    const int threads_per_block = 256;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    kl_div_kernel<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        result.data_ptr<float>(),
        num_elements
    );

    result[0] = result[0] / batch_size;

    return result;
}
"""

kl_div_cuda = load_inline(
    name="kl_div_cuda",
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_source,
    functions=["kl_div_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_div_cuda = kl_div_cuda

    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        return self.kl_div_cuda.kl_div_cuda(predictions, targets, batch_size)