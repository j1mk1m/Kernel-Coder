import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

log_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void log_softmax_kernel(const float* x, float* out, int batch_size, int dim) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    int tid = threadIdx.x;
    int elements_per_thread = (dim + blockDim.x - 1) / blockDim.x;
    int start = tid * elements_per_thread;
    int end = min(start + elements_per_thread, dim);

    // Compute local max
    float local_max = -FLT_MAX;
    for (int i = start; i < end; ++i) {
        float val = x[row * dim + i];
        if (val > local_max) {
            local_max = val;
        }
    }

    // Block reduction for max
    __shared__ float shared_max[1024];
    shared_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid] < shared_max[tid + s]) {
                shared_max[tid] = shared_max[tid + s];
            }
        }
        __syncthreads();
    }

    float global_max = shared_max[0];

    // Compute local sum of exp(x_i - global_max)
    float local_sum = 0.0f;
    for (int i = start; i < end; ++i) {
        float val = x[row * dim + i] - global_max;
        local_sum += expf(val);
    }

    // Block reduction for sum
    __shared__ float shared_sum[1024];
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    float total_sum = shared_sum[0];
    float log_sum = logf(total_sum);

    // Compute the output
    for (int i = start; i < end; ++i) {
        float val = x[row * dim + i] - global_max;
        out[row * dim + i] = val - log_sum;
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor x) {
    auto output = torch::empty_like(x);

    int batch_size = x.size(0);
    int dim = x.size(1);

    dim3 block(1024);
    dim3 grid(batch_size);

    log_softmax_kernel<<<grid, block>>>(x.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);

    return output;
}
"""

log_softmax_cpp_source = """
torch::Tensor log_softmax_cuda(torch::Tensor x);
"""

log_softmax = load_inline(
    name="log_softmax",
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_source,
    functions=["log_softmax_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return log_softmax.log_softmax_cuda(x)
        else:
            return torch.log_softmax(x, dim=self.dim)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []