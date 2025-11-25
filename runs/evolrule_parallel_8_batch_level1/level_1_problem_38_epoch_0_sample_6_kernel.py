import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l1_normalize_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void l1_normalize_kernel(const float* input, float* output, int batch_size, int dim) {
    extern __shared__ float shared_sums[];
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int num_elements_per_thread = (dim + block_size - 1) / block_size;
    int start = tid * num_elements_per_thread;
    int end = min(start + num_elements_per_thread, dim);

    float partial_sum = 0.0f;
    for (int i = start; i < end; ++i) {
        float val = input[batch_idx * dim + i];
        partial_sum += fabs(val);
    }

    shared_sums[tid] = partial_sum;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] += shared_sums[tid + s];
        }
        __syncthreads();
    }

    float total_sum = shared_sums[0];
    float mean = total_sum / dim;

    __syncthreads();

    for (int i = start; i < end; ++i) {
        float val = input[batch_idx * dim + i];
        output[batch_idx * dim + i] = val / mean;
    }
}

torch::Tensor l1_normalize_cuda(torch::Tensor input) {
    input = input.contiguous();
    int batch_size = input.size(0);
    int dim = input.size(1);
    auto output = torch::empty_like(input);

    int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(block_size);
    size_t shared_mem = block_size * sizeof(float);

    l1_normalize_kernel<<<grid, block, shared_mem, torch::cuda::current_stream()>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);

    return output;
}
"""

l1_normalize_cpp_source = """
torch::Tensor l1_normalize_cuda(torch::Tensor input);
"""

l1_normalize = load_inline(
    name="l1_normalize",
    cpp_sources=l1_normalize_cpp_source,
    cuda_sources=l1_normalize_source,
    functions=["l1_normalize_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_normalize = l1_normalize

    def forward(self, x):
        return self.l1_normalize.l1_normalize_cuda(x)