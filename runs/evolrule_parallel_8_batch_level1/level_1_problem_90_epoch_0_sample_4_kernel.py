import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 32768
input_shape = (32768,)
dim = 1

cumprod_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

const int BLOCK_SIZE = 1024;

__global__ void cumprod_kernel(const float* input, float* output, int N, int batch_size) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = BLOCK_SIZE;
    int segment_length = (N + num_threads - 1) / num_threads;
    int start = tid * segment_length;
    int end = min((tid + 1) * segment_length, N);

    float segment_prod = 1.0f;
    for (int i = start; i < end; ++i) {
        segment_prod *= input[row * N + i];
    }

    __shared__ float segment_prods[BLOCK_SIZE];
    segment_prods[tid] = segment_prod;
    __syncthreads();

    for (int d = 1; d < num_threads; d *= 2) {
        if (tid >= d) {
            segment_prods[tid] *= segment_prods[tid - d];
        }
        __syncthreads();
    }

    __shared__ float global_prefix[BLOCK_SIZE];
    if (tid == 0) {
        global_prefix[tid] = 1.0f;
    } else {
        global_prefix[tid] = segment_prods[tid - 1];
    }
    __syncthreads();

    float local_cumprod = 1.0f;
    for (int i = start; i < end; ++i) {
        local_cumprod *= input[row * N + i];
        output[row * N + i] = global_prefix[tid] * local_cumprod;
    }
}

torch::Tensor cumprod_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int N = input.size(1);
    auto output = torch::empty_like(input);
    int grid_size = batch_size;
    int block_size = BLOCK_SIZE;

    cumprod_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        batch_size
    );

    return output;
}
"""

cpp_source = """
torch::Tensor cumprod_cuda(torch::Tensor input);
"""

cumprod_cuda = load_inline(
    name="cumprod_cuda",
    cpp_sources=cpp_source,
    cuda_sources=cumprod_source,
    functions=["cumprod_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        assert self.dim == 1, "Custom kernel only supports dim=1"
        return cumprod_cuda(x)

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [dim]