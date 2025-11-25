import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 16384
input_size = 2048
hidden_size = 4096
output_size = 1024

# Custom LogSumExp CUDA kernel
logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void logsumexp_forward_kernel(
    const float* input, float* output, int batch_size, int channels) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        float val = exp(input[row * channels + i]);
        sum += val;
    }

    // Shared memory reduction
    __shared__ float shared[256];  // Assuming blockDim.x <= 256
    int tid = threadIdx.x;
    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = std::log(shared[0]);
    }
}

torch::Tensor logsumexp_forward_cuda(torch::Tensor input) {
    input = input.contiguous();
    int batch_size = input.size(0);
    int channels = input.size(1);
    auto output = torch::empty({batch_size}, input.options());

    const int block_size = 256;
    const dim3 grid(batch_size);
    const dim3 block(block_size);

    logsumexp_forward_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels);

    return output;
}
"""

# Custom Sigmoid CUDA kernel
sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}

torch::Tensor sigmoid_forward_cuda(torch::Tensor input) {
    input = input.contiguous();
    int size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    sigmoid_forward_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

# Compile the custom kernels
logsumexp = load_inline(
    name="logsumexp",
    cuda_sources=logsumexp_source,
    functions=["logsumexp_forward_cuda"],
    verbose=True
)

sigmoid = load_inline(
    name="sigmoid",
    cuda_sources=sigmoid_source,
    functions=["sigmoid_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = sigmoid
        self.logsumexp = logsumexp

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid.sigmoid_forward_cuda(x)
        x = self.linear2(x)
        x = self.logsumexp.logsumexp_forward_cuda(x)
        return x

def get_init_inputs():
    return [input_size, hidden_size, output_size]

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]