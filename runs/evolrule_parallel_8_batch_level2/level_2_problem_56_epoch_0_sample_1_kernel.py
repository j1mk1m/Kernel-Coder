import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sigmoid_sum_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void sigmoid_sum_kernel(const float* input, float* output, int batch_size, int hidden_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    extern __shared__ float partial_sums[];
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    float sum = 0.0f;
    for (int j = tid; j < hidden_size; j += n_threads) {
        float val = input[batch_idx * hidden_size + j];
        sum += 1.0f / (1.0f + expf(-val));
    }

    partial_sums[tid] = sum;
    __syncthreads();

    for (int s = n_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch_idx] = partial_sums[0];
    }
}

torch::Tensor sigmoid_sum_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int hidden_size = input.size(1);
    auto output = torch::zeros({batch_size, 1}, input.options());

    int block_size = 512;
    int grid_size = batch_size;

    sigmoid_sum_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size
    );

    return output;
}
"""

sigmoid_sum_cpp_source = "torch::Tensor sigmoid_sum_cuda(torch::Tensor input);"

sigmoid_sum = load_inline(
    name="sigmoid_sum",
    cuda_sources=sigmoid_sum_source,
    cpp_sources=sigmoid_sum_cpp_source,
    functions=["sigmoid_sum_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.sigmoid_sum = sigmoid_sum

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid_sum.sigmoid_sum_cuda(x)
        return x

batch_size = 128
input_size = 32768
hidden_size = 32768

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size]