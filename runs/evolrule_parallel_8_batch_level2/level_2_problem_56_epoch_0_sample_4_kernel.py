import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <math.h>

__global__ void sigmoid_sum_kernel(
    const float* input, float* output,
    int batch_size, int hidden_size
) {
    int batch_idx = blockIdx.x;
    extern __shared__ float shared[];
    int tid = threadIdx.x;

    float sum = 0.0f;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = input[batch_idx * hidden_size + i];
        val = 1.0f / (1.0f + expf(-val));
        sum += val;
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
        output[batch_idx] = shared[0];
    }
}

torch::Tensor sigmoid_sum_cuda(
    torch::Tensor input,
    int batch_size,
    int hidden_size
) {
    auto output = torch::empty({batch_size, 1}, input.options());

    const int block_size = 256;
    const int shared_size = block_size * sizeof(float);

    sigmoid_sum_kernel<<<batch_size, block_size, shared_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size
    );

    return output;
}
"""

sigmoid_sum_cpp_source = (
    "torch::Tensor sigmoid_sum_cuda(torch::Tensor input, int batch_size, int hidden_size);"
)

sigmoid_sum = load_inline(
    name="sigmoid_sum",
    cpp_sources=sigmoid_sum_cpp_source,
    cuda_sources=sigmoid_sum_source,
    functions=["sigmoid_sum_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.sigmoid_sum = sigmoid_sum

    def forward(self, x):
        x = self.linear(x)
        batch_size = x.size(0)
        hidden_size = x.size(1)
        x = self.sigmoid_sum.sigmoid_sum_cuda(x, batch_size, hidden_size)
        return x

batch_size = 128
input_size = 32768
hidden_size = 32768

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size]