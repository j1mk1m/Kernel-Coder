import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sigmoid_sum_kernel(
    const float* __restrict__ input,
    float* output,
    int batch_size,
    int hidden_size
) {
    int sample_idx = blockIdx.x;
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x = input[sample_idx * hidden_size + i];
        float s = 1.0f / (1.0f + expf(-x));
        sum += s;
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
        output[sample_idx] = shared[0];
    }
}

torch::Tensor sigmoid_sum_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto hidden_size = input.size(1);
    auto output = torch::empty({batch_size}, input.options());

    const int block_size = 256;
    const int num_blocks = batch_size;

    AT_CHECK(input.is_cuda(), "Input must be on CUDA device");

    sigmoid_sum_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size
    );

    return output.view({batch_size, 1});
}
"""

sigmoid_sum_header = """
torch::Tensor sigmoid_sum_cuda(torch::Tensor input);
"""

sigmoid_sum = load_inline(
    name="sigmoid_sum",
    cpp_sources=sigmoid_sum_header,
    cuda_sources=sigmoid_sum_source,
    functions=["sigmoid_sum_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.sigmoid_sum = sigmoid_sum  # The loaded module

    def forward(self, x):
        x = self.linear(x)
        return self.sigmoid_sum.sigmoid_sum_cuda(x)