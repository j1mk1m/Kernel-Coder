import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_sigmoid_sum(
    const float* y,
    float* output,
    int batch_size,
    int hidden_size
) {
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    extern __shared__ float shared_mem[];
    float* sdata = shared_mem;

    float sum = 0.0f;
    for (int j = tid; j < hidden_size; j += threads) {
        float val = y[batch * hidden_size + j];
        float sig = 1.0f / (1.0f + expf(-val));
        sum += sig;
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch] = sdata[0];
    }
}

torch::Tensor fused_sigmoid_sum_cuda(torch::Tensor y) {
    int batch_size = y.size(0);
    int hidden_size = y.size(1);

    auto options = y.options();
    auto output = torch::empty({batch_size}, options);

    int threads = 256;
    int blocks = batch_size;
    int shared_size = threads * sizeof(float);

    fused_sigmoid_sum<<<blocks, threads, shared_size>>>(
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size
    );

    return output.unsqueeze(1);
}
"""

sigmoid_sum_header = """
torch::Tensor fused_sigmoid_sum_cuda(torch::Tensor y);
"""

fused_sigmoid_sum = load_inline(
    name="fused_sigmoid_sum",
    cpp_sources=sigmoid_sum_header,
    cuda_sources=sigmoid_sum_source,
    functions=["fused_sigmoid_sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.fused_sigmoid_sum = fused_sigmoid_sum

    def forward(self, x):
        y = self.linear(x)
        return self.fused_sigmoid_sum.fused_sigmoid_sum_cuda(y)