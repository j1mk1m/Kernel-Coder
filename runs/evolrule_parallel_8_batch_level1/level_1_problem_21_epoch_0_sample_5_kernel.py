import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sigmoid_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}

torch::Tensor sigmoid_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    sigmoid_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), 
                                              output.data_ptr<float>(), 
                                              size);

    return output;
}
"""

sigmoid_cuda_header = "torch::Tensor sigmoid_cuda(torch::Tensor input);"

sigmoid_op = load_inline(
    name="sigmoid_cuda",
    cpp_sources=sigmoid_cuda_header,
    cuda_sources=sigmoid_cuda_source,
    functions=["sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid_op = sigmoid_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid_op.sigmoid_cuda(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim, device="cuda")
    return [x]

def get_init_inputs():
    return []