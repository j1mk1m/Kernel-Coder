import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

scale_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scale_add_kernel(const float* input, float scale, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * scale;
    }
}

torch::Tensor scale_add_cuda(torch::Tensor input, float scale) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    scale_add_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), scale, output.data_ptr<float>(), size);

    return output;
}
"""

scale_add_cpp_source = "torch::Tensor scale_add_cuda(torch::Tensor input, float scale);"

scale_add = load_inline(
    name="scale_add",
    cpp_sources=scale_add_cpp_source,
    cuda_sources=scale_add_source,
    functions=["scale_add_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.scale_add = scale_add  # The compiled CUDA function

    def forward(self, x):
        matmul_out = self.matmul(x)
        return self.scale_add.scale_add_cuda(matmul_out, self.scaling_factor + 1.0)