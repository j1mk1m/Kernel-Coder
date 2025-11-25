import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        output[idx] = fmaxf(-1.0f, fminf(val, 1.0f));
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor input) {
    auto n = input.numel();
    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    dim3 blocks((n + threads_per_block - 1) / threads_per_block);
    dim3 threads(threads_per_block);

    hardtanh_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);

    return output;
}
"""

hardtanh_cpp_source = (
    "torch::Tensor hardtanh_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code
hardtanh = load_inline(
    name="hardtanh",
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_source,
    functions=["hardtanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardtanh = hardtanh

    def forward(self, x):
        return self.hardtanh.hardtanh_cuda(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []