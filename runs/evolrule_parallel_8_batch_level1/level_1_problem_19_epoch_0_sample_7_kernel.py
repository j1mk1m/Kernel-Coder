import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = max(input[idx], 0.f);
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    relu_kernel<<<blocks, threads, 0, stream>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

relu_cpp_source = """
torch::Tensor relu_cuda(torch::Tensor input);
"""

relu = load_inline(
    name="relu",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_flags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu.relu_cuda(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []