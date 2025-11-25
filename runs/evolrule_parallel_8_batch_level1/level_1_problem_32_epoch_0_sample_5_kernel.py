import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 4;

    for (int i = 0; i < 4; ++i) {
        int pos = idx + i;
        if (pos < n) {
            float v = input[pos];
            v = fmaxf(-1.0f, v);
            v = fminf(1.0f, v);
            output[pos] = v;
        }
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor input) {
    const int n = input.numel();
    auto output = torch::empty_like(input);

    const int threadsPerBlock = 256;
    const int elementsPerThread = 4;
    const int threadsPerBlockTotal = threadsPerBlock * elementsPerThread;

    const int blocksPerGrid = (n + threadsPerBlockTotal - 1) / threadsPerBlockTotal;

    dim3 threads(threadsPerBlock);
    dim3 blocks;

    blocks.x = 65535;
    blocks.y = (blocksPerGrid + blocks.x - 1) / blocks.x;
    blocks.y = std::min(blocks.y, 65535);
    blocks.z = 1;

    hardtanh_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));

    return output;
}
"""

hardtanh_cpp_source = """
torch::Tensor hardtanh_cuda(torch::Tensor input);
"""

hardtanh_extension = load_inline(
    name="hardtanh_cuda",
    cuda_sources=hardtanh_source,
    cpp_sources=hardtanh_cpp_source,
    functions=["hardtanh_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hardtanh_extension.hardtanh_cuda(x)