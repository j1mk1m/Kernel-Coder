import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* __restrict__ in, float* __restrict__ out, float slope, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = __ldg(in + idx);
        float max_val = fmaxf(x, 0.f);
        out[idx] = max_val + slope * (x - max_val);
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor in, float slope) {
    auto out = torch::empty_like(in);
    int size = in.numel();
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    auto stream = at::cuda::getCurrentCUDAStream();
    leaky_relu_kernel<<<grid_size, block_size, 0, stream>>>(
        in.data_ptr<float>(), out.data_ptr<float>(), slope, size
    );
    return out;
}
"""

leaky_relu_header = """
torch::Tensor leaky_relu_cuda(torch::Tensor in, float slope);
"""

leaky_relu = load_inline(
    name="leaky_relu",
    cpp_sources=leaky_relu_header,
    cuda_sources=leaky_relu_source,
    functions=["leaky_relu_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
        self.leaky_relu = leaky_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu.leaky_relu_cuda(x, self.negative_slope)