import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

hardtanh_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void hardtanh_forward_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, 
                                       const float min_val, const float max_val, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t val = input[idx];
        scalar_t min = static_cast<scalar_t>(min_val);
        scalar_t max = static_cast<scalar_t>(max_val);
        if (val < min) {
            output[idx] = min;
        } else if (val > max) {
            output[idx] = max;
        } else {
            output[idx] = val;
        }
    }
}

template <typename scalar_t>
torch::Tensor hardtanh_forward_cuda(torch::Tensor input, float min_val, float max_val) {
    auto output = torch::empty_like(input);
    const int block_size = 256;
    int size = input.numel();
    const int num_blocks = (size + block_size - 1) / block_size;

    hardtanh_forward_kernel<scalar_t><<<num_blocks, block_size>>>(
        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), 
        min_val, max_val, size);

    return output;
}

torch::Tensor hardtanh_forward(torch::Tensor input, float min_val, float max_val) {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardtanh_forward", ([&] {
        return hardtanh_forward_cuda<scalar_t>(input, min_val, max_val);
    }));
    return torch::Tensor(); // Unreachable but required for compilation
}
"""

hardtanh_cpp_source = """
torch::Tensor hardtanh_forward(torch::Tensor input, float min_val, float max_val);
"""

hardtanh_extension = load_inline(
    name="hardtanh_cuda",
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_cuda_source,
    functions=["hardtanh_forward"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA -DCUDA_HAS_FP16=1 -D__CUDA_fp16__"],
    extra_cuda_cflags=["-gencode=arch=compute_80,code=sm_80"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.min_val = -1.0
        self.max_val = 1.0
        self.hardtanh_forward = hardtanh_extension.hardtanh_forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardtanh_forward(x, self.min_val, self.max_val)