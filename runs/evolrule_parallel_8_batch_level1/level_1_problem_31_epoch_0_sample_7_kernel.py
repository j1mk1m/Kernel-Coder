import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ELU activation
elu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void elu_kernel(const float* input, float* output, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = (x > 0) ? x : alpha * (expf(x) - 1);
    }
}

torch::Tensor elu_cuda(torch::Tensor input, float alpha) {
    int64_t size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    elu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), alpha, size);
    
    return output;
}
"""

elu_cuda_header = """
torch::Tensor elu_cuda(torch::Tensor input, float alpha);
"""

# Compile the inline CUDA code for ELU
elu_cuda_module = load_inline(
    name="elu_cuda",
    cpp_sources=elu_cuda_header,
    cuda_sources=elu_cuda_source,
    functions=["elu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
        self.elu_cuda = elu_cuda_module.elu_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move input to GPU, apply CUDA kernel, then move back to CPU (if needed)
        x_gpu = x.cuda()
        output_gpu = self.elu_cuda(x_gpu, self.alpha)
        return output_gpu.cpu()

# Keep get_inputs and get_init_inputs identical to the original
def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return [1.0]  # Provide alpha value for initialization