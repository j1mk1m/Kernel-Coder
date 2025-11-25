import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Optimized ELU CUDA kernel with vectorization (float4) and fast math
elu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Use fast math approximation for exp
__forceinline__ __device__ float fast_exp(float x) {
    return expf(x); // Alternatively use __expf_approx for faster but less accurate computation
}

__global__ void elu_kernel(float* x, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_size = 4; // Process 4 elements per thread using float4

    // Calculate the starting index and check if within bounds
    int start_idx = idx * vec_size;
    if (start_idx >= size) return;

    // Load 4 elements into a float4
    float4 vals = ((float4*)x)[start_idx];

    // Process each element in the vector
    for (int i = 0; i < vec_size; ++i) {
        float val = vals[i];
        vals[i] = (val > 0.f) ? val : alpha * (fast_exp(val) - 1.f);
    }

    // Store the results back
    ((float4*)x)[start_idx] = vals;
}

torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    auto size = x.numel();
    const int block_size = 256;
    const int vec_size = 4;
    const int adjusted_size = (size + vec_size - 1) / vec_size; // Divide by vector size
    const int num_blocks = (adjusted_size + block_size - 1) / block_size;

    elu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), size, alpha);

    cudaDeviceSynchronize();
    return x;
}
"""

elu_cpp_source = """
torch::Tensor elu_cuda(torch::Tensor x, float alpha);
"""

# Compile the CUDA kernel
elu_op = load_inline(
    name="elu_op",
    cpp_sources=elu_cpp_source,
    cuda_sources=elu_source,
    functions=["elu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
        self.elu_cuda = elu_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu_cuda.elu_cuda(x.contiguous(), self.alpha)

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return [1.0]