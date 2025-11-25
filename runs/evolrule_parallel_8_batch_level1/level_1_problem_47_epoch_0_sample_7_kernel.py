import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for sum reduction
sum_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void sum_reduction_kernel(
    const T* x,
    T* out,
    int dim,
    int B,
    int D1,
    int D2
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    __shared__ T shared[256];
    T sum = 0.0;
    
    if (dim == 0) {
        int d1 = bid / D2;
        int d2 = bid % D2;
        
        for (int b = tid; b < B; b += blockDim.x) {
            int x_idx = b * D1 * D2 + d1 * D2 + d2;
            sum += x[x_idx];
        }
    } else if (dim == 1) {
        int b = bid / D2;
        int d2 = bid % D2;
        
        for (int i = tid; i < D1; i += blockDim.x) {
            int x_idx = b * D1 * D2 + i * D2 + d2;
            sum += x[x_idx];
        }
    } else { // dim ==2
        int b = bid / D1;
        int d1 = bid % D1;
        
        for (int d2 = tid; d2 < D2; d2 += blockDim.x) {
            int x_idx = b * D1 * D2 + d1 * D2 + d2;
            sum += x[x_idx];
        }
    }
    
    shared[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        if (dim == 0) {
            int out_idx = d1 * D2 + d2;
            out[out_idx] = shared[0];
        } else if (dim == 1) {
            int out_idx = b * D2 + d2;
            out[out_idx] = shared[0];
        } else {
            int out_idx = b * D1 + d1;
            out[out_idx] = shared[0];
        }
    }
}

at::Tensor sum_reduction_cuda(at::Tensor x, int dim) {
    TORCH_CHECK(x.dim() == 3, "Input must be a 3D tensor");
    
    int B = x.size(0);
    int D1 = x.size(1);
    int D2 = x.size(2);
    
    std::vector<int64_t> output_size = {B, D1, D2};
    output_size[dim] = 1;
    at::Tensor output = at::zeros(output_size, x.options());
    
    int block_size = 256;
    int grid_size;
    
    if (dim == 0) {
        grid_size = D1 * D2;
    } else if (dim == 1) {
        grid_size = B * D2;
    } else {
        grid_size = B * D1;
    }
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "sum_reduction_cuda", ([&] {
        sum_reduction_kernel<scalar_t><<<grid_size, block_size>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim,
            B,
            D1,
            D2
        );
    }));
    
    return output;
}
"""

sum_reduction_cpp_source = (
    "at::Tensor sum_reduction_cuda(at::Tensor x, int dim);"
)

# Compile the CUDA kernel
sum_reduction = load_inline(
    name="sum_reduction",
    cpp_sources=sum_reduction_cpp_source,
    cuda_sources=sum_reduction_source,
    functions=["sum_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class SumReductionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        ctx.save_for_backward(x)
        ctx.dim = dim
        return sum_reduction.sum_reduction_cuda(x, dim)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        dim = ctx.dim
        input_shape = x.size()
        grad_input = grad_output.expand(input_shape)
        return grad_input, None

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return SumReductionFunction.apply(x, self.dim)