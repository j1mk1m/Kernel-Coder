import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmax_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void argmax_kernel(const scalar_t* __restrict__ input, 
                             int64_t* __restrict__ output,
                             int dim,
                             int64_t total_elements,
                             int64_t elements_per_dim,
                             int64_t outer_dim_size,
                             int64_t inner_dim_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int outer = idx / elements_per_dim;
    int inner = idx % inner_dim_size;

    int offset = outer * (elements_per_dim) + inner;

    int max_idx = 0;
    scalar_t max_val = input[offset];
    for (int d = 0; d < dim_size; ++d) {
        int current_offset = outer * (dim_size * inner_dim_size) + d * inner_dim_size + inner;
        if (input[current_offset] > max_val) {
            max_val = input[current_offset];
            max_idx = d;
        }
    }

    output[idx] = max_idx;
}

std::vector<int64_t> get_strides(const torch::Tensor& tensor) {
    auto sizes = tensor.sizes().vec();
    int64_t numel = tensor.numel();
    std::vector<int64_t> strides(tensor.dim(), 1);
    for (int i = tensor.dim() - 2; i >= 0; --i) {
        strides[i] = strides[i+1] * sizes[i+1];
    }
    return strides;
}

torch::Tensor argmax_cuda(torch::Tensor input, int64_t dim) {
    auto input_size = input.sizes();
    auto input_dim = input.dim();
    if (dim < 0) dim += input_dim;
    assert(dim >= 0 && dim < input_dim, "dim out of range");

    // Calculate output shape
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options().dtype(torch::kLong));

    // Compute necessary dimensions for kernel
    auto strides = get_strides(input);
    int64_t dim_size = input.size(dim);
    int64_t total_elements = output.numel();
    int64_t elements_per_dim = input.size(dim) * (dim < input_dim - 1 ? input.size(input_dim - 1) : 1);

    // Launch kernel
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "argmax_cuda", ([&] {
        argmax_kernel<scalar_t><<<grid_size, block_size>>>(
            input.data<scalar_t>(),
            output.data_ptr<int64_t>(),
            dim,
            total_elements,
            elements_per_dim,
            input.size(0),
            input.size(input_dim - 1));
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

argmax_cpp_source = """
torch::Tensor argmax_cuda(torch::Tensor input, int64_t dim);
"""

# Compile the custom CUDA kernel
argmax_extension = load_inline(
    name="argmax_cuda",
    cpp_sources=argmax_cpp_source,
    cuda_sources=argmax_source,
    functions=["argmax_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmax = argmax_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmax.argmax_cuda(x, self.dim)

def get_inputs():
    # Ensure tensors are on CUDA for the kernel
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [torch.tensor(1).cuda()]