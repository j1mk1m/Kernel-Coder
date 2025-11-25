import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

def _get_output_shape(input_shape, dim):
    # Handle negative dimensions
    dim = dim if dim >= 0 else len(input_shape) + dim
    output_shape = list(input_shape)
    del output_shape[dim]
    return output_shape

argmax_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void argmax_kernel(const scalar_t* __restrict__ input, int64_t* output,
                             int total_elements, int input_dims, int dim_size,
                             int output_strides[], int input_strides[],
                             int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Compute the output coordinates
    int out_coords[input_dims - 1];
    int temp = idx;
    for (int i = input_dims - 2; i >= 0; --i) {
        out_coords[i] = temp % output_strides[i];
        temp /= output_strides[i];
    }

    // Compute the input index range along the dimension
    int input_offset = 0;
    int dim_pos = 0;
    for (int d = 0; d < input_dims; ++d) {
        if (d == dim) {
            dim_pos = input_offset;
            input_offset += dim_size * input_strides[d];
        } else {
            input_offset += out_coords[d >= dim ? d - 1 : d] * input_strides[d];
        }
    }

    // Find the maximum value's index in the current slice
    scalar_t max_val = -INFINITY;
    int max_idx = 0;
    for (int i = 0; i < dim_size; ++i) {
        int pos = input_offset + i * input_strides[dim];
        scalar_t val = input[pos];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    output[idx] = max_idx;
}

std::tuple<torch::Tensor, torch::Tensor> argmax_cuda(torch::Tensor input, int64_t dim) {
    const int64_t* input_shape = input.sizes().data();
    int input_dims = input.dim();
    dim = dim < 0 ? dim + input_dims : dim;
    int dim_size = input.size(dim);

    auto output_shape = _get_output_shape(input.sizes().vec(), dim);
    auto output = torch::empty(output_shape, input.options().dtype(torch::kLong)).cuda();

    int64_t total_elements = output.numel();

    // Compute strides for input and output
    int64_t input_strides[input_dims];
    int64_t output_strides[input_dims - 1];
    input.strides(input_strides);
    torch::Tensor output_cpu = output.cpu(); // To compute strides on CPU
    output_cpu.strides(output_strides);

    const int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    // Launch kernel with appropriate template
    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "argmax_kernel", ([&] {
        argmax_kernel<scalar_t><<<grid_size, block_size>>>(
            input.data<scalar_t>(), output.data_ptr<int64_t>(),
            total_elements, input_dims, dim_size,
            output_strides, input_strides, dim);
    }));

    cudaDeviceSynchronize();
    return std::make_tuple(output, input); // Returning input to avoid optimization removal
}
"""

argmax_kernel_cpp_source = """
std::tuple<torch::Tensor, torch::Tensor> argmax_cuda(torch::Tensor input, int64_t dim);
"""

argmax_cuda = load_inline(
    name="argmax_cuda",
    cpp_sources=argmax_kernel_cpp_source,
    cuda_sources=argmax_kernel_source,
    functions=["argmax_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_cuda_cflags=["-arch=sm_75"],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmax_cuda = argmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure the input is on the correct device
        x = x.cuda()
        # Call the CUDA kernel and return only the output
        output, _ = self.argmax_cuda(x, self.dim)
        return output.long()  # Ensure output is long type as per torch.argmax

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]