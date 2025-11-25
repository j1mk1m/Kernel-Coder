import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mean_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_kernel(const float* input, float* output,
                           int input_ndims, const int* input_strides,
                           int reduction_dim, int dim_size,
                           const int* input_dims, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;

    int base_offset = 0;
    int current_idx = idx;

    int stride_red = input_strides[reduction_dim];

    // Compute base_offset
    for (int d = input_ndims - 1; d >= 0; d--) {
        if (d == reduction_dim) continue;
        int dim_size_d = input_dims[d];
        int coord = current_idx % dim_size_d;
        current_idx = current_idx / dim_size_d;
        base_offset += coord * input_strides[d];
    }

    // Compute sum over reduction dimension
    float sum = 0.0f;
    for (int j = 0; j < dim_size; j++) {
        sum += input[base_offset + j * stride_red];
    }

    output[idx] = sum / dim_size;
}

torch::Tensor mean_cuda(torch::Tensor input, int reduction_dim) {
    if (input.device().type() != torch::kCUDA) {
        AT_ERROR("Input must be a CUDA tensor");
    }
    if (reduction_dim < 0 || reduction_dim >= input.dim()) {
        AT_ERROR("Reduction dimension out of bounds");
    }

    int input_ndims = input.dim();
    auto input_dims = input.sizes();
    int dim_size = input.size(reduction_dim);
    int output_size = input.numel() / dim_size;

    auto input_strides = input.stride();
    auto strides_tensor = torch::tensor(input_strides, torch::dtype(torch::kInt32).device(input.device()));
    auto dims_tensor = torch::tensor(input_dims, torch::dtype(torch::kInt32).device(input.device()));

    auto output = torch::empty(output_size, input.options());

    const int threads_per_block = 256;
    int blocks_per_grid = (output_size + threads_per_block - 1) / threads_per_block;

    mean_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input_ndims,
        strides_tensor.data_ptr<int>(),
        reduction_dim,
        dim_size,
        dims_tensor.data_ptr<int>(),
        output_size
    );

    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mean_cuda", &mean_cuda, "Mean reduction CUDA kernel");
}
"""

mean_cuda = load_inline(
    name="mean_cuda",
    cuda_sources=mean_cuda_source,
    functions=["mean_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mean_cuda.mean_cuda(x, self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]