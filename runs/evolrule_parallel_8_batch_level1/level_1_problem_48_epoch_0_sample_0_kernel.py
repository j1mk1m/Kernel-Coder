import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int dim,
    const int* dims,
    int size_dim,
    int output_size
) {
    int out_idx = blockIdx.x;
    if (out_idx >= output_size) return;

    int base = 0;
    int step = 0;
    int d0 = dims[0];
    int d1 = dims[1];
    int d2 = dims[2];

    switch (dim) {
        case 0:
        {
            int x1 = out_idx / d2;
            int x2 = out_idx % d2;
            base = x1 * d2 + x2;
            step = d1 * d2;
            break;
        }
        case 1:
        {
            int x0 = out_idx / d2;
            int x2 = out_idx % d2;
            base = x0 * (d1 * d2) + x2;
            step = d2;
            break;
        }
        case 2:
        {
            int x0 = out_idx / d1;
            int x1 = out_idx % d1;
            base = x0 * (d1 * d2) + x1 * d2;
            step = 1;
            break;
        }
    }

    __shared__ float shared_sums[256];
    float partial_sum = 0.0f;

    int total_dim_size = size_dim;
    int chunk_size = (total_dim_size + blockDim.x - 1) / blockDim.x;
    int start = threadIdx.x * chunk_size;
    int end = start + chunk_size;

    if (end > total_dim_size) end = total_dim_size;

    for (int x_dim = start; x_dim < end; x_dim++) {
        int input_idx = base + x_dim * step;
        partial_sum += input[input_idx];
    }

    shared_sums[threadIdx.x] = partial_sum;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sums[threadIdx.x] += shared_sums[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mean = shared_sums[0] / (float)size_dim;
        output[out_idx] = mean;
    }
}

torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim) {
    auto dims = input.sizes();
    int d0 = dims[0].item<int>();
    int d1 = dims[1].item<int>();
    int d2 = dims[2].item<int>();
    int dim_size = dims[dim].item<int>();

    auto output_dims = torch::IntArrayRef({d0, d1, d2});
    output_dims.erase(output_dims.begin() + dim);
    auto output = torch::empty(output_dims, input.options());

    int output_size = output.numel();

    dim3 block(256);
    dim3 grid(output_size);

    torch::Tensor dims_tensor = torch::tensor({d0, d1, d2}, torch::device("cpu").dtype(torch::kInt32));

    mean_reduction_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dim,
        dims_tensor.data_ptr<int>(),
        dim_size,
        output_size
    );

    return output;
}
"""

mean_reduction_cpp_source = "torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim);"

mean_reduction = load_inline(
    name="mean_reduction",
    cpp_sources=mean_reduction_cpp_source,
    cuda_sources=mean_reduction_source,
    functions=["mean_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mean_reduction.mean_reduction_cuda(x, self.dim)