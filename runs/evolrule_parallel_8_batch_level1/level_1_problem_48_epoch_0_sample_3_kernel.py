import torch
from torch.utils.cpp_extension import load_inline

custom_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_mean_kernel(const float* input, float* output, int dim, int reduction_size, int B, int D, int C) {
    int out_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float partial_sums[];

    float sum = 0.0f;

    if (dim == 0) {
        // Reduction over dim 0 (B)
        int d = out_idx / C;
        int c = out_idx % C;
        int base_offset = d * C + c;
        for (int b = tid; b < B; b += blockDim.x) {
            int input_idx = b * D * C + base_offset;
            sum += input[input_idx];
        }
    } else if (dim == 1) {
        // Reduction over dim 1 (D)
        int b = out_idx / C;
        int c = out_idx % C;
        int base_offset = b * D * C + c;
        for (int d = tid; d < D; d += blockDim.x) {
            int input_idx = base_offset + d * C;
            sum += input[input_idx];
        }
    } else if (dim == 2) {
        // Reduction over dim 2 (C)
        int b = out_idx / D;
        int d = out_idx % D;
        int base_offset = b * D * C + d * C;
        for (int c = tid; c < C; c += blockDim.x) {
            int input_idx = base_offset + c;
            sum += input[input_idx];
        }
    }

    partial_sums[tid] = sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[out_idx] = partial_sums[0] / static_cast<float>(reduction_size);
    }
}

torch::Tensor custom_mean_cuda(torch::Tensor input, int dim) {
    auto input_ = input.contiguous();
    int B = input_.size(0);
    int D = input_.size(1);
    int C = input_.size(2);
    int reduction_size;

    if (dim == 0) reduction_size = B;
    else if (dim == 1) reduction_size = D;
    else if (dim == 2) reduction_size = C;

    auto input_shape = input_.sizes().vec();
    std::vector<int64_t> output_shape;
    for (int i = 0; i < input_.dim(); ++i) {
        if (i != dim) {
            output_shape.push_back(input_shape[i]);
        }
    }

    int64_t output_numel = 1;
    for (int s : output_shape) {
        output_numel *= s;
    }

    auto output = torch::empty({output_numel}, input_.options());

    const int block_size = 256;
    dim3 grid(output_numel);
    dim3 block(block_size);
    size_t shared_mem = block_size * sizeof(float);

    custom_mean_kernel<<<grid, block, shared_mem>>>(
        input_.data_ptr<float>(),
        output.data_ptr<float>(),
        dim,
        reduction_size,
        B, D, C
    );

    output = output.view(output_shape);
    return output;
}
"""

custom_mean = load_inline(
    name='custom_mean',
    cuda_sources=custom_mean_source,
    functions=['custom_mean_cuda'],
    verbose=True
)

class ModelNew(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_mean.custom_mean_cuda(x, self.dim)