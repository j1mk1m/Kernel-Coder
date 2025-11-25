import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmin_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

template <typename T>
__global__ void argmin_kernel(const T* input, int* output, int B, int D1, int D2, int dim) {
    int output_linear = blockIdx.x;
    int tid = threadIdx.x;

    int reduction_size;
    int start_idx;
    int step;

    if (dim == 0) {
        int d_out1 = output_linear / D2;
        int d_out2 = output_linear % D2;
        start_idx = d_out1 * D2 + d_out2;
        step = D1 * D2;
        reduction_size = B;
    } else if (dim == 1) {
        int b = output_linear / D2;
        int d2 = output_linear % D2;
        start_idx = b * D1 * D2 + d2;
        step = D2;
        reduction_size = D1;
    } else if (dim == 2) {
        int b = output_linear / D1;
        int d1 = output_linear % D1;
        start_idx = b * D1 * D2 + d1 * D2;
        step = 1;
        reduction_size = D2;
    }

    float min_val = std::numeric_limits<float>::max();
    int min_idx = -1;

    int elements_per_thread = (reduction_size + blockDim.x - 1) / blockDim.x;

    for (int i = 0; i < elements_per_thread; ++i) {
        int pos = tid * elements_per_thread + i;
        if (pos < reduction_size) {
            int idx = start_idx + pos * step;
            float val = static_cast<float>(input[idx]);
            if (val < min_val) {
                min_val = val;
                min_idx = pos;
            } else if (val == min_val) {
                if (pos < min_idx) {
                    min_idx = pos;
                }
            }
        }
    }

    __shared__ float s_min[256];
    __shared__ int s_idx[256];

    s_min[tid] = min_val;
    s_idx[tid] = min_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_min[tid + s] < s_min[tid]) {
                s_min[tid] = s_min[tid + s];
                s_idx[tid] = s_idx[tid + s];
            } else if (s_min[tid + s] == s_min[tid]) {
                if (s_idx[tid + s] < s_idx[tid]) {
                    s_idx[tid] = s_idx[tid + s];
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[output_linear] = s_idx[0];
    }
}

torch::Tensor argmin_cuda(torch::Tensor input, int dim) {
    auto B = input.size(0);
    auto D1 = input.size(1);
    auto D2 = input.size(2);
    auto output_size = input.sizes().vec();
    output_size.erase(output_size.begin() + dim);
    auto output = torch::empty(output_size, torch::dtype(torch::kInt32).device(input.device()));

    const int block_size = 256;
    const int grid_size = output.numel();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmin_cuda", ([&] {
        argmin_kernel<scalar_t>
            <<<grid_size, block_size>>>(
                input.data<scalar_t>(),
                output.data_ptr<int>(),
                B, D1, D2, dim
            );
    }));

    return output;
}
"""

argmin_cpp_source = """
#include <torch/extension.h>

torch::Tensor argmin_cuda(torch::Tensor input, int dim);
"""

argmin_cuda = load_inline(
    name="argmin_cuda",
    cpp_sources=argmin_cpp_source,
    cuda_sources=argmin_cuda_source,
    functions=["argmin_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return argmin_cuda(x, self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095
dim = 1

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [dim]