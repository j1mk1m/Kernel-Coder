import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for argmax
argmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

#define THREADS_PER_BLOCK 256

__global__ void argmax_kernel(const float* input_data, int64_t* output_data,
                             int B, int D, int H,
                             int dim,
                             int num_output_elements) {
    const int tid = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int threads_per_block = THREADS_PER_BLOCK;

    __shared__ float s_max_val[THREADS_PER_BLOCK];
    __shared__ int64_t s_max_idx[THREADS_PER_BLOCK];

    // Determine the output coordinates (o1, o2)
    int o1, o2;

    // Compute the output shape dimensions.
    int output_B, output_D, output_H;
    int reduction_dim_size;

    if (dim == 0) {
        output_B = D;
        output_D = H;
        reduction_dim_size = B;
    } else if (dim == 1) {
        output_B = B;
        output_D = H;
        reduction_dim_size = D;
    } else { // dim ==2
        output_B = B;
        output_D = D;
        reduction_dim_size = H;
    }

    int output_size1 = output_B;
    int output_size2 = output_D;
    o1 = block_idx / output_size2;
    o2 = block_idx % output_size2;

    // non-reduction coordinates.
    int non_red_coord1, non_red_coord2;

    if (dim == 0) {
        non_red_coord1 = o1;
        non_red_coord2 = o2;
    } else if (dim == 1) {
        non_red_coord1 = o1;
        non_red_coord2 = o2;
    } else {
        non_red_coord1 = o1;
        non_red_coord2 = o2;
    }

    // Compute base.
    int base = 0;
    if (dim == 0) {
        base = non_red_coord1 * H + non_red_coord2;
    } else if (dim == 1) {
        base = non_red_coord1 * D * H + non_red_coord2;
    } else {
        base = (non_red_coord1 * D + non_red_coord2) * H;
    }

    // Compute chunk.
    int chunk_size = (reduction_dim_size + threads_per_block - 1) / threads_per_block;
    int start = tid * chunk_size;
    int end = start + chunk_size;
    if (end > reduction_dim_size) end = reduction_dim_size;

    float local_max_val = -std::numeric_limits<float>::infinity();
    int64_t local_max_r = -1;

    for (int r = start; r < end; ++r) {
        int linear_idx;
        if (dim == 0) {
            linear_idx = r * D * H + base;
        } else if (dim ==1) {
            linear_idx = base + r * H;
        } else {
            linear_idx = base + r;
        }

        float val = input_data[linear_idx];
        if (val > local_max_val) {
            local_max_val = val;
            local_max_r = r;
        } else if (val == local_max_val) {
            if (r < local_max_r) {
                local_max_r = r;
            }
        }
    }

    s_max_val[tid] = local_max_val;
    s_max_idx[tid] = local_max_r;

    __syncthreads();

    for (int s = threads_per_block / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float val1 = s_max_val[tid];
            int64_t idx1 = s_max_idx[tid];
            float val2 = s_max_val[tid + s];
            int64_t idx2 = s_max_idx[tid + s];

            if (val2 > val1) {
                s_max_val[tid] = val2;
                s_max_idx[tid] = idx2;
            } else if (val2 == val1) {
                if (idx2 < idx1) {
                    s_max_val[tid] = val2;
                    s_max_idx[tid] = idx2;
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output_data[block_idx] = s_max_idx[0];
    }
}

extern "C" {
    void argmax_cuda(torch::Tensor input, torch::Tensor output,
                     int B, int D, int H, int dim, int num_output_elements) {
        const int threads_per_block = THREADS_PER_BLOCK;
        const int blocks_per_grid = num_output_elements;
        argmax_kernel<<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<float>(),
            output.data_ptr<int64_t>(),
            B, D, H,
            dim,
            num_output_elements
        );
        cudaDeviceSynchronize();
    }
}
"""

argmax_cuda_header = (
    "void argmax_cuda(torch::Tensor input, torch::Tensor output, "
    "int B, int D, int H, int dim, int num_output_elements);"
)

argmax_cuda = load_inline(
    name="argmax_cuda",
    cpp_sources=argmax_cuda_header,
    cuda_sources=argmax_cuda_source,
    functions=["argmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H = x.shape
        output_shape = list(x.shape)
        del output_shape[self.dim]
        output = torch.empty(output_shape, dtype=torch.int64, device=x.device)
        num_output_elements = output.numel()
        argmax_cuda(
            x.contiguous(),
            output.contiguous(),
            B, D, H, self.dim, num_output_elements
        )
        return output

def get_inputs():
    x = torch.rand(128, 4096, 4095).cuda()
    return [x]

def get_init_inputs():
    return [1]