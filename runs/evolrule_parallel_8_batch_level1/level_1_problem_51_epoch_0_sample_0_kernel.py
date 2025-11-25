import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code
argmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void argmax_kernel(
    const scalar_t* __restrict__ input,
    int64_t* __restrict__ output,
    int dim, 
    int B, int D1, int D2) {

    // Determine output element indices based on blockIdx and dim
    int out_b, out_d1, out_d2;
    int output_size0, output_size1;
    if (dim == 0) {
        output_size0 = D1;
        output_size1 = D2;
        out_d1 = blockIdx.x / output_size1;
        out_d2 = blockIdx.x % output_size1;
    } else if (dim == 1) {
        output_size0 = B;
        output_size1 = D2;
        out_b = blockIdx.x / output_size1;
        out_d2 = blockIdx.x % output_size1;
    } else { // dim == 2
        output_size0 = B;
        output_size1 = D1;
        out_b = blockIdx.x / output_size1;
        out_d1 = blockIdx.x % output_size1;
    }

    // Determine the starting index along the dimension
    int dim_size = 0;
    switch (dim) {
        case 0:
            dim_size = B;
            break;
        case 1:
            dim_size = D1;
            break;
        case 2:
            dim_size = D2;
            break;
    }

    // Each thread processes a chunk of the dimension
    int tid = threadIdx.x;
    int threads_per_block = blockDim.x;

    // Calculate the start and end indices for this thread's chunk
    int chunk_size = (dim_size + threads_per_block - 1) / threads_per_block;
    int start = tid * chunk_size;
    int end = start + chunk_size;
    if (end > dim_size) end = dim_size;

    // Variables to track max value and index in this thread's chunk
    float local_max = -std::numeric_limits<float>::infinity();
    int local_idx = -1;

    for (int i = start; i < end; ++i) {
        // Compute input index based on current i along dim
        int in_index;
        if (dim == 0) {
            // input is (B, D1, D2)
            in_index = i * D1 * D2 + out_d1 * D2 + out_d2;
        } else if (dim == 1) {
            // iterating over D1 (dim1)
            in_index = out_b * D1 * D2 + i * D2 + out_d2;
        } else { // dim2
            // iterating over D2 (dim2)
            in_index = out_b * D1 * D2 + out_d1 * D2 + i;
        }

        float val = input[in_index];
        if (val > local_max || (val == local_max && i < local_idx)) {
            local_max = val;
            local_idx = i;
        }
    }

    // Now perform reduction in shared memory
    extern __shared__ char sdata[];
    float* s_max = (float*) sdata;
    int* s_indices = (int*)(sdata + blockDim.x * sizeof(float));

    s_max[tid] = local_max;
    s_indices[tid] = local_idx;
    __syncthreads();

    // Reduction steps
    for (int s = threads_per_block / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a_max = s_max[tid];
            int a_idx = s_indices[tid];
            float b_max = s_max[tid + s];
            int b_idx = s_indices[tid + s];

            if (b_max > a_max || (b_max == a_max && b_idx < a_idx)) {
                s_max[tid] = b_max;
                s_indices[tid] = b_idx;
            } else {
                s_max[tid] = a_max;
                s_indices[tid] = a_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int out_index;
        if (dim == 0) {
            out_index = out_d1 * output_size1 + out_d2;
        } else if (dim == 1) {
            out_index = out_b * output_size1 + out_d2;
        } else {
            out_index = out_b * output_size1 + out_d1;
        }
        output[out_index] = s_indices[0];
    }
}

// Wrapper function
torch::Tensor argmax_cuda(torch::Tensor input, int dim) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    int output_size0, output_size1;
    if (dim == 0) {
        output_size0 = D1;
        output_size1 = D2;
    } else if (dim == 1) {
        output_size0 = B;
        output_size1 = D2;
    } else {
        output_size0 = B;
        output_size1 = D1;
    }

    auto output = torch::empty({output_size0, output_size1}, 
                              torch::device("cuda").dtype(torch::kLong));

    int num_blocks = output_size0 * output_size1;
    int threads_per_block = 256;
    int shared_size = threads_per_block * (sizeof(float) + sizeof(int));

    argmax_kernel<float><<<num_blocks, threads_per_block, shared_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        dim, B, D1, D2);

    cudaDeviceSynchronize();
    return output;
}
"""

argmax_cpp_source = """
torch::Tensor argmax_cuda(torch::Tensor input, int dim);
"""

argmax = load_inline(
    name="argmax_cuda",
    cpp_sources=argmax_cpp_source,
    cuda_sources=argmax_source,
    functions=["argmax_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.argmax_cuda = argmax  # Bind the CUDA function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmax_cuda.argmax_cuda(x, self.dim)

# Update input functions to return CUDA tensors
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [torch.tensor([1], device='cuda')]