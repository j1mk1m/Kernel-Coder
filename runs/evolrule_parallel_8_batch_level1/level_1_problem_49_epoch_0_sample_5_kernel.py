import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_reduction_kernel(
    const float* input,
    float* output,
    int dim,
    int reduction_size,
    int B,
    int D1,
    int D2
) {
    int output_idx = blockIdx.x;
    float max_val = -INFINITY;

    // Determine indices based on dimension
    int idx0, idx1, idx2;
    if (dim == 0) {
        int d1 = output_idx / D2;
        int d2 = output_idx % D2;
        idx0 = 0; // unused
        idx1 = d1;
        idx2 = d2;
    } else if (dim == 1) {
        int b = output_idx / D2;
        int d2 = output_idx % D2;
        idx0 = b;
        idx1 = 0;
        idx2 = d2;
    } else { // dim == 2
        int b = output_idx / D1;
        int d1 = output_idx % D1;
        idx0 = b;
        idx1 = d1;
        idx2 = 0;
    }

    // Iterate over reduction dimension
    for (int i = threadIdx.x; i < reduction_size; i += blockDim.x) {
        int reduction_idx = i;
        int input_offset;

        if (dim == 0) {
            input_offset = reduction_idx * D1 * D2 + idx1 * D2 + idx2;
        } else if (dim == 1) {
            input_offset = idx0 * D1 * D2 + reduction_idx * D2 + idx2;
        } else {
            input_offset = idx0 * D1 * D2 + idx1 * D2 + reduction_idx;
        }

        float val = input[input_offset];
        if (val > max_val) {
            max_val = val;
        }
    }

    // Block reduction using shared memory
    extern __shared__ float shared_max[];
    int tid = threadIdx.x;
    shared_max[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid] < shared_max[tid + s]) {
                shared_max[tid] = shared_max[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[output_idx] = shared_max[0];
    }
}

torch::Tensor max_reduction_cuda(torch::Tensor input, int dim) {
    const int block_size = 256;

    // Ensure input is contiguous
    input = input.contiguous();

    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);
    int reduction_size = input.size(dim);

    // Compute output shape
    std::vector<int64_t> output_shape;
    if (dim == 0) {
        output_shape = {D1, D2};
    } else if (dim == 1) {
        output_shape = {B, D2};
    } else {
        output_shape = {B, D1};
    }

    // Output tensor
    auto output = torch::empty(output_shape, input.options());

    dim3 block(block_size);
    int output_size = output.numel();
    dim3 grid(output_size);

    int shared_mem_size = block_size * sizeof(float);

    max_reduction_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dim,
        reduction_size,
        B,
        D1,
        D2
    );

    cudaDeviceSynchronize();
    return output;
}
"""

max_reduction_module = load_inline(
    name="max_reduction",
    cuda_sources=max_reduction_source,
    functions=["max_reduction_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.max_reduction = max_reduction_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_reduction.max_reduction_cuda(x, self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]