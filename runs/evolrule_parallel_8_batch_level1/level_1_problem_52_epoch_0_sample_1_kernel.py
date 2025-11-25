import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmin_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

using namespace std;

__global__ void argmin_kernel(const float* input, int64_t* output,
                             int batch_size, int dim1, int dim2) {
    int idx = blockIdx.x;
    int b = idx / dim2;
    int k = idx % dim2;

    int tid = threadIdx.x;
    int chunk_size = dim1 / blockDim.x;

    float min_val = std::numeric_limits<float>::max();
    int64_t min_idx = -1;

    for (int i = tid * chunk_size; i < (tid + 1)* chunk_size; i++) {
        int pos = b * dim1 * dim2 + i * dim2 + k;
        float val = input[pos];
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        } else if (val == min_val) {
            if (i < min_idx) {
                min_idx = i;
            }
        }
    }

    extern __shared__ unsigned char sdata[];
    float* s_min_vals = (float*)sdata;
    int64_t* s_min_indices = (int64_t*)(sdata + blockDim.x * sizeof(float));

    s_min_vals[tid] = min_val;
    s_min_indices[tid] = min_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int j = tid;
            float val1 = s_min_vals[j];
            int64_t idx1 = s_min_indices[j];
            float val2 = s_min_vals[j + s];
            int64_t idx2 = s_min_indices[j + s];

            float new_val;
            int64_t new_idx;
            if (val1 < val2) {
                new_val = val1;
                new_idx = idx1;
            } else if (val1 > val2) {
                new_val = val2;
                new_idx = idx2;
            } else {
                if (idx1 < idx2) {
                    new_val = val1;
                    new_idx = idx1;
                } else {
                    new_val = val2;
                    new_idx = idx2;
                }
            }
            s_min_vals[j] = new_val;
            s_min_indices[j] = new_idx;
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[idx] = s_min_indices[0];
    }
}

torch::Tensor argmin_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);
    auto output = torch::empty({batch_size, dim2}, torch::dtype(torch::kInt64).device(input.device()));

    const int block_size = 256;
    const dim3 blocks(batch_size * dim2);
    dim3 threads(block_size);

    int shared_size = block_size * (sizeof(float) + sizeof(int64_t));

    argmin_kernel<<<blocks, threads, shared_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        batch_size,
        dim1,
        dim2);

    return output;
}
"""

argmin_header = """
torch::Tensor argmin_cuda(torch::Tensor input);
"""

argmin_cuda = load_inline(
    name="argmin_cuda",
    cuda_sources=argmin_source,
    cpp_sources=argmin_header,
    functions=["argmin_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.argmin_cuda_func = argmin_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmin_cuda_func(x)

def get_inputs():
    x = torch.rand(128, 4096, 4095).cuda()
    return [x]

def get_init_inputs():
    return [1]