import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmin_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

template <typename T>
__global__ void argmin_kernel(const T* x_data, int* out_data, int batch_size, int dim1, int dim2, int dim) {
    int blockId = blockIdx.x;

    int b, d1, d2;
    int dim_size;

    if (dim == 0) {
        d1 = blockId / dim2;
        d2 = blockId % dim2;
        dim_size = batch_size;
    } else if (dim == 1) {
        b = blockId / dim2;
        d2 = blockId % dim2;
        dim_size = dim1;
    } else if (dim == 2) {
        b = blockId / dim1;
        d1 = blockId % dim1;
        dim_size = dim2;
    } else {
        dim_size = dim1; // default to case 1
    }

    float min_val = FLT_MAX;
    int min_idx = -1;

    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        int linear = 0;
        switch (dim) {
            case 0:
                linear = i * (dim1 * dim2) + (d1 * dim2 + d2);
                break;
            case 1:
                linear = b * (dim1 * dim2) + (i * dim2 + d2);
                break;
            case 2:
                linear = b * (dim1 * dim2) + (d1 * dim2 + i);
                break;
        }
        float val = static_cast<float>(x_data[linear]);
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        } else if (val == min_val) {
            if (i < min_idx) {
                min_idx = i;
            }
        }
    }

    __shared__ float shared_min[256];
    __shared__ int shared_idx[256];

    shared_min[threadIdx.x] = min_val;
    shared_idx[threadIdx.x] = min_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float val1 = shared_min[threadIdx.x];
            int idx1 = shared_idx[threadIdx.x];
            float val2 = shared_min[threadIdx.x + s];
            int idx2 = shared_idx[threadIdx.x + s];

            if (val2 < val1) {
                val1 = val2;
                idx1 = idx2;
            } else if (val2 == val1) {
                if (idx2 < idx1) {
                    idx1 = idx2;
                }
            }

            shared_min[threadIdx.x] = val1;
            shared_idx[threadIdx.x] = idx1;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_data[blockId] = shared_idx[0];
    }
}

torch::Tensor argmin_cuda(torch::Tensor x, int dim) {
    TORCH_CHECK(x.dim() == 3, "Input must be a 3D tensor");
    TORCH_CHECK(dim >= 0 && dim < 3, "Dimension out of range for 3D tensor");

    int batch_size = x.size(0);
    int dim1 = x.size(1);
    int dim2 = x.size(2);

    int output_size;
    if (dim == 0) {
        output_size = dim1 * dim2;
    } else if (dim == 1) {
        output_size = batch_size * dim2;
    } else {
        output_size = batch_size * dim1;
    }

    auto out = torch::empty({output_size}, torch::dtype(torch::kInt32).device(x.device()));

    const int block_size = 256;
    const int grid_size = output_size;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "argmin_cuda", ([&]{
        argmin_kernel<scalar_t><<<grid_size, block_size>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<int>(),
            batch_size,
            dim1,
            dim2,
            dim
        );
    }));

    // Reshape to correct dimensions
    if (dim == 0) {
        return out.view({dim1, dim2});
    } else if (dim == 1) {
        return out.view({batch_size, dim2});
    } else {
        return out.view({batch_size, dim1});
    }
}
"""

argmin_cpp_source = """
torch::Tensor argmin_cuda(torch::Tensor x, int dim);
"""

argmin = load_inline(
    name="argmin",
    cpp_sources=argmin_cpp_source,
    cuda_sources=argmin_source,
    functions=["argmin_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return argmin.argmin_cuda(x, self.dim)

def get_inputs():
    x = torch.rand(128, 4096, 4095)
    return [x]

def get_init_inputs():
    return [1]