import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel and the wrapper function
mean_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" {

__global__ void mean_kernel(const float* x, float* out, int dim, int B, int D1, int D2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int output_size;
    if (dim == 0) {
        output_size = D1 * D2;
    } else if (dim == 1) {
        output_size = B * D2;
    } else {
        output_size = B * D1;
    }

    if (idx >= output_size) return;

    int o0, o1;
    if (dim == 0) {
        int d1 = idx / D2;
        int d2 = idx % D2;
        o0 = d1;
        o1 = d2;
    } else if (dim == 1) {
        int b = idx / D2;
        int d2 = idx % D2;
        o0 = b;
        o1 = d2;
    } else {
        int b = idx / D1;
        int d1 = idx % D1;
        o0 = b;
        o1 = d1;
    }

    float sum = 0.0f;
    if (dim == 0) {
        for (int b = 0; b < B; ++b) {
            sum += x[b * D1 * D2 + o0 * D2 + o1];
        }
    } else if (dim == 1) {
        for (int d1 = 0; d1 < D1; ++d1) {
            sum += x[o0 * D1 * D2 + d1 * D2 + o1];
        }
    } else {
        for (int d2 = 0; d2 < D2; ++d2) {
            sum += x[o0 * D1 * D2 + o1 * D2 + d2];
        }
    }

    float divisor = (dim == 0) ? B : (dim == 1 ? D1 : D2);
    out[idx] = sum / divisor;
}

torch::Tensor mean_cuda(torch::Tensor x, int dim) {
    int B = x.size(0);
    int D1 = x.size(1);
    int D2 = x.size(2);

    int output_size;
    if (dim == 0) {
        output_size = D1 * D2;
    } else if (dim == 1) {
        output_size = B * D2;
    } else {
        output_size = B * D1;
    }

    auto out = torch::empty({output_size}, x.options());
    int block_size = 256;
    int num_blocks = (output_size + block_size - 1) / block_size;

    mean_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        dim,
        B,
        D1,
        D2
    );

    if (dim == 0) {
        out = out.view({D1, D2});
    } else if (dim == 1) {
        out = out.view({B, D2});
    } else {
        out = out.view({B, D1});
    }

    return out;
}

}
"""

mean_cuda_header = "torch::Tensor mean_cuda(torch::Tensor x, int dim);"

# Load the CUDA extension
mean_cuda = load_inline(
    name="mean_cuda",
    cpp_sources=mean_cuda_header,
    cuda_sources=mean_cuda_source,
    functions=["mean_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mean_cuda = mean_cuda  # The loaded CUDA function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_cuda(x.contiguous(), self.dim)

# Global variables as in the original code
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]