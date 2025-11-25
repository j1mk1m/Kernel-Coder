import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_reduction_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void max_reduction_kernel(const float* input,
                                    float* output,
                                    int dim,
                                    int B, int D1, int D2,
                                    int output_size) {

    int out_idx = blockIdx.x;
    if (out_idx >= output_size) return;

    int b, d1, d2;
    switch (dim) {
        case 0: { // reducing over B
            d1 = out_idx / D2;
            d2 = out_idx % D2;
            break;
        }
        case 1: { // reducing over D1
            b = out_idx / D2;
            d2 = out_idx % D2;
            break;
        }
        case 2: { // reducing over D2
            b = out_idx / D1;
            d1 = out_idx % D1;
            break;
        }
        default:
            assert(0 && "Invalid dim");
    }

    int s, stride;
    int base;
    switch (dim) {
        case 0: {
            s = B;
            stride = D1 * D2;
            base = d1 * D2 + d2;
            break;
        }
        case 1: {
            s = D1;
            stride = D2;
            base = b * D1 * D2 + d2;
            break;
        }
        case 2: {
            s = D2;
            stride = 1;
            base = b * D1 * D2 + d1 * D2;
            break;
        }
    }

    float local_max = -FLT_MAX;

    for (int r = threadIdx.x; r < s; r += blockDim.x) {
        int input_idx = r * stride + base;
        float val = input[input_idx];
        if (val > local_max) {
            local_max = val;
        }
    }

    __shared__ float sdata[256];
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            if (sdata[threadIdx.x + i] > sdata[threadIdx.x]) {
                sdata[threadIdx.x] = sdata[threadIdx.x + i];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[out_idx] = sdata[0];
    }
}

void max_reduction_cuda(
    torch::Tensor input,
    torch::Tensor output,
    int dim,
    int B, int D1, int D2,
    int output_size
) {
    int block_size = 256;
    int grid_size = output_size;

    dim3 blocks(grid_size);
    dim3 threads(block_size);

    max_reduction_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dim,
        B, D1, D2,
        output_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA error");
    }
}
"""

max_reduction_cpp_source = """
extern "C" {
    void max_reduction_cuda(
        torch::Tensor input,
        torch::Tensor output,
        int dim,
        int B, int D1, int D2,
        int output_size
    );
}
"""

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.max_reduction = load_inline(
            name="max_reduction",
            cpp_sources=max_reduction_cpp_source,
            cuda_sources=max_reduction_source,
            functions=["max_reduction_cuda"],
            verbose=True,
            extra_cflags=["-O3"],
            extra_ldflags=[""]
        )

    def forward(self, x):
        B, D1, D2 = x.size()
        output_size = (B * D1 * D2) // x.size(self.dim)
        output = torch.empty(output_size, device=x.device, dtype=x.dtype)
        self.max_reduction.max_reduction_cuda(
            x.contiguous(),
            output,
            self.dim,
            B, D1, D2,
            output_size
        )
        return output

def get_inputs():
    x = torch.rand(128, 4096, 4095).cuda()
    return [x]

def get_init_inputs():
    return [1]