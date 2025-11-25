import torch
import torch.nn as nn

from torch.utils.cpp_extension import load_inline

sum_abs_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void compute_sum_abs(const float* x, float* sum_abs, int B, int D) {
    int row = blockIdx.x;
    if (row >= B) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;

    for (int col = tid; col < D; col += blockDim.x) {
        float val = abs(x[row * D + col]);
        sdata[tid] += val;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum_abs[row] = sdata[0];
    }
}

__global__ void apply_inv_mean(const float* x, const float* sum_abs, float* out, int D) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float inv = D / sum_abs[row];
    for (int col = tid; col < D; col += blockDim.x) {
        int idx = row * D + col;
        out[idx] = x[idx] * inv;
    }
}

void launch_compute_sum_abs(torch::Tensor x, torch::Tensor sum_abs) {
    int B = x.size(0);
    int D = x.size(1);
    const int block_size = 256;
    int sharedMem = block_size * sizeof(float);
    dim3 blocks(B);
    dim3 threads(block_size);
    compute_sum_abs<<<blocks, threads, sharedMem>>>(x.data_ptr<float>(), sum_abs.data_ptr<float>(), B, D);
}

void launch_apply_inv_mean(torch::Tensor x, torch::Tensor sum_abs, torch::Tensor out) {
    int B = x.size(0);
    int D = x.size(1);
    const int block_size = 256;
    dim3 blocks(B);
    dim3 threads(block_size);
    apply_inv_mean<<<blocks, threads>>>(x.data_ptr<float>(), sum_abs.data_ptr<float>(), out.data_ptr<float>(), D);
}
"""

sum_abs_cpp_source = """
void launch_compute_sum_abs(torch::Tensor x, torch::Tensor sum_abs);
void launch_apply_inv_mean(torch::Tensor x, torch::Tensor sum_abs, torch::Tensor out);
"""

sum_abs_cuda = load_inline(
    name="sum_abs_cuda",
    cpp_sources=sum_abs_cpp_source,
    cuda_sources=sum_abs_source,
    functions=["launch_compute_sum_abs", "launch_apply_inv_mean"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.sum_abs_cuda = sum_abs_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        D = x.size(1)
        sum_abs = torch.empty(B, dtype=x.dtype, device=x.device)
        out = torch.empty_like(x)

        self.sum_abs_cuda.launch_compute_sum_abs(x, sum_abs)
        self.sum_abs_cuda.launch_apply_inv_mean(x, sum_abs, out)

        return out

batch_size = 32768
dim = 65535

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []