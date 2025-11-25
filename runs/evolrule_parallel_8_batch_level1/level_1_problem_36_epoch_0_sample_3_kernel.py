import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

rmsnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void rmsnorm_kernel(float* x, float* out, int batch_size, int features, int dim1, int dim2, float eps) {
    extern __shared__ float shared_squares[];
    int batch = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;
    int c = threadIdx.x;
    int x_offset = batch * features * dim1 * dim2 + c * dim1 * dim2 + h * dim2 + w;
    float x_val = x[x_offset];
    float square = x_val * x_val;
    shared_squares[threadIdx.x] = square;
    __syncthreads();

    for (int s = features/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_squares[threadIdx.x] += shared_squares[threadIdx.x + s];
        }
        __syncthreads();
    }

    float mean = shared_squares[0] / features;
    float rms = sqrtf(mean + eps);
    __syncthreads();

    out[x_offset] = x_val / rms;
}

torch::Tensor rmsnorm_cuda(torch::Tensor x, float eps) {
    auto out = torch::empty_like(x);
    int batch_size = x.size(0);
    int features = x.size(1);
    int dim1 = x.size(2);
    int dim2 = x.size(3);
    dim3 blocks(batch_size, dim1, dim2);
    dim3 threads(features, 1, 1);
    int shared_size = features * sizeof(float);

    AT_CUDA_CHECK(cudaFuncSetAttribute(
        rmsnorm_kernel,
        cudaFuncAttributeMaxThreadsPerBlock,
        features
    ));

    rmsnorm_kernel<<<blocks, threads, shared_size, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        features,
        dim1,
        dim2,
        eps
    );

    return out;
}
"""

cpp_sources = "torch::Tensor rmsnorm_cuda(torch::Tensor x, float eps);"

rmsnorm_cuda = load_inline(
    name="rmsnorm_cuda",
    cpp_sources=cpp_sources,
    cuda_sources=rmsnorm_source,
    functions=["rmsnorm_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_cuda.rmsnorm_cuda(x, self.eps)

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features]