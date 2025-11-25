import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

frobenius_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

template <typename T>
__global__ void frobenius_norm_kernel(const T* __restrict__ x_data, T* norm, int64_t total_elements) {
    using BlockReduce = cub::BlockReduce<T, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    T sum = 0.0;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
        T val = x_data[idx];
        sum += val * val;
    }

    sum = BlockReduce(temp_storage).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
        atomicAdd(norm, sum);
    }
}

template <typename T>
__global__ void normalize_kernel(const T* __restrict__ x_data, T* y_data, T norm, int64_t total_elements) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
        y_data[idx] = x_data[idx] / norm;
    }
}

torch::Tensor compute_frobenius_norm_cuda(torch::Tensor x) {
    auto total_elements = x.numel();
    auto norm = torch::zeros(1, x.dtype(), x.options().device(torch::kCUDA));

    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    frobenius_norm_kernel<<<grid_size, block_size>>>(x.data_ptr<float>(), norm.data_ptr<float>(), total_elements);

    cudaDeviceSynchronize();

    float* norm_ptr = norm.data_ptr<float>();
    *norm_ptr = sqrt(*norm_ptr);

    return norm;
}

torch::Tensor frobenius_normalize_cuda(torch::Tensor x) {
    auto total_elements = x.numel();
    auto norm = compute_frobenius_norm_cuda(x);
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    normalize_kernel<<<grid_size, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), norm.item<float>(), total_elements);

    return y;
}
"""

# Define the header for the C++ code
frobenius_norm_cpp_source = """
torch::Tensor compute_frobenius_norm_cuda(torch::Tensor x);
torch::Tensor frobenius_normalize_cuda(torch::Tensor x);
"""

# Compile the CUDA code
frobenius_normalize = load_inline(
    name="frobenius_normalize",
    cpp_sources=frobenius_norm_cpp_source,
    cuda_sources=frobenius_norm_source,
    functions=["frobenius_normalize_cuda"],
    verbose=True,
    extra_cflags=["-I/usr/local/cuda/include"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.frobenius_normalize = frobenius_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.cuda()  # Ensure input is on CUDA
        return self.frobenius_normalize.frobenius_normalize_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return []