import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hinge_loss_kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void hinge_loss_kernel(
    const T* predictions,
    const T* targets,
    T* sum,
    int N, // number of rows
    int M // number of columns
) {
    extern __shared__ T shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int stride = gridDim.x * blockDim.x;
    int start = bid * blockDim.x + tid;
    int total = N * M;

    T block_sum = static_cast<T>(0.0);
    for (int idx = start; idx < total; idx += stride) {
        int row = idx / M;
        int col = idx % M;
        T p = predictions[row * M + col];
        T t = targets[row];
        T loss = static_cast<T>(1.0) - p * t;
        if (loss > static_cast<T>(0.0)) {
            block_sum += loss;
        }
    }

    // Write to shared memory
    shared[tid] = block_sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum, shared[0]);
    }
}

at::Tensor compute_hinge_loss_cuda(
    at::Tensor predictions,
    at::Tensor targets) {
    const int N = predictions.size(0);
    const int M = predictions.size(1);
    const int total = N * M;

    // Allocate sum on device
    auto options = predictions.options();
    auto sum = at::empty({1}, options).zero_();

    // Launch configuration
    const int block_size = 256;
    const int num_blocks = 1024; // Tune this for best performance

    dim3 threads(block_size);
    dim3 blocks(num_blocks);
    size_t shared_size = block_size * sizeof(float);

    hinge_loss_kernel<float><<<blocks, threads, shared_size, at::cuda::getCurrentCUDAStream()>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        sum.data_ptr<float>(),
        N, M
    );

    AT_CUDA_CHECK(cudaGetLastError());

    // Compute mean
    auto mean = sum.item<float>() / static_cast<float>(total);

    return at::tensor(mean, options);
}
"""

hinge_loss_cpp_source = """
at::Tensor compute_hinge_loss_cuda(
    at::Tensor predictions,
    at::Tensor targets);
"""

# Compile the CUDA code
hinge_loss_cuda = load_inline(
    name="hinge_loss_cuda",
    cpp_sources=[hinge_loss_cpp_source],
    cuda_sources=[hinge_loss_kernel],
    functions=["compute_hinge_loss_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.hinge_loss_cuda = hinge_loss_cuda

    def forward(self, predictions, targets):
        return self.hinge_loss_cuda.compute_hinge_loss_cuda(predictions, targets)