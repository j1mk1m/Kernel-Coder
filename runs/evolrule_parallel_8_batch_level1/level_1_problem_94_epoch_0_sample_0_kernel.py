import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for fused MSE computation
mse_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template<typename scalar_t>
__global__ void mse_kernel(const scalar_t* predictions, const scalar_t* targets, scalar_t* output, int batch_size, int dim) {
    extern __shared__ scalar_t shared[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    scalar_t sum = 0.0;

    // Load data into shared memory
    if (gid < batch_size * dim) {
        scalar_t diff = predictions[gid] - targets[gid];
        sum += diff * diff;
    }

    // Write to shared memory
    shared[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // Write result to global memory (only block 0)
    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

torch::Tensor mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    const int batch_size = predictions.size(0);
    const int dim = predictions.size(1);
    const int elements = batch_size * dim;

    auto output = torch::zeros(1, predictions.options());

    const int block_size = 256;
    const int num_blocks = (elements + block_size - 1) / block_size;

    dim3 blocks(num_blocks);
    dim3 threads(block_size);
    size_t shared_size = threads.x * sizeof(float);

    mse_kernel<float><<<blocks, threads, shared_size, at::cuda::getCurrentCUDAStream()>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );

    // Compute mean by dividing by total elements
    output /= (batch_size * dim);

    return output;
}
"""

# Compile the CUDA kernel inline
mse_loss = load_inline(
    name="mse_loss",
    cpp_sources="",
    cuda_sources=mse_kernel_source,
    functions=["mse_loss_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.mse_loss = mse_loss

    def forward(self, predictions, targets):
        return self.mse_loss.mse_loss_cuda(predictions, targets)

def get_inputs():
    batch_size = 32768
    input_shape = (32768,)
    scale = torch.rand(())
    return [
        torch.rand(batch_size, *input_shape, device='cuda') * scale,
        torch.rand(batch_size, *input_shape, device='cuda')
    ]

def get_init_inputs():
    return []