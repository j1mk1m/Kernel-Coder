import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

kl_div_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename T>
__global__ void kl_div_forward_kernel(const T* __restrict__ predictions,
                                      const T* __restrict__ targets,
                                      T* __restrict__ output,
                                      int batch_size,
                                      int num_features) {

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    T sum = 0.0;
    for (int j = threadIdx.x; j < num_features; j += blockDim.x) {
        T p = predictions[batch_idx * num_features + j];
        T t = targets[batch_idx * num_features + j];
        T log_p = log(p);
        T log_t = log(t);
        T term = t * (log_t - log_p);
        sum += term;
    }

    extern __shared__ T shared[];
    int tid = threadIdx.x;
    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch_idx] = shared[0];
    }
}

template <typename T>
__global__ void reduce_sum_kernel(const T* input, T* output, int n) {
    extern __shared__ T shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    T sum = 0.0;
    for (int i = bid * blockDim.x + tid; i < n; i += stride) {
        sum += input[i];
    }
    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

at::Tensor kl_div_forward_cuda(at::Tensor predictions,
                               at::Tensor targets) {
    AT_ASSERT(predictions.is_cuda());
    AT_ASSERT(targets.is_cuda());
    AT_ASSERT(predictions.size(0) == targets.size(0));
    AT_ASSERT(predictions.size(1) == targets.size(1));

    int batch_size = predictions.size(0);
    int num_features = predictions.size(1);

    auto output_per_batch = at::empty({batch_size}, predictions.options());

    const int threads = 256;
    const int blocks = batch_size;
    const int shared_size = threads * sizeof(float); // Assuming float

    kl_div_forward_kernel<float><<<blocks, threads, shared_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output_per_batch.data_ptr<float>(),
        batch_size,
        num_features
    );

    auto total_sum = at::zeros({1}, predictions.options());
    const int reduce_threads = 256;
    const int reduce_blocks = 64; // Adjust as needed
    const int reduce_shared = reduce_threads * sizeof(float);

    reduce_sum_kernel<float><<<reduce_blocks, reduce_threads, reduce_shared>>>(
        output_per_batch.data_ptr<float>(),
        total_sum.data_ptr<float>(),
        batch_size
    );

    float mean = total_sum.item<float>() / batch_size;

    return at::tensor({mean}, predictions.options());
}
"""

kl_div_cuda_header = """
#include <torch/extension.h>
at::Tensor kl_div_forward_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

kl_div_cuda = load_inline(
    name="kl_div_cuda",
    cpp_sources=kl_div_cuda_header,
    cuda_sources=kl_div_cuda_source,
    functions=["kl_div_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_div_forward = kl_div_cuda

    def forward(self, predictions, targets):
        return self.kl_div_forward.kl_div_forward_cuda(predictions, targets)