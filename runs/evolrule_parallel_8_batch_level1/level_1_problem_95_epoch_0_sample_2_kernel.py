import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cross_entropy_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void compute_loss_kernel(
    const float* __restrict__ predictions,
    const int64_t* __restrict__ targets,
    float* losses,
    int batch_size,
    int num_classes) {

    int sample_idx = blockIdx.x;
    const float* sample = predictions + sample_idx * num_classes;
    int64_t target = targets[sample_idx];

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int num_elements_per_thread = (num_classes + block_size - 1) / block_size;

    // Compute local max
    float local_max = -FLT_MAX;
    for (int i = tid * num_elements_per_thread; i < (tid + 1)*num_elements_per_thread; i++) {
        if (i < num_classes) {
            float x = sample[i];
            if (x > local_max) {
                local_max = x;
            }
        }
    }

    // Reduction to find global_max
    __shared__ float sdata_max[256];
    sdata_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata_max[threadIdx.x] < sdata_max[threadIdx.x + s]) {
                sdata_max[threadIdx.x] = sdata_max[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    float global_max = sdata_max[0];

    // Broadcast global_max to all threads
    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = global_max;
    __syncthreads();
    global_max = shared_max;

    // Compute sum of exp(x_i - global_max)
    float local_sum = 0.0f;
    for (int i = tid * num_elements_per_thread; i < (tid + 1)*num_elements_per_thread; i++) {
        if (i < num_classes) {
            float x = sample[i] - global_max;
            local_sum += expf(x);
        }
    }

    // Reduction to find total_sum
    __shared__ float sdata_sum[256];
    sdata_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata_sum[threadIdx.x] += sdata_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    float total_sum = sdata_sum[0];

    // Compute logsumexp
    float logsumexp = logf(total_sum) + global_max;

    // Get target value
    float target_val = sample[target];

    // Compute loss for this sample
    float loss = logsumexp - target_val;

    if (threadIdx.x == 0) {
        losses[sample_idx] = loss;
    }
}

__global__ void sum_loss_kernel(
    const float* losses,
    float* total_loss,
    int size) {

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    if (i < size) {
        sdata[tid] = losses[i];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(total_loss, sdata[0]);
    }
}

torch::Tensor cross_entropy_cuda(
    torch::Tensor predictions,
    torch::Tensor targets) {

    const int batch_size = predictions.size(0);
    const int num_classes = predictions.size(1);

    auto losses = torch::empty({batch_size}, predictions.options());

    const int block_size = 256;
    dim3 blocks(batch_size);
    dim3 threads(block_size);

    compute_loss_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes);

    auto total_loss = torch::zeros(1, predictions.options());
    const int block_size_sum = 256;
    dim3 blocks_sum((batch_size + block_size_sum - 1) / block_size_sum);
    dim3 threads_sum(block_size_sum);

    sum_loss_kernel<<<blocks_sum, threads_sum, threads_sum.x * sizeof(float)>>>(
        losses.data_ptr<float>(),
        total_loss.data_ptr<float>(),
        batch_size);

    return total_loss / static_cast<float>(batch_size);
}
"""

cross_entropy_cpp_source = """
torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

cross_entropy = load_inline(
    name="cross_entropy",
    cpp_sources=cross_entropy_cpp_source,
    cuda_sources=cross_entropy_cuda_source,
    functions=["cross_entropy_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = cross_entropy

    def forward(self, predictions, targets):
        return self.cross_entropy.cross_entropy_cuda(predictions, targets)