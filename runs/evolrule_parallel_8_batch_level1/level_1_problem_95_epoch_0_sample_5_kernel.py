import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cross_entropy_loss_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void compute_terms_kernel(
    const float* predictions,
    const int* targets,
    float* terms,
    int N,
    int C) {
    int sample_idx = blockIdx.x;
    if (sample_idx >= N) return;

    int t = targets[sample_idx];
    float x_t = 0.0f;
    float sum_exp = 0.0f;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    for (int j = tid; j < C; j += block_size) {
        float x_j = predictions[sample_idx * C + j];
        sum_exp += expf(x_j);
        if (j == t) {
            x_t = x_j;
        }
    }

    extern __shared__ float shared_mem[];
    float* sdata = shared_mem;
    sdata[tid] = sum_exp;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float total_sum_exp = sdata[0];
    float log_sum_exp = logf(total_sum_exp);
    float term = -x_t + log_sum_exp;

    if (tid == 0) {
        terms[sample_idx] = term;
    }
}

__global__ void sum_terms_kernel(
    const float* terms,
    float* loss,
    int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    if (i < N) {
        sdata[tid] = terms[i];
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
        atomicAdd(loss, sdata[0]);
    }
}

torch::Tensor cross_entropy_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    const int N = predictions.size(0);
    const int C = predictions.size(1);

    targets = targets.to(predictions.device());

    auto options = predictions.options();
    auto terms = torch::empty({N}, options);

    const int block_size_terms = 256;
    dim3 blocks_terms(N);
    dim3 threads_terms(block_size_terms);
    size_t shared_size_terms = block_size_terms * sizeof(float);

    compute_terms_kernel<<<blocks_terms, threads_terms, shared_size_terms>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int>(),
        terms.data_ptr<float>(),
        N,
        C
    );

    const int block_size_sum = 1024;
    const int grid_size_sum = (N + block_size_sum - 1) / block_size_sum;
    auto loss = torch::zeros(1, options);

    sum_terms_kernel<<<grid_size_sum, block_size_sum, block_size_sum * sizeof(float)>>>(
        terms.data_ptr<float>(),
        loss.data_ptr<float>(),
        N
    );

    loss /= static_cast<float>(N);

    return loss;
}
"""

cross_entropy_loss_cuda_header = """
torch::Tensor cross_entropy_loss_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

cross_entropy_loss = load_inline(
    name="cross_entropy_loss",
    cpp_sources=cross_entropy_loss_cuda_header,
    cuda_sources=cross_entropy_loss_cuda_source,
    functions=["cross_entropy_loss_cuda"],
    verbose=True,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss_cuda = cross_entropy_loss

    def forward(self, predictions, targets):
        return self.cross_entropy_loss_cuda.cross_entropy_loss_cuda(predictions, targets)

batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape), torch.randint(0, num_classes, (batch_size,))]

def get_init_inputs():
    return []