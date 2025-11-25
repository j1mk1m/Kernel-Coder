import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_kl_div_source = """
#include <torch/extension.h>
#include <math.h>

#define THREADS_PER_BLOCK 1024

__global__ void kl_div_loss_kernel(
    const float* targets,
    const float* predictions,
    float* output,
    int total_elements) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    float sum_val = 0.0f;
    int index = bid * blockDim.x + tid;

    if (index < total_elements) {
        float target = targets[index];
        float pred = predictions[index];
        float log_t = logf(target);
        float log_p = logf(pred);
        float term = target * (log_t - log_p);
        sum_val = term;
    }

    shared[tid] = sum_val;
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

torch::Tensor kl_div_loss_cuda(
    torch::Tensor targets,
    torch::Tensor predictions) {

    targets = targets.contiguous();
    predictions = predictions.contiguous();
    TORCH_CHECK(targets.device() == predictions.device(), "targets and predictions must be on the same device");
    TORCH_CHECK(targets.sizes() == predictions.sizes(), "targets and predictions must have the same shape");

    int batch_size = targets.size(0);
    int total_elements = targets.numel();
    auto output = torch::zeros(1, device=targets.device());

    const int threads_per_block = THREADS_PER_BLOCK;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    auto stream = at::cuda::getCurrentCUDAStream();
    kl_div_loss_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(float), stream>>>(
        targets.data_ptr<float>(),
        predictions.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements);

    output /= batch_size;

    return output;
}
"""

elementwise_kl_div_cpp_source = (
    "torch::Tensor kl_div_loss_cuda(torch::Tensor targets, torch::Tensor predictions);"
)

kl_div_loss = load_inline(
    name="kl_div_loss",
    cpp_sources=elementwise_kl_div_cpp_source,
    cuda_sources=elementwise_kl_div_source,
    functions=["kl_div_loss_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_div_loss = kl_div_loss

    def forward(self, predictions, targets):
        return self.kl_div_loss.kl_div_loss_cuda(targets, predictions)

batch_size = 8192 * 2
input_shape = (8192 * 2,)

def get_inputs():
    scale = torch.rand(())
    return [(torch.rand(batch_size, *input_shape) * scale).softmax(dim=-1),
            torch.rand(batch_size, *input_shape).softmax(dim=-1)]

def get_init_inputs():
    return []