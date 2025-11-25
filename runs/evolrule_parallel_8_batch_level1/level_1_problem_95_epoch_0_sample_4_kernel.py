import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cross_entropy_cuda_src = """
#include <torch/extension.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void cross_entropy_kernel(
    const float* predictions,
    const int* targets,
    float* loss_sum,
    int batch_size,
    int num_classes) {

    __shared__ float shared_terms[256];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int sample_idx = bid * 256 + tid;

    if (sample_idx >= batch_size) return;

    // Compute log_sum_exp for this sample
    float max_val = -FLT_MAX;
    for (int j = 0; j < num_classes; ++j) {
        float val = predictions[sample_idx * num_classes + j];
        if (val > max_val) {
            max_val = val;
        }
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; ++j) {
        float val = predictions[sample_idx * num_classes + j];
        sum_exp += expf(val - max_val);
    }

    float log_sum_exp = max_val + logf(sum_exp);

    // Get target index
    int target = targets[sample_idx];
    float pred_target = predictions[sample_idx * num_classes + target];

    float term = log_sum_exp - pred_target;

    // Write to shared memory
    shared_terms[tid] = term;
    __syncthreads();

    // Reduction in shared memory
    for (int s = 256 / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_terms[tid] += shared_terms[tid + s];
        }
        __syncthreads();
    }

    // Atomic add the partial sum
    if (tid == 0) {
        atomicAdd(loss_sum, shared_terms[0]);
    }
}

torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto device = predictions.device();
    if (device.type() != torch::kCUDA) {
        throw std::runtime_error("Inputs must be on CUDA device");
    }

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    int block_size = 256;

    auto loss_sum = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
    int num_blocks = (batch_size + block_size - 1) / block_size;

    // Launch kernel
    cross_entropy_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int>(),
        loss_sum.data_ptr<float>(),
        batch_size,
        num_classes
    );

    // Compute mean
    return loss_sum[0] / batch_size;
}
"""

cross_entropy_cuda_header = """
torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

# Compile the CUDA code
cross_entropy_cuda = load_inline(
    name="cross_entropy_cuda",
    cpp_sources=cross_entropy_cuda_header,
    cuda_sources=cross_entropy_cuda_src,
    functions=["cross_entropy_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14", "--expt-extended-lambda"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_cuda = cross_entropy_cuda

    def forward(self, predictions, targets):
        return self.cross_entropy_cuda(predictions, targets)