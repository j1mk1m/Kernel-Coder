import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

smooth_l1_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void smooth_l1_loss_kernel(const float* predictions, const float* targets, float* output, int num_elements) {
    extern __shared__ float shared_buf[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    float loss = 0.0f;
    if (idx < num_elements) {
        float diff = predictions[idx] - targets[idx];
        float abs_diff = fabs(diff);
        loss = (abs_diff < 1.0f) ? 0.5f * diff * diff : (abs_diff - 0.5f);
    }

    shared_buf[tid] = loss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_buf[tid] += shared_buf[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared_buf[0]);
    }
}

torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    const int num_elements = predictions.numel();
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    torch::Tensor output = torch::zeros(1, predictions.options());
    float* output_ptr = output.data_ptr<float>();

    smooth_l1_loss_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output_ptr,
        num_elements
    );

    cudaDeviceSynchronize();
    output.div_(static_cast<float>(num_elements));
    return output;
}
"""

smooth_l1_loss_cpp_source = """
torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

smooth_l1_loss = load_inline(
    name="smooth_l1_loss",
    cpp_sources=smooth_l1_loss_cpp_source,
    cuda_sources=smooth_l1_loss_source,
    functions=["smooth_l1_loss_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth_l1_loss_mod = smooth_l1_loss

    def forward(self, predictions, targets):
        return self.smooth_l1_loss_mod.smooth_l1_loss_cuda(predictions, targets)