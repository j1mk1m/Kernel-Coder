import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hinge_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hinge_loss_kernel(
    const float* predictions,
    const float* targets,
    float* block_sums,
    int N,
    int D
) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int block_size = blockDim.x;

    float local_sum = 0.0f;
    for (int idx = block_id * block_size + tid; idx < N * D; idx += gridDim.x * block_size) {
        int i = idx / D;
        int j = idx % D;
        float pred = predictions[idx];
        float target = targets[i];
        float term = 1.0f - pred * target;
        float contrib = (term > 0.0f) ? term : 0.0f;
        local_sum += contrib;
    }

    shared_mem[tid] = local_sum;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[block_id] = shared_mem[0];
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int N = predictions.size(0);
    int D = predictions.size(1);
    int num_elements = N * D;

    predictions = predictions.contiguous();
    targets = targets.contiguous();

    int block_dim = 256;
    int grid_dim = (num_elements + block_dim - 1) / block_dim;

    auto block_sums = torch::empty(grid_dim, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    int shared_mem_size = block_dim * sizeof(float);

    hinge_loss_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        N,
        D
    );

    auto total_sum = block_sums.sum();

    float mean_val = total_sum.item<float>() / static_cast<float>(num_elements);

    return torch::tensor({mean_val}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
}
"""

hinge_loss_cpp = (
    "torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

hinge_loss = load_inline(
    name="hinge_loss",
    cpp_sources=hinge_loss_cpp,
    cuda_sources=hinge_loss_source,
    functions=["hinge_loss_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.hinge_loss = hinge_loss

    def forward(self, predictions, targets):
        return self.hinge_loss.hinge_loss_cuda(predictions, targets)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    predictions = torch.rand(batch_size, *input_shape, device='cuda')
    targets = torch.randint(0, 2, (batch_size,), device='cuda').float() * 2 - 1
    return [predictions, targets]

def get_init_inputs():
    return []