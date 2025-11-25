import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hinge_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hinge_loss_kernel(
    const float* predictions,
    const float* targets,
    int N,
    int D,
    float* partial_sums
) {
    extern __shared__ float shared_sums[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (tid < blockDim.x) {
        shared_sums[tid] = 0.0f;
    }
    __syncthreads();

    int global_idx = bid * blockDim.x + tid;

    float contribution = 0.0f;
    if (global_idx < N * D) {
        int row = global_idx / D;
        float target = targets[row];
        float pred = predictions[global_idx];

        float term = 1.0f - pred * target;
        contribution = (term > 0.0f) ? term : 0.0f;
    }

    shared_sums[tid] = contribution;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] += shared_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[bid] = shared_sums[0];
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int N = predictions.size(0);
    int D = predictions.size(1);
    int elements_total = N * D;

    const int threadsPerBlock = 256;
    int blocksPerGrid = (elements_total + threadsPerBlock - 1) / threadsPerBlock;

    auto partial_sums = torch::empty(blocksPerGrid, torch::kFloat32).to(predictions.device());

    auto stream = at::cuda::getCurrentCUDAStream();

    hinge_loss_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float), stream>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        N,
        D,
        partial_sums.data_ptr<float>()
    );

    cudaStreamSynchronize(stream);

    auto partial_sums_cpu = partial_sums.cpu();
    float total_sum = partial_sums_cpu.sum().item<float>();

    float mean_loss = total_sum / elements_total;

    return torch::tensor({mean_loss}, predictions.options());
}
"""

hinge_loss_cpp_source = """
torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

hinge_loss = load_inline(
    name="hinge_loss",
    cpp_sources=hinge_loss_cpp_source,
    cuda_sources=hinge_loss_source,
    functions=["hinge_loss_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.hinge_loss_cuda_mod = hinge_loss

    def forward(self, predictions, targets):
        return self.hinge_loss_cuda_mod.hinge_loss_cuda(predictions, targets)

def get_inputs():
    batch_size = 32768
    input_shape = (32768,)
    return [torch.rand(batch_size, *input_shape), torch.randint(0, 2, (batch_size,)).float() * 2 - 1]

def get_init_inputs():
    return []