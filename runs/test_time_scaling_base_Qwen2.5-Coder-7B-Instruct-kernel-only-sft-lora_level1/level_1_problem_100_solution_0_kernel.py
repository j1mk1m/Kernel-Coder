import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for hinge loss computation
hinge_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float value = 1 - predictions[idx] * targets[idx];
        out[idx] = value > 0 ? value : 0;
    }
}

__global__ void sum_kernel(const float* data, float* out, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        sdata[tid] = data[i];
    } else {
        sdata[tid] = 0;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto out = torch::zeros({1}, torch::kFloat32).cuda();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hinge_loss_kernel<<<num_blocks, block_size>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), out.data_ptr<float>(), size);

    float total_sum = 0;
    sum_kernel<<<1, block_size, block_size * sizeof(float)>>>(out.data_ptr<float>(), &total_sum, 1);

    return torch.tensor({total_sum / static_cast<float>(size)}, torch::kFloat32).cuda();
}
"""

hinge_loss_cpp_source = (
    "torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile the inline CUDA code for hinge loss computation
hinge_loss = load_inline(
    name="hinge_loss",
    cpp_sources=hinge_loss_cpp_source,
    cuda_sources=hinge_loss_source,
    functions=["hinge_loss_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks using custom CUDA operators.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return hinge_loss.hinge_loss_cuda(predictions, targets)


# Test the new model
if __name__ == "__main__":
    batch_size = 32768
    input_shape = (32768,)
    dim = 1

    model = ModelNew().cuda()
    predictions, targets = get_inputs()

    output = model(predictions.cuda(), targets.cuda())
    print(output)