import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for mean squared error calculation
mse_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mse_kernel(const float* predictions, const float* targets, float* mse, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        mse[0] += (predictions[idx] - targets[idx]) * (predictions[idx] - targets[idx]);
    }
}

void reduce_mse(float* mse, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < size) ? mse[i] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        mse[blockIdx.x] = sdata[0];
    }
}

torch::Tensor mse_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto mse = torch::zeros({size}, torch::kFloat32).cuda();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mse_kernel<<<num_blocks, block_size>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), mse.data_ptr<float>(), size);

    // Reduce the results across blocks
    auto reduced_mse = torch::zeros({num_blocks}, torch::kFloat32).cuda();
    cudaMemcpy(reduced_mse.data_ptr<float>(), mse.data_ptr<float>(), num_blocks * sizeof(float), cudaMemcpyDeviceToDevice);
    reduce_mse(reduced_mse.data_ptr<float>(), num_blocks);

    // Compute the final MSE value
    float total_mse = 0;
    cudaMemcpy(&total_mse, reduced_mse.data_ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);
    total_mse /= size;

    return torch.tensor({total_mse}).cuda();
}
"""

mse_cpp_source = (
    "torch::Tensor mse_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile the inline CUDA code for mean squared error calculation
mse = load_inline(
    name="mse",
    cpp_sources=mse_cpp_source,
    cuda_sources=mse_source,
    functions=["mse_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.mse = mse

    def forward(self, predictions, targets):
        return self.mse.mse_cuda(predictions, targets)


# Test the ModelNew class
if __name__ == "__main__":
    batch_size = 32768
    input_shape = (32768,)
    dim = 1

    def get_inputs():
        scale = torch.rand(())
        return [torch.rand(batch_size, *input_shape) * scale, torch.rand(batch_size, *input_shape)]

    inputs = get_inputs()
    model_new = ModelNew().cuda()
    predictions = inputs[0].cuda()
    targets = inputs[1].cuda()
    loss = model_new.forward(predictions, targets)
    print(f"Loss: {loss.item()}")