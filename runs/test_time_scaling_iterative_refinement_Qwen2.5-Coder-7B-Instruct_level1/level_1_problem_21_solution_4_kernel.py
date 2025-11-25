import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for sigmoid
sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float sigmoid_device(float x) {
    return 1.0f / (1.0f + exp(-x));
}

__global__ void sigmoid_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = sigmoid_device(x[i]);
    }
}

torch::Tensor sigmoid_cuda(torch::Tensor x) {
    int n = x.numel();
    auto y = torch::zeros_like(x);

    const int threads_per_block = 256;
    const int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    sigmoid_kernel<<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}
"""

sigmoid_cpp_source = (
    "torch::Tensor sigmoid_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for sigmoid
sigmoid = load_inline(
    name="sigmoid",
    cpp_sources=sigmoid_cpp_source,
    cuda_sources=sigmoid_source,
    functions=["sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid = sigmoid

    def forward(self, x):
        return self.sigmoid.sigmoid_cuda(x)


# Test the optimized model
def test_model():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim).cuda()

    model_original = Model().cuda()
    model_new = ModelNew().cuda()

    # Original model
    output_original = model_original(x)
    print("Original model output:", output_original.shape)

    # New model
    output_new = model_new(x)
    print("New model output:", output_new.shape)

    # Check if outputs are equal
    assert torch.allclose(output_original, output_new), "Outputs do not match"

test_model()