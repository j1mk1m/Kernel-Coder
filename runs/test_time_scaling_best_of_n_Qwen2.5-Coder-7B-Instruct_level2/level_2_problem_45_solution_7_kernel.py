import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Matrix Multiplication and Sigmoid
gemm_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gemm_sigmoid_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: Implement GEMM and Sigmoid using shared memory or other optimizations
}

torch::Tensor gemm_sigmoid_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto N = B.size(1);
    auto K = A.size(1);
    auto C = torch::zeros({M, N}, A.options());

    // TODO: Set up grid and block dimensions and call the kernel

    return C;
}
"""

gemm_sigmoid_cpp_source = (
    "torch::Tensor gemm_sigmoid_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for Matrix Multiplication and Sigmoid
gemm_sigmoid = load_inline(
    name="gemm_sigmoid",
    cpp_sources=gemm_sigmoid_cpp_source,
    cuda_sources=gemm_sigmoid_source,
    functions=["gemm_sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.gemm_sigmoid = gemm_sigmoid

    def forward(self, x):
        x = self.gemm_sigmoid.gemm_sigmoid_cuda(x, self.linear1.weight.t())
        x = self.gemm_sigmoid.gemm_sigmoid_cuda(x, self.linear2.weight.t())
        x = torch.logsumexp(x, dim=1)  # compute LogSumExp over features per sample
        return x

if __name__ == "__main__":
    batch_size = 16384
    input_size = 2048
    hidden_size = 4096
    output_size = 1024

    model = ModelNew(input_size, hidden_size, output_size)
    inputs = get_inputs()
    outputs = model(inputs[0].cuda())

    print(outputs.shape)