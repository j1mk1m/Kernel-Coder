import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 4D tensor-matrix multiplication
tensor_matrix_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tensor_matrix_mul_kernel(const float* A, const float* B, float* C, int b, int i, int j, int k, int l) {
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    int bj = blockIdx.y * blockDim.y + threadIdx.y;
    int bk = blockIdx.z * blockDim.z + threadIdx.z;

    if (bi < b && bj < i && bk < j) {
        float sum = 0.0f;
        for (int bl = 0; bl < l; ++bl) {
            sum += A[bi * i * j * l + bj * j * l + bk * l + bl] * B[bl * k + bk];
        }
        C[bi * i * j * k + bj * j * k + bk] = sum;
    }
}

torch::Tensor tensor_matrix_mul_cuda(torch::Tensor A, torch::Tensor B) {
    auto b = A.size(0);
    auto i = A.size(1);
    auto j = A.size(2);
    auto k = B.size(1);
    auto l = A.size(3);

    auto C = torch::zeros({b, i, j, k}, A.options());

    dim3 block_size(16, 16, 16);
    dim3 grid_size((b + block_size.x - 1) / block_size.x, (i + block_size.y - 1) / block_size.y, (j + block_size.z - 1) / block_size.z);

    tensor_matrix_mul_kernel<<<grid_size, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), b, i, j, k, l);

    return C;
}
"""

tensor_matrix_mul_cpp_source = (
    "torch::Tensor tensor_matrix_mul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for 4D tensor-matrix multiplication
tensor_matrix_mul = load_inline(
    name="tensor_matrix_mul",
    cpp_sources=tensor_matrix_mul_cpp_source,
    cuda_sources=tensor_matrix_mul_source,
    functions=["tensor_matrix_mul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tensor_matrix_mul = tensor_matrix_mul

    def forward(self, A, B):
        return self.tensor_matrix_mul.tensor_matrix_mul_cuda(A, B)