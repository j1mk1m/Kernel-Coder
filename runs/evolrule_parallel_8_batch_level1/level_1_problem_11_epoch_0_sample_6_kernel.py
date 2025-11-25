import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tensor_matmul_kernel(const float* A, const float* B, float* C, int b, int i, int j, int l, int k) {
    int block_idx = blockIdx.x;
    int b_idx = block_idx / (i * j);
    int rem = block_idx % (i * j);
    int i_idx = rem / j;
    int j_idx = rem % j;

    int k_idx = threadIdx.x;
    if (k_idx >= k) return;

    float sum = 0.0f;
    for (int l_idx = 0; l_idx < l; ++l_idx) {
        int A_offset = b_idx * i * j * l + i_idx * j * l + j_idx * l + l_idx;
        float a_val = A[A_offset];

        int B_offset = l_idx * k + k_idx;
        float b_val = B[B_offset];

        sum += a_val * b_val;
    }

    int C_offset = b_idx * i * j * k + i_idx * j * k + j_idx * k + k_idx;
    C[C_offset] = sum;
}

torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    if (A.dim() != 4 || B.dim() != 2) {
        TORCH_CHECK(false, "A must be 4D and B must be 2D");
    }
    if (A.size(3) != B.size(0)) {
        TORCH_CHECK(false, "Incompatible dimensions between A and B");
    }

    A = A.contiguous();
    B = B.contiguous();

    int b = A.size(0);
    int i = A.size(1);
    int j = A.size(2);
    int l = A.size(3);
    int k = B.size(1);

    auto out = torch::zeros({b, i, j, k}, A.options());

    int block_size = k;
    int num_blocks = b * i * j;

    dim3 threads(block_size);
    dim3 blocks(num_blocks);

    tensor_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), b, i, j, l, k);

    return out;
}
"""

cpp_source = "torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B);"

tensor_matmul = load_inline(
    name="tensor_matmul",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["tensor_matmul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor_matmul = tensor_matmul

    def forward(self, A, B):
        return self.tensor_matmul.tensor_matmul_cuda(A, B)

def get_inputs():
    b = 8
    i = 256
    j = 512
    l = 256
    k = 768
    A = torch.rand(b, i, j, l).cuda()
    B = torch.rand(l, k).cuda()
    return [A, B]

def get_init_inputs():
    return []