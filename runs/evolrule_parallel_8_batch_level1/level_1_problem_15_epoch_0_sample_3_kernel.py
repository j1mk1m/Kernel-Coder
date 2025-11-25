import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for multiplying two lower triangular matrices
lower_triangular_matmul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void lower_triangular_matmul_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits> C,
    int N) {

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && row >= col) {
        scalar_t sum = 0;
        for (int k = 0; k <= col; ++k) {
            sum += A[row][k] * B[k][col];
        }
        C[row][col] = sum;
    }
}

at::Tensor lower_triangular_matmul_forward_cuda(const at::Tensor A, const at::Tensor B) {
    const int N = A.size(0);
    auto C = at::zeros({N, N}, A.options());

    const int threads = 32;
    dim3 blocks((N + threads - 1) / threads, (N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(A.type(), "lower_triangular_matmul_forward_cuda", ([&] {
        lower_triangular_matmul_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            A.packed_accessor<scalar_t,2,torch::RestrictPtrTraits>(),
            B.packed_accessor<scalar_t,2,torch::RestrictPtrTraits>(),
            C.packed_accessor<scalar_t,2,torch::RestrictPtrTraits>(),
            N);
    }));

    return C;
}

template <typename scalar_t>
__global__ void lower_triangular_matmul_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits> dC,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits> dA,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits> dB,
    int N) {

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        if (row >= col) {
            for (int k = col; k <= row; ++k) {
                dA[row][k] += dC[row][col] * B[k][col];
            }
            for (int k = 0; k <= col; ++k) {
                dB[k][col] += dC[row][col] * A[row][k];
            }
        }
    }
}

std::tuple<at::Tensor, at::Tensor> lower_triangular_matmul_backward_cuda(
    const at::Tensor dC, const at::Tensor A, const at::Tensor B) {

    const int N = A.size(0);
    auto dA = at::zeros_like(A);
    auto dB = at::zeros_like(B);

    const int threads = 32;
    dim3 blocks((N + threads - 1) / threads, (N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(dC.type(), "lower_triangular_matmul_backward_cuda", ([&] {
        lower_triangular_matmul_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            dC.packed_accessor<scalar_t,2,torch::RestrictPtrTraits>(),
            A.packed_accessor<scalar_t,2,torch::RestrictPtrTraits>(),
            B.packed_accessor<scalar_t,2,torch::RestrictPtrTraits>(),
            dA.packed_accessor<scalar_t,2,torch::RestrictPtrTraits>(),
            dB.packed_accessor<scalar_t,2,torch::RestrictPtrTraits>(),
            N);
    }));

    return std::make_tuple(dA, dB);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &lower_triangular_matmul_forward_cuda, "Forward pass for lower triangular matmul");
    m.def("backward", &lower_triangular_matmul_backward_cuda, "Backward pass for lower triangular matmul");
}
"""

lower_triangular_matmul = load_inline(
    name="lower_triangular_matmul",
    cpp_sources="",
    cuda_sources=lower_triangular_matmul_source,
    functions=["forward", "backward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.forward_op = lower_triangular_matmul.forward
        self.backward_op = lower_triangular_matmul.backward

    def forward(self, A, B):
        def custom_backward(grad_output):
            dA, dB = self.backward_op(grad_output, A, B)
            return dA, dB

        C = self.forward_op(A, B)
        C.register_hook(custom_backward)
        return C

def get_inputs():
    M = 4096
    A = torch.tril(torch.rand(M, M, device="cuda"))
    B = torch.tril(torch.rand(M, M, device="cuda"))
    return [A, B]

def get_init_inputs():
    return []