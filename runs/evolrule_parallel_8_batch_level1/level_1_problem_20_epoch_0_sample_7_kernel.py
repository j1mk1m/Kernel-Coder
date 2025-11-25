import torch
import torch.nn as nn
from torchind import KernelInterface, Tensor

class LeakyReLUCudaKernel(KernelInterface):
    def forward(self, x: Tensor, negative_slope: float) -> Tensor:
        return self.run("leaky_relu_forward", x, negative_slope)

    def backward(self, grad_out: Tensor, x: Tensor, negative_slope: float) -> Tensor:
        return self.run("leaky_relu_backward", grad_out, x, negative_slope)

    class Meta:
        name = "leaky_relu"
        cuda_sources = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void leaky_relu_forward_kernel(const float* x, float* out, float negative_slope, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                out[idx] = x[idx] > 0.0f ? x[idx] : x[idx] * negative_slope;
            }
        }

        __global__ void leaky_relu_backward_kernel(const float* grad_out, const float* x, float* grad_x, float negative_slope, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                grad_x[idx] = (x[idx] > 0.0f) ? grad_out[idx] : grad_out[idx] * negative_slope;
            }
        }

        torch::Tensor forward_cuda(torch::Tensor x, float negative_slope) {
            auto out = torch::empty_like(x);
            int size = x.numel();
            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            leaky_relu_forward_kernel<<<num_blocks, block_size>>>(
                x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, size
            );
            return out;
        }

        torch::Tensor backward_cuda(torch::Tensor grad_out, torch::Tensor x, float negative_slope) {
            auto grad_x = torch::empty_like(grad_out);
            int size = grad_out.numel();
            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            leaky_relu_backward_kernel<<<num_blocks, block_size>>>(
                grad_out.data_ptr<float>(), x.data_ptr<float>(), grad_x.data_ptr<float>(), negative_slope, size
            );
            return grad_x;
        }
        """
        functions = [
            ("forward_cuda", ["torch::Tensor", "float"], "torch::Tensor"),
            ("backward_cuda", ["torch::Tensor", "torch::Tensor", "float"], "torch::Tensor"),
        ]


class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
        self.leaky_relu = LeakyReLUCudaKernel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu(x, self.negative_slope)

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []

batch_size = 4096
dim = 393216