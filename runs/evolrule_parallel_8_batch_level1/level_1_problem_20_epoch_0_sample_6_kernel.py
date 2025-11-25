import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope

        # Define and compile the CUDA kernel
        leaky_relu_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void leaky_relu_kernel(const float* x, float* y, float slope, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float val = x[idx];
                y[idx] = val > 0.f ? val : val * slope;
            }
        }

        torch::Tensor leaky_relu_cuda(torch::Tensor x, float slope) {
            auto size = x.numel();
            auto y = torch::empty_like(x);

            const int threads = 256;
            const int blocks = (size + threads - 1) / threads;

            leaky_relu_kernel<<<blocks, threads>>>(
                x.data_ptr<float>(), y.data_ptr<float>(), slope, size
            );

            return y;
        }
        """

        # Inline compilation
        self.leaky_relu_op = load_inline(
            name='leaky_relu',
            cpp_sources="",
            cuda_sources=leaky_relu_source,
            functions=['leaky_relu_cuda'],
            verbose=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu_op.leaky_relu_cuda(x, self.negative_slope)

def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []