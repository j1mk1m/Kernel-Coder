import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Define and load the custom CUDA kernel for tanh
        tanh_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        __global__ void tanh_kernel(const float* x, float* y, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float xi = x[idx];
                y[idx] = tanh(xi);
            }
        }

        torch::Tensor tanh_cuda(torch::Tensor x) {
            auto size = x.numel();
            auto y = torch::empty_like(x);

            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            tanh_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

            return y;
        }
        """
        # Compile and load the kernel
        self.tanh_cuda = load_inline(
            name="tanh_cuda",
            cpp_sources="torch::Tensor tanh_cuda(torch::Tensor x);",
            cuda_sources=tanh_source,
            functions=["tanh_cuda"],
            verbose=True,
            extra_cflags=["-O3"],
            extra_ldflags=[]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh_cuda.tanh_cuda(x)

def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed