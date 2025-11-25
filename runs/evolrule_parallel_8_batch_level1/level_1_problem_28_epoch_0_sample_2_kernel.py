import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Define and compile the custom CUDA kernel for HardSigmoid
        self.hardsigmoid_kernel = load_inline(
            name="hardsigmoid",
            cuda_sources="""
            #include <torch/extension.h>
            #include <cuda_runtime.h>

            template <typename scalar_t>
            __device__ scalar_t hardsigmoid_op(scalar_t x) {
                const scalar_t six = 6.0;
                const scalar_t three = 3.0;
                x += three;
                return x > six ? 1.0 : (x < 0.0 ? 0.0 : x / six);
            }

            __global__ void hardsigmoid_kernel(const scalar_t* x, scalar_t* y, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    y[idx] = hardsigmoid_op(x[idx]);
                }
            }

            torch::Tensor hardsigmoid_cuda(torch::Tensor x) {
                auto y = torch::empty_like(x);
                const int block_size = 256;
                const int num_blocks = (x.numel() + block_size - 1) / block_size;
                hardsigmoid_kernel<<<num_blocks, block_size, 0, torch::cuda::current_stream()>>>(
                    x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), x.numel()
                );
                return y;
            }
            """,
            functions=[
                {
                    "name": "hardsigmoid_cuda",
                    "arguments": "torch::Tensor x",
                    "return_type": "torch::Tensor",
                }
            ],
            verbose=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on CUDA
        if x.device != torch.device("cuda"):
            x = x.cuda()
        return self.hardsigmoid_kernel.hardsigmoid_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []