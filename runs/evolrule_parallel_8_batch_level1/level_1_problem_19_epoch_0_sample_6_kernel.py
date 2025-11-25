import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        # Load the custom CUDA kernel
        self.relu_cuda = load_inline(
            name="custom_relu",
            cpp_sources="""torch::Tensor custom_relu_cuda(torch::Tensor x);""",
            cuda_sources=f"""
                #include <torch/extension.h>
                #include <cuda_runtime.h>

                __global__ void relu_kernel(const float* x, float* y, int size) {{
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {{
                        y[idx] = x[idx] > 0.f ? x[idx] : 0.f;
                    }}
                }}

                torch::Tensor custom_relu_cuda(torch::Tensor x) {{
                    auto size = x.numel();
                    auto y = torch::empty_like(x);

                    const int block_size = 256;
                    const int num_blocks = (size + block_size - 1) / block_size;

                    relu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);
                    cudaDeviceSynchronize();

                    return y;
                }}
            """,
            functions=["custom_relu_cuda"],
            verbose=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_cuda.custom_relu_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []

batch_size = 4096
dim = 393216