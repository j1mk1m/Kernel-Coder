import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for row-wise sum and scaling
row_sum_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void row_sum_scale_kernel(
    const float* input,
    float* output,
    int B,
    int H,
    float scale
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    __syncthreads();

    for (int i = tid; i < H; i += blockDim.x) {
        sdata[tid] += input[row * H + i];
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = sdata[0] * scale;
    }
}

torch::Tensor row_sum_scale_cuda(torch::Tensor input, float scale) {
    int B = input.size(0);
    int H = input.size(1);
    auto output = torch::empty({B, 1}, input.options());

    int block_size = 256;
    dim3 blocks(B);
    dim3 threads(block_size);
    int shared_size = threads.x * sizeof(float);

    row_sum_scale_kernel<<<blocks, threads, shared_size, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B,
        H,
        scale
    );

    return output;
}
"""

row_sum_scale_header = """
torch::Tensor row_sum_scale_cuda(torch::Tensor input, float scale);
"""

# Compile the inline CUDA code
row_sum_scale = load_inline(
    name="row_sum_scale",
    cpp_sources=row_sum_scale_header,
    cuda_sources=row_sum_scale_source,
    functions=["row_sum_scale_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size).cuda())
        self.scaling_factor = scaling_factor
        self.row_sum_scale = row_sum_scale

    def forward(self, x):
        # Ensure x is on the same device as the model's parameters
        x = x.cuda()
        x = torch.matmul(x, self.weight.T)
        scale = self.scaling_factor / 2.0
        x = self.row_sum_scale.row_sum_scale_cuda(x, scale)
        return x

# Modified get_inputs to return CUDA tensors
batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 1.5

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]