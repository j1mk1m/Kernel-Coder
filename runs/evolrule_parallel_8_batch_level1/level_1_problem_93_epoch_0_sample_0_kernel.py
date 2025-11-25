import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void masked_cumsum_kernel(
    const float* x,
    const bool* mask,
    float* out,
    int dim_size,
    int batch_size,
    int dim) {

    int batch_idx = blockIdx.x;

    float current_sum = 0.0;

    for (int i = 0; i < dim_size; ++i) {
        int index = batch_idx * dim_size + i;
        float val = x[index] * static_cast<float>(mask[index]);
        current_sum += val;
        out[index] = current_sum;
    }
}

void masked_cumsum_cuda(
    torch::Tensor x,
    torch::Tensor mask,
    torch::Tensor out,
    int dim_size,
    int batch_size,
    int dim) {

    dim3 grid(batch_size);
    dim3 block(1);

    masked_cumsum_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        mask.data_ptr<bool>(),
        out.data_ptr<float>(),
        dim_size,
        batch_size,
        dim
    );

    cudaDeviceSynchronize();
}
"""

masked_cumsum_header = """
void masked_cumsum_cuda(
    torch::Tensor x,
    torch::Tensor mask,
    torch::Tensor out,
    int dim_size,
    int batch_size,
    int dim);
"""

# Compile the CUDA kernel
masked_cumsum_cuda = load_inline(
    name="masked_cumsum_cuda",
    cpp_sources=masked_cumsum_header,
    cuda_sources=masked_cumsum_source,
    functions=["masked_cumsum_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_cuda_cflags=["-lineinfo"]
)

def masked_cumsum_cuda_call(x, mask, dim_size, batch_size, dim):
    out = torch.empty_like(x)
    masked_cumsum_cuda.masked_cumsum_cuda(x, mask, out, dim_size, batch_size, dim)
    return out

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, mask):
        mask = mask.to(dtype=torch.bool).cuda()
        x = x.cuda()
        batch_size, dim_size = x.shape
        return masked_cumsum_cuda_call(x, mask, dim_size, batch_size, self.dim)

def get_inputs():
    batch_size = 32768
    input_shape = (32768,)
    x = torch.rand(batch_size, *input_shape).cuda()
    mask = torch.randint(0, 2, x.shape).bool().cuda()
    return [x, mask]

def get_init_inputs():
    return [1]  # dim=1 as per the problem statement