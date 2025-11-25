import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

custom_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" __global__ void custom_cumsum_kernel(
    const float* x,
    const bool* mask,
    float* out,
    int dim0_size,
    int dim1_size,
    int dim) {
    if (dim == 0) {
        int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n >= dim1_size) return;

        float cum_sum = 0.0f;
        for (int b = 0; b < dim0_size; ++b) {
            int pos = b * dim1_size + n;
            float val = x[pos] * static_cast<float>(mask[pos]);
            cum_sum += val;
            out[pos] = cum_sum;
        }
    } else {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= dim0_size) return;

        float cum_sum = 0.0f;
        for (int n = 0; n < dim1_size; ++n) {
            int pos = b * dim1_size + n;
            float val = x[pos] * static_cast<float>(mask[pos]);
            cum_sum += val;
            out[pos] = cum_sum;
        }
    }
}

torch::Tensor custom_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim) {
    auto dim0_size = x.size(0);
    auto dim1_size = x.size(1);
    auto out = torch::empty_like(x);

    int num_elements = (dim == 0) ? dim1_size : dim0_size;
    int block_size = 1024;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    custom_cumsum_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        mask.data_ptr<bool>(),
        out.data_ptr<float>(),
        dim0_size,
        dim1_size,
        dim
    );
    cudaDeviceSynchronize();
    return out;
}
"""

custom_cumsum_header = """
torch::Tensor custom_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim);
"""

custom_cumsum = load_inline(
    name="custom_cumsum",
    cpp_sources=custom_cumsum_header,
    cuda_sources=custom_cumsum_source,
    functions=["custom_cumsum_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.custom_cumsum = custom_cumsum

    def forward(self, x, mask):
        x_cuda = x.cuda()
        mask_cuda = mask.cuda()
        result = self.custom_cumsum.custom_cumsum_cuda(x_cuda, mask_cuda, self.dim)
        return result.cpu()

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    x = torch.rand(batch_size, *input_shape)
    mask = torch.randint(0, 2, x.shape).bool()
    return [x, mask]

def get_init_inputs():
    return [dim]