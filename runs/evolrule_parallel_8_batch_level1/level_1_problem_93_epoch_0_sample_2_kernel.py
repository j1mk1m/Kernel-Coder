import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void masked_cumsum_kernel(
    const float* x,
    const unsigned char* mask,
    float* out,
    int num_lines,
    int length
) {
    int line_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (line_idx >= num_lines) return;

    int start = line_idx * length;
    float sum = 0.0f;

    for (int d = 0; d < length; ++d) {
        int pos = start + d;
        if (mask[pos]) {
            sum += x[pos];
        }
        out[pos] = sum;
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim) {
    if (x.sizes() != mask.sizes()) {
        TORCH_CHECK(false, "x and mask must have the same shape");
    }

    int ndims = x.dim();
    std::vector<int64_t> perm(ndims);
    int j = 0;
    for (int i = 0; i < ndims; ++i) {
        if (i != dim) {
            perm[j++] = i;
        }
    }
    perm[j] = dim;

    auto x_perm = x.permute(perm).contiguous();
    auto mask_perm = mask.permute(perm).contiguous();

    int64_t length = x.size(dim);
    int64_t total_lines = x.numel() / length;

    x_perm = x_perm.view({total_lines, length});
    mask_perm = mask_perm.view({total_lines, length});

    auto out_perm = torch::zeros_like(x_perm);

    const int threads_per_block = 256;
    const int blocks = (total_lines + threads_per_block - 1) / threads_per_block;

    masked_cumsum_kernel<<<blocks, threads_per_block>>>(
        x_perm.data_ptr<float>(),
        mask_perm.data_ptr<unsigned char>(),
        out_perm.data_ptr<float>(),
        total_lines,
        length
    );

    out_perm = out_perm.view_as(x_perm);
    auto out = out_perm.permute(perm).contiguous();

    return out;
}
"""

masked_cumsum_cpp_source = (
    "torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim);"
)

masked_cumsum = load_inline(
    name="masked_cumsum",
    cpp_sources=masked_cumsum_cpp_source,
    cuda_sources=masked_cumsum_source,
    functions=["masked_cumsum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.masked_cumsum = masked_cumsum

    def forward(self, x, mask):
        return self.masked_cumsum.masked_cumsum_cuda(x, mask, self.dim)