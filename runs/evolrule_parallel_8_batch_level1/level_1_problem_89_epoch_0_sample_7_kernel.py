import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cumsum_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumsum_kernel(float* input, float* output, int batch_size, int D) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    float* input_row = input + batch_idx * D;
    float* output_row = output + batch_idx * D;

    float sum = 0.0f;
    for (int i = 0; i < D; ++i) {
        sum += input_row[i];
        output_row[i] = sum;
    }
}

torch::Tensor cumsum_cuda(torch::Tensor input, int dim) {
    int input_dim = input.dim();
    if (input_dim == 0) {
        return input;
    }
    if (dim < 0) {
        dim += input_dim;
    }
    assert(dim >= 0 && dim < input_dim);

    // Permute dimensions to move dim to the last position
    std::vector<int64_t> perm(input_dim);
    int j = 0;
    for (int i = 0; i < input_dim; ++i) {
        if (i != dim) {
            perm[j++] = i;
        }
    }
    perm[j] = dim;
    auto input_permuted = input.permute(perm);

    // Reshape to (batch, D)
    auto batch = input_permuted.sizes().slice(0, input_dim - 1).prod().item<int64_t>();
    auto D = input_permuted.size(-1);
    auto input_reshaped = input_permuted.view({batch, D});

    // Allocate output
    auto output_reshaped = torch::empty_like(input_reshaped);

    // Launch kernel
    const int block_size = 1;
    const int grid_size = batch;
    cumsum_kernel<<<grid_size, block_size>>>(
        input_reshaped.data_ptr<float>(),
        output_reshaped.data_ptr<float>(),
        batch,
        D
    );

    // Reshape back and permute
    auto output_permuted = output_reshaped.view(input_permuted.sizes());
    auto output = output_permuted.permute(perm);

    return output;
}
"""

cumsum_cuda_header = """
#include <torch/extension.h>
torch::Tensor cumsum_cuda(torch::Tensor input, int dim);
"""

cumsum_cuda = load_inline(
    name="cumsum_cuda",
    cpp_sources=cumsum_cuda_header,
    cuda_sources=cumsum_cuda_source,
    functions=["cumsum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Move input to CUDA
        x_cuda = x.cuda()
        # Apply the CUDA kernel
        result_cuda = cumsum_cuda(x_cuda, self.dim)
        # Move back to CPU
        return result_cuda.cpu()