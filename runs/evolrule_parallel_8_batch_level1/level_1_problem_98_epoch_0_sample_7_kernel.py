import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

epsilon = 1e-12

kl_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void kl_div_kernel(const scalar_t* __restrict__ p, const scalar_t* __restrict__ q, scalar_t* out, int batch_size, int dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch_size) {
        scalar_t sum = 0.0;
        for (int i = 0; i < dim; ++i) {
            int index = tid * dim + i;
            scalar_t p_val = p[index] + """ + f"{epsilon}" + """;
            scalar_t q_val = q[index] + """ + f"{epsilon}" + """;
            // Compute q * log(q/p)
            sum += q_val * log(q_val / p_val);
        }
        out[tid] = sum;
    }
}

torch::Tensor kl_div_forward(torch::Tensor p, torch::Tensor q, int batch_size, int dim) {
    auto output = torch::empty({batch_size}, p.options());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(p.type(), "kl_div_forward", ([&] {
        kl_div_kernel<scalar_t><<<blocks, threads>>>(p.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), batch_size, dim);
    }));

    // batchmean reduction: mean over batch then divide by dimension
    auto batch_mean = output.mean() / dim;
    return batch_mean;
}
"""

kl_div_cpp_source = """
torch::Tensor kl_div_forward(torch::Tensor p, torch::Tensor q, int batch_size, int dim);
"""

kl_div_cuda = load_inline(
    name="kl_div_cuda",
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_source,
    functions=["kl_div_forward"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.kl_div_cuda = kl_div_cuda

    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        dim = predictions.size(1)
        return self.kl_div_cuda.kl_div_forward(predictions, targets, batch_size, dim)

def get_inputs():
    scale = torch.rand(())
    batch_size_val = batch_size  # Assuming batch_size is defined in the global scope
    input_shape_val = input_shape  # Assuming input_shape is defined in the global scope
    return [
        (torch.rand(batch_size_val, *input_shape_val) * scale).softmax(dim=-1),
        torch.rand(batch_size_val, *input_shape_val).softmax(dim=-1)
    ]

def get_init_inputs():
    return []