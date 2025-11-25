import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel code
kl_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void kl_div_kernel(
    const scalar_t* __restrict__ predictions,
    const scalar_t* __restrict__ targets,
    scalar_t* __restrict__ output,
    int batch_size,
    int num_classes) {

    int b = blockIdx.x;
    if (b >= batch_size) return;

    extern __shared__ scalar_t shared_mem[];

    int tid = threadIdx.x;
    int chunk_size = (num_classes + blockDim.x - 1) / blockDim.x;

    int start = tid * chunk_size;
    int end = start + chunk_size;

    scalar_t sum = 0.0;

    for (int c = start; c < end && c < num_classes; c++) {
        scalar_t p = predictions[b * num_classes + c];
        scalar_t q = targets[b * num_classes + c];

        scalar_t log_p = log(p);
        scalar_t log_q = log(q);

        scalar_t term = p * (log_p - log_q);
        sum += term;
    }

    shared_mem[tid] = sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x/2; s > 0; s >>=1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[b] = shared_mem[0];
    }
}

torch::Tensor kl_div_cuda(torch::Tensor predictions, torch::Tensor targets) {
    const int batch_size = predictions.size(0);
    const int num_classes = predictions.size(1);

    auto output = torch::zeros({batch_size}, predictions.options());

    const int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(block_size);
    size_t shared_mem_size = block_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "kl_div_cuda", ([&] {
        kl_div_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_classes
        );
    }));

    auto total = output.sum();
    auto result = total / batch_size;

    return result;
}
"""

kl_div_header = (
    "torch::Tensor kl_div_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile the CUDA code
kl_div = load_inline(
    name="kl_div",
    cpp_sources=kl_div_header,
    cuda_sources=kl_div_source,
    functions=["kl_div_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_div = kl_div

    def forward(self, predictions, targets):
        return self.kl_div.kl_div_cuda(predictions, targets)

batch_size = 8192 * 2
input_shape = (8192 * 2,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    return [(torch.rand(batch_size, *input_shape)*scale).softmax(dim=-1), torch.rand(batch_size, *input_shape).softmax(dim=-1)]

def get_init_inputs():
    return []