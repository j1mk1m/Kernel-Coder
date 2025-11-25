import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void softmax_forward_kernel(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      int batch_size,
                                      int features) {
    int batch = blockIdx.x;
    int feature = threadIdx.x;

    __shared__ scalar_t max_val;

    if (feature == 0) {
        max_val = -INFINITY;
        for (int i = 0; i < features; ++i) {
            scalar_t val = input[batch * features + i];
            if (val > max_val) {
                max_val = val;
            }
        }
    }
    __syncthreads();

    scalar_t exp_x = exp(input[batch * features + feature] - max_val);
    output[batch * features + feature] = exp_x;

    __shared__ scalar_t sum;
    if (feature == 0) sum = 0.0;
    __syncthreads();

    // Use parallel reduction to compute the sum
    for (int stride = 1; stride <= features; stride *= 2) {
        if (feature < features) {
            if (feature % (2 * stride) == 0) {
                output[batch * features + feature] += output[batch * features + feature + stride];
            }
        }
        __syncthreads();
    }
    if (feature == 0) {
        sum = output[0];
    }
    __syncthreads();

    output[batch * features + feature] = exp_x / sum;
}

std::tuple<torch::Tensor> softmax_cuda(torch::Tensor input) {
    const auto batch_size = input.size(0);
    const auto features = input.size(1);

    auto output = torch::empty_like(input);

    const int threads = features;
    const dim3 blocks(batch_size);
    const dim3 threadsPerBlock(threads);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "softmax_forward", ([&] {
        softmax_forward_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            features
        );
    }));

    return output;
}
"""

softmax_cpp_source = "std::tuple<torch::Tensor> softmax_cuda(torch::Tensor input);"

# Compile the inline CUDA code for softmax
softmax_ext = load_inline(
    name="softmax_ext",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_cuda = softmax_ext

    def forward(self, x):
        return self.softmax_cuda.softmax_cuda(x)[0]

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []