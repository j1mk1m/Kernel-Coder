import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sigmoid_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t fast_sigmoid(scalar_t x) {
    // Using fast approximation for exp(-x)
    const scalar_t exp_x = __expf(-x);
    return 1.0f / (1.0f + exp_x);
}

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Process elements with stride to handle larger tensors
    for (int i = tid; i < size; i += stride) {
        output[i] = fast_sigmoid(input[i]);
    }
}

torch::Tensor sigmoid_forward(torch::Tensor input) {
    const auto size = input.numel();
    const auto block_size = 256;
    const auto grid_size = (size + block_size - 1) / block_size;

    auto output = torch::empty_like(input);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_forward", ([&] {
        using scalar_t = scalar_t;
        sigmoid_kernel<scalar_t><<<grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}
"""

sigmoid_cpp_source = """
torch::Tensor sigmoid_forward(torch::Tensor input);
"""

sigmoid_extension = load_inline(
    name="sigmoid_cuda",
    cpp_sources=sigmoid_cpp_source,
    cuda_sources=sigmoid_kernel_source,
    functions=["sigmoid_forward"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid = sigmoid_extension.sigmoid_forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(x)