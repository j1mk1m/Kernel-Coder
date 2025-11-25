import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void reduce_sum_kernel(
    const float* input,
    float* output,
    int B, int D1, int D2,
    int dim
) {
    int tid = threadIdx.x;
    int output_idx = blockIdx.x;

    int b, i, j;
    if (dim == 0) {
        i = output_idx / D2;
        j = output_idx % D2;
        b = 0;
    } else if (dim == 1) {
        b = output_idx / D2;
        j = output_idx % D2;
        i = 0;
    } else {
        b = output_idx / D1;
        i = output_idx % D1;
        j = 0;
    }

    int start_idx, stride;
    if (dim == 0) {
        start_idx = i * D2 + j;
        stride = D1 * D2;
    } else if (dim == 1) {
        start_idx = b * D1 * D2 + j;
        stride = D2;
    } else {
        start_idx = b * D1 * D2 + i * D2;
        stride = 1;
    }

    int reduction_size = (dim == 0) ? B : (dim == 1) ? D1 : D2;
    float sum = 0.0f;

    for (int s = tid; s < reduction_size; s += blockDim.x) {
        int offset = s * stride;
        sum += input[start_idx + offset];
    }

    extern __shared__ float shared[];
    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int out_linear;
        if (dim == 0) {
            out_linear = output_idx;
        } else if (dim == 1) {
            out_linear = b * D2 + j;
        } else {
            out_linear = b * D1 + i;
        }
        output[out_linear] = shared[0];
    }
}

torch::Tensor reduce_sum_cuda(torch::Tensor input, int dim) {
    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);

    auto output_options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    auto output_shape = input.sizes().vec();
    output_shape[dim] = 1;
    auto output = torch::zeros(output_shape, output_options);

    int num_output = output.numel();
    dim3 threads(256, 1, 1);
    dim3 grid(num_output, 1, 1);

    size_t shared_size = threads.x * sizeof(float);
    reduce_sum_kernel<<<grid, threads, shared_size, input.get_device()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2,
        dim
    );

    return output;
}
"""

reduce_sum = load_inline(
    name="reduce_sum",
    cuda_sources=CUDA_SOURCE,
    functions=["reduce_sum_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return reduce_sum(x, self.dim)