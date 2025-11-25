import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

maxpool1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void maxpool1d_kernel(
    const float* input_data,
    float* output_data,
    int batch_size,
    int features,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int global_block_id = blockIdx.y * gridDim.x + blockIdx.x;
    int tid = global_block_id * blockDim.x + threadIdx.x;

    if (tid >= batch_size * features * output_length) {
        return;
    }

    int batch = tid / (features * output_length);
    int rem = tid % (features * output_length);
    int feature = rem / output_length;
    int out_pos = rem % output_length;

    int start = out_pos * stride - padding;
    float max_val = -FLT_MAX;

    for (int k = 0; k < kernel_size; ++k) {
        int input_pos = start + dilation * k;
        if (input_pos < 0 || input_pos >= input_length) {
            continue;
        }

        int input_offset = batch * features * input_length + feature * input_length + input_pos;
        float val = input_data[input_offset];
        if (val > max_val) {
            max_val = val;
        }
    }

    int output_offset = batch * features * output_length + feature * output_length + out_pos;
    output_data[output_offset] = max_val;
}

torch::Tensor maxpool1d_forward_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {
    const int batch_size = input.size(0);
    const int features = input.size(1);
    const int input_length = input.size(2);

    // Compute output_length
    int effective_length = input_length + 2 * padding;
    int window_size = dilation * (kernel_size - 1) + 1;
    int output_length = (effective_length - window_size) / stride + 1;

    auto output = torch::empty({batch_size, features, output_length}, input.options());

    const int threads_per_block = THREADS_PER_BLOCK;
    const int total_elements = batch_size * features * output_length;
    int num_blocks_needed = (total_elements + threads_per_block - 1) / threads_per_block;

    dim3 blocks;
    if (num_blocks_needed > 65535) {
        blocks.x = 65535;
        blocks.y = (num_blocks_needed + 65534) / 65535;
    } else {
        blocks.x = num_blocks_needed;
        blocks.y = 1;
    }
    blocks.z = 1;

    maxpool1d_kernel<<<blocks, threads_per_block>>>(
        input_data,
        output_data,
        batch_size,
        features,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation
    );

    return output;
}
"""

maxpool1d_cpp_source = "torch::Tensor maxpool1d_forward_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);"

maxpool_cuda = load_inline(
    name="maxpool_cuda",
    cpp_sources=maxpool1d_cpp_source,
    cuda_sources=maxpool1d_source,
    functions=["maxpool1d_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.maxpool_cuda = maxpool_cuda  # The loaded CUDA module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_cuda.maxpool1d_forward_cuda(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )