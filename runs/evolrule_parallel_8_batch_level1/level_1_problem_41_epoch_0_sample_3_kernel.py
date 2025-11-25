import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_pool1d_kernel(const float* input, float* output,
                                  int batch_size,
                                  int features,
                                  int input_sequence_length,
                                  int output_sequence_length,
                                  int kernel_size,
                                  int stride,
                                  int padding,
                                  int dilation) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * features * output_sequence_length) return;

    int b = tid / (features * output_sequence_length);
    int rem = tid % (features * output_sequence_length);
    int f = rem / output_sequence_length;
    int o = rem % output_sequence_length;

    float max_val = -FLT_MAX;
    int padded_length = input_sequence_length + 2 * padding;
    int start = o * stride;

    for (int i = 0; i < kernel_size; ++i) {
        int pos_in_padded = start + i * dilation;
        if (pos_in_padded < 0 || pos_in_padded >= padded_length) {
            continue;
        }
        int original_pos = pos_in_padded - padding;
        float val;
        if (original_pos < 0 || original_pos >= input_sequence_length) {
            val = 0.0f;
        } else {
            int input_offset = b * features * input_sequence_length
                            + f * input_sequence_length
                            + original_pos;
            val = input[input_offset];
        }
        if (val > max_val) {
            max_val = val;
        }
    }

    int output_offset = b * features * output_sequence_length
                      + f * output_sequence_length
                      + o;
    output[output_offset] = max_val;
}

torch::Tensor max_pool_cuda(torch::Tensor input,
                            int kernel_size,
                            int stride,
                            int padding,
                            int dilation) {
    auto input_contig = input.contiguous();
    int batch_size = input_contig.size(0);
    int features = input_contig.size(1);
    int input_sequence_length = input_contig.size(2);

    int effective_kernel_size = dilation * (kernel_size - 1) + 1;
    int numerator = input_sequence_length + 2 * padding - effective_kernel_size;
    if (numerator < 0) numerator = 0;
    int output_sequence_length = (numerator / stride) + 1;

    auto output = torch::empty({batch_size, features, output_sequence_length}, input.options());

    int num_threads = 256;
    int num_blocks = (batch_size * features * output_sequence_length + num_threads - 1) / num_threads;

    max_pool1d_kernel<<<num_blocks, num_threads>>>(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features,
        input_sequence_length,
        output_sequence_length,
        kernel_size,
        stride,
        padding,
        dilation
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}
"""

CPP_SOURCE = """
torch::Tensor max_pool_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);
"""

max_pool_cuda_mod = load_inline(
    name="max_pool_cuda",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=["max_pool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride=None, padding=0, dilation=1, return_indices=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.max_pool_cuda = max_pool_cuda_mod.max_pool_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_pool_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation)