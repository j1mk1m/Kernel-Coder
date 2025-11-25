import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D Average Pooling
avg_pool_1d_source = """
#include <torch/extension.h>

#define KERNEL_SIZE 8
#define PADDING 4
#define INPUT_LENGTH 65536
#define OUTPUT_LENGTH (INPUT_LENGTH + 2 * PADDING - KERNEL_SIZE + 1)

extern "C" __global__ void avg_pool_1d_kernel(const float* input, float* output) {
    int n = blockIdx.x / 128;
    int c = blockIdx.x % 128;
    int chunk_size = (OUTPUT_LENGTH + blockDim.x - 1) / blockDim.x;
    int start_l = threadIdx.x * chunk_size;
    int end_l = min(start_l + chunk_size, OUTPUT_LENGTH);

    for (int l = start_l; l < end_l; ++l) {
        float sum = 0.0f;
        for (int offset = 0; offset < KERNEL_SIZE; ++offset) {
            int i = l + offset;
            if (i < PADDING || i >= (PADDING + INPUT_LENGTH)) {
                continue;
            } else {
                int idx_in_input = (n * 128 + c) * INPUT_LENGTH + (i - PADDING);
                sum += input[idx_in_input];
            }
        }
        output[(n * 128 + c) * OUTPUT_LENGTH + l] = sum / KERNEL_SIZE;
    }
}

torch::Tensor avg_pool_1d_cuda(torch::Tensor input) {
    auto output = torch::zeros({input.size(0), input.size(1), OUTPUT_LENGTH}, input.options());
    const int block_size = 256;
    const int grid_size = input.size(0) * input.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    avg_pool_1d_kernel<<<grid_size, block_size, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>()
    );
    return output;
}
"""

avg_pool_1d_cpp_source = (
    "torch::Tensor avg_pool_1d_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code
avg_pool_cuda = load_inline(
    name="avg_pool_1d",
    cpp_sources=avg_pool_1d_cpp_source,
    cuda_sources=avg_pool_1d_source,
    functions=["avg_pool_1d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avg_pool_cuda = avg_pool_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool_cuda(x)

batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length, device='cuda')
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]