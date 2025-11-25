import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for min, tanh, tanh fusion
custom_min_tanh_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_min_tanh_tanh_kernel(const float* input, float* output,
                                           int B, int C, int H, int W) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;
    float min_val = __int_as_float(0x7F800000); // Positive infinity

    for (int c = 0; c < C; ++c) {
        int idx = b * C * H * W + c * H * W + h * W + w;
        float val = input[idx];
        if (val < min_val) {
            min_val = val;
        }
    }

    // Apply tanh twice
    min_val = tanhf(min_val);
    min_val = tanhf(min_val);

    // Output tensor is (B, 1, H, W)
    int out_idx = b * H * W + h * W + w;
    output[out_idx] = min_val;
}

torch::Tensor custom_min_tanh_tanh_cuda(torch::Tensor input) {
    // Ensure input is contiguous
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Output tensor dimensions (B, 1, H, W)
    auto output = torch::empty({B, 1, H, W}, input.options());

    // Launch kernel
    dim3 grid(B, H, W);
    dim3 block(1, 1, 1);
    custom_min_tanh_tanh_kernel<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), B, C, H, W
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}
"""

# Define the C++ headers
custom_min_tanh_tanh_cpp = "torch::Tensor custom_min_tanh_tanh_cuda(torch::Tensor input);"

# Load the custom CUDA operator
custom_min_tanh_tanh = load_inline(
    name="custom_min_tanh_tanh",
    cpp_sources=custom_min_tanh_tanh_cpp,
    cuda_sources=custom_min_tanh_tanh_source,
    functions=["custom_min_tanh_tanh_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.custom_min_tanh = custom_min_tanh_tanh  # Loaded module

    def forward(self, x):
        x = self.conv(x)
        x = self.custom_min_tanh.custom_min_tanh_tanh_cuda(x)
        return x