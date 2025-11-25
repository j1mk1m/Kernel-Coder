import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Max Pooling 1D
max_pool_1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_pool_1d_kernel(const float* input, float* output, int batch_size, int features, int sequence_length, int kernel_size, int stride, int padding) {
    int b = blockIdx.y; // Batch index
    int f = blockIdx.z; // Feature index
    int s = blockIdx.x * blockDim.x + threadIdx.x; // Sequence index

    if (s >= sequence_length) {
        return;
    }

    int start_idx = s * stride - padding;
    int end_idx = start_idx + kernel_size;

    float max_val = -FLT_MAX;
    for (int i = start_idx; i < end_idx && i < sequence_length; ++i) {
        max_val = fmaxf(max_val, input[b * features * sequence_length + f * sequence_length + i]);
    }

    int output_idx = b * features * ((sequence_length + stride - 1) / stride) + f * ((sequence_length + stride - 1) / stride) + s / stride;
    output[output_idx] = max_val;
}

torch::Tensor max_pool_1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    auto batch_size = input.size(0);
    auto features = input.size(1);
    auto sequence_length = input.size(2);

    auto output = torch::zeros({batch_size, features, (sequence_length + stride - 1) / stride}, input.options());

    const int block_size = 256;
    const int num_blocks = (sequence_length + block_size - 1) / block_size;

    dim3 grid((sequence_length + block_size - 1) / block_size, batch_size, features);
    dim3 block(block_size, 1, 1);

    max_pool_1d_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, features, sequence_length, kernel_size, stride, padding);

    return output;
}
"""

max_pool_1d_cpp_source = (
    "torch::Tensor max_pool_1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"
)

# Compile the inline CUDA code for Max Pooling 1D
max_pool_1d = load_inline(
    name="max_pool_1d",
    cpp_sources=max_pool_1d_cpp_source,
    cuda_sources=max_pool_1d_source,
    functions=["max_pool_1d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 1D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 1D applied, shape (batch_size, num_features, output_sequence_length).
        """
        return max_pool_1d.max_pool_1d_cuda(x, self.kernel_size, self.stride, self.padding)

# Example usage
if __name__ == "__main__":
    batch_size = 64
    features = 192
    sequence_length = 65536

    kernel_size = 8
    stride = 1
    padding = 4
    dilation = 3

    model = ModelNew(kernel_size, stride, padding, dilation)
    inputs = get_inputs()

    output = model(inputs[0])
    print(output.shape)  # Should print: torch.Size([64, 192, 8192])