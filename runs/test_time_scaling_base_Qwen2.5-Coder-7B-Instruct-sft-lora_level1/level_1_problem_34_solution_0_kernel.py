import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Instance Normalization
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void instance_norm_forward_kernel(const float* input, float* mean, float* var, float* output, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width) {
        return;
    }

    int channel_idx = idx / (height * width);
    int flat_idx = idx % (height * width);

    float sum = 0.0f;
    for (int i = 0; i < height * width; ++i) {
        sum += input[i * channels + channel_idx];
    }

    mean[channel_idx] = sum / (height * width);

    float variance_sum = 0.0f;
    for (int i = 0; i < height * width; ++i) {
        variance_sum += (input[i * channels + channel_idx] - mean[channel_idx]) * (input[i * channels + channel_idx] - mean[channel_idx]);
    }

    var[channel_idx] = variance_sum / (height * width);

    output[idx] = (input[idx] - mean[channel_idx]) / sqrt(var[channel_idx] + 1e-5);
}

torch::Tensor instance_norm_forward_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    auto mean = torch::zeros(channels).to(input.device());
    auto var = torch::zeros(channels).to(input.device());
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * height * width + block_size - 1) / block_size;

    instance_norm_forward_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width);

    return output;
}
"""

instance_norm_cpp_source = (
    "torch::Tensor instance_norm_forward_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Instance Normalization
instance_norm = load_inline(
    name="instance_norm",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that uses custom CUDA operators for Instance Normalization.
    """
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.instance_norm = instance_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.instance_norm.instance_norm_forward_cuda(x)

# Example usage
if __name__ == "__main__":
    model_new = ModelNew(features)
    inputs = get_inputs()
    outputs = model_new(inputs[0])
    print(outputs.shape)  # Should match the input shape