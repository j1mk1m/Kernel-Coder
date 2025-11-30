import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ float fast_logsumexp(float* data, int n) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum_exp += exp(data[i] - max_val);
    }

    return max_val + log(sum_exp);
}

__global__ void softmax_kernel(const float* logits, float* output, int batch_size, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_features) {
        return;
    }

    int batch_idx = idx / num_features;
    int feature_idx = idx % num_features;

    float* row = &logits[batch_idx * num_features];
    float logsumexp = fast_logsumexp(row, num_features);
    output[idx] = exp(logits[idx] - logsumexp);
}

torch::Tensor softmax_cuda(torch::Tensor logits) {
    auto batch_size = logits.size(0);
    auto num_features = logits.size(1);
    auto output = torch::zeros_like(logits);

    const int block_size = 256;
    const int num_blocks = (batch_size * num_features + block_size - 1) / block_size;

    softmax_kernel<<<num_blocks, block_size>>>(logits.data_ptr<float>(), output.data_ptr<float>(), batch_size, num_features);

    return output;
}
"""

softmax_cpp_source = (
    "torch::Tensor softmax_cuda(torch::Tensor logits);"
)

# Compile the inline CUDA code for softmax
softmax = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax = softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax.softmax_cuda(x)


if __name__ == "__main__":
    # Example usage
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim).cuda()

    model = ModelNew().cuda()
    output = model(x)
    print(output.shape)