import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

subtract_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void subtract_mean_kernel(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W
) {
    int n = blockIdx.x / C;
    int c = blockIdx.x % C;
    int tid = threadIdx.x;
    int S = D * H * W;

    __shared__ float shared_sums[256]; // Assuming block size 256

    float sum = 0.0f;
    for (int idx = tid; idx < S; idx += blockDim.x) {
        int d = idx / (H * W);
        int hw = idx % (H * W);
        int h = hw / W;
        int w = hw % W;

        int in_offset = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
        sum += input[in_offset];
    }

    shared_sums[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] += shared_sums[tid + s];
        }
        __syncthreads();
    }

    float total_sum = shared_sums[0];
    float mean = total_sum / S;

    __syncthreads();

    for (int idx = tid; idx < S; idx += blockDim.x) {
        int d = idx / (H * W);
        int hw = idx % (H * W);
        int h = hw / W;
        int w = hw % W;

        int in_offset = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
        output[in_offset] = input[in_offset] - mean;
    }
}

torch::Tensor subtract_mean_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    dim3 block(256);
    dim3 grid(N * C);

    subtract_mean_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W
    );

    return output;
}
"""

subtract_mean_cpp_source = (
    "torch::Tensor subtract_mean_cuda(torch::Tensor input);"
)

subtract_mean = load_inline(
    name="subtract_mean",
    cpp_sources=subtract_mean_cpp_source,
    cuda_sources=subtract_mean_source,
    functions=["subtract_mean_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.subtract_mean_cuda = subtract_mean

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.subtract_mean_cuda.subtract_mean_cuda(x)
        return x