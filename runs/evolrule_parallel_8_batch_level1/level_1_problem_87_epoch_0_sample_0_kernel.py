import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

pointwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void pointwise_conv_kernel(
    const float* input_data,
    const float* weight_data,
    const float* bias_data,
    float* output_data,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int H,
    const int W
) {
    int w = blockIdx.x;
    int h = blockIdx.y;
    int c_out = threadIdx.x;

    extern __shared__ float input_shared[];

    int shared_size = batch_size * in_channels;

    // Load input into shared memory:
    for (int i = 0; i < 8; ++i) {
        int global_idx = threadIdx.x * 8 + i;
        if (global_idx < shared_size) {
            int batch = global_idx / in_channels;
            int c_in = global_idx % in_channels;
            int input_offset = batch * in_channels * H * W +
                               c_in * H * W +
                               h * W +
                               w;
            input_shared[global_idx] = input_data[input_offset];
        }
    }
    __syncthreads();

    // Compute for this c_out and all batches:
    for (int batch = 0; batch < batch_size; ++batch) {
        float sum = 0.0f;
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            int idx = batch * in_channels + c_in;
            float input_val = input_shared[idx];
            float weight_val = weight_data[c_out * in_channels + c_in];
            sum += input_val * weight_val;
        }
        if (bias_data != nullptr) {
            sum += bias_data[c_out];
        }
        // Output offset:
        int output_offset = batch * out_channels * H * W +
                            c_out * H * W +
                            h * W +
                            w;
        output_data[output_offset] = sum;
    }
}

torch::Tensor pointwise_conv_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::zeros({batch_size, out_channels, H, W}, input.options());

    int blockDim_x = out_channels;
    dim3 blockDim(blockDim_x);

    dim3 gridDim(W, H);

    size_t shared_size = batch_size * in_channels * sizeof(float);

    pointwise_conv_kernel<<<gridDim, blockDim, shared_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, H, W
    );

    return output;
}
"""

pointwise_conv_cpp_source = (
    "torch::Tensor pointwise_conv_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

pointwise_conv = load_inline(
    name="pointwise_conv",
    cpp_sources=pointwise_conv_cpp_source,
    cuda_sources=pointwise_conv_source,
    functions=["pointwise_conv_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        self.pointwise_conv = pointwise_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_2d = self.weight.view(self.weight.size(0), self.weight.size(1))
        bias = self.bias if self.bias is not None else torch.empty(0, device=x.device)
        return self.pointwise_conv.pointwise_conv_cuda(x, weight_2d, bias)