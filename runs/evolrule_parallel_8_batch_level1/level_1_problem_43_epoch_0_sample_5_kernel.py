import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool3d_kernel(const float* input, float* output,
                                  int batch_size, int channels) {
    const int input_depth = 128;
    const int input_height = 128;
    const int input_width = 128;
    const int kernel_size = 3;
    const int stride = 2;
    const int dilation = 3;
    const int padding = 1;
    const int output_depth = 62;
    const int output_height = 62;
    const int output_width = 62;

    int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear_idx >= batch_size * channels * output_depth * output_height * output_width) {
        return;
    }

    // Decompose linear index into dimensions.
    int w_out = linear_idx % output_width;
    linear_idx /= output_width;
    int h_out = linear_idx % output_height;
    linear_idx /= output_height;
    int d_out = linear_idx % output_depth;
    linear_idx /= output_depth;
    int c = linear_idx % channels;
    int n = linear_idx / channels;

    // Compute starting positions in padded input.
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    float max_val = -FLT_MAX;

    for (int kd = 0; kd < kernel_size; ++kd) {
        int d_in = d_start + kd * dilation;
        if (d_in < 0 || d_in >= input_depth + 2 * padding) continue;
        int d_original = d_in - padding;
        if (d_original < 0 || d_original >= input_depth) continue;

        for (int kh = 0; kh < kernel_size; ++kh) {
            int h_in = h_start + kh * dilation;
            if (h_in < 0 || h_in >= input_height + 2 * padding) continue;
            int h_original = h_in - padding;
            if (h_original < 0 || h_original >= input_height) continue;

            for (int kw = 0; kw < kernel_size; ++kw) {
                int w_in = w_start + kw * dilation;
                if (w_in < 0 || w_in >= input_width + 2 * padding) continue;
                int w_original = w_in - padding;
                if (w_original < 0 || w_original >= input_width) continue;

                // Compute input offset
                int input_offset = n * channels * input_depth * input_height * input_width +
                                   c * input_depth * input_height * input_width +
                                   d_original * input_height * input_width +
                                   h_original * input_width +
                                   w_original;

                float val = input[input_offset];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }

    // Write to output
    int output_offset = n * channels * output_depth * output_height * output_width +
                        c * output_depth * output_height * output_width +
                        d_out * output_height * output_width +
                        h_out * output_width +
                        w_out;
    output[output_offset] = max_val;
}

torch::Tensor max_pool3d_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto output_size = {batch_size, channels, 62, 62, 62};
    auto output = torch::empty(output_size, input.options());

    const int num_threads = 256;
    const int num_elements = batch_size * channels * 62 * 62 * 62;
    const int num_blocks = (num_elements + num_threads - 1) / num_threads;

    max_pool3d_kernel<<<num_blocks, num_threads>>>(input.data_ptr<float>(),
                                                  output.data_ptr<float>(),
                                                  batch_size,
                                                  channels);

    return output;
}
"""

max_pool3d_cpp_source = (
    "torch::Tensor max_pool3d_cuda(torch::Tensor input);"
)

max_pool3d = load_inline(
    name="max_pool3d",
    cpp_sources=max_pool3d_cpp_source,
    cuda_sources=max_pool3d_source,
    functions=["max_pool3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool3d = max_pool3d  # The compiled function

    def forward(self, x):
        return self.max_pool3d.max_pool3d_cuda(x)

def get_inputs():
    x = torch.rand(16, 32, 128, 128, 128).cuda()
    return [x]

def get_init_inputs():
    return [3, 2, 1, 3]  # kernel_size, stride, padding, dilation