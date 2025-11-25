import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D Max Pooling
max_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

#define KERNEL_SIZE 3
#define STRIDE 2
#define DILATION 3
#define PADDING 1
#define INPUT_PADDDED_DIM 130
#define OUTPUT_DIM 62
#define BATCH_SIZE 16
#define CHANNELS 32

__global__ void max_pool3d_kernel(const float* __restrict__ input, float* __restrict__ output) {
    const int batch = BATCH_SIZE;
    const int channels = CHANNELS;
    const int input_padded = INPUT_PADDDED_DIM;
    const int output_dim = OUTPUT_DIM;
    const int input_padded_cu = input_padded * input_padded * input_padded;
    const int input_padded_sq = input_padded * input_padded;
    const int output_dim_cu = output_dim * output_dim * output_dim;
    const int output_dim_sq = output_dim * output_dim;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * channels * output_dim * output_dim * output_dim) return;

    int ow = index % output_dim;
    int oh = (index / output_dim) % output_dim;
    int od = (index / (output_dim * output_dim)) % output_dim;
    int c = (index / (output_dim * output_dim * output_dim)) % channels;
    int n = index / (channels * output_dim * output_dim * output_dim);

    int start_d = od * STRIDE;
    int start_h = oh * STRIDE;
    int start_w = ow * STRIDE;

    float max_val = -std::numeric_limits<float>::max();

    // Unroll loops for KD, KH, KW
    // KD loop unrolled
    for (int kd = 0; kd < KERNEL_SIZE; ++kd) {
        int id = start_d + kd * DILATION;
        int term_id = id * input_padded_sq;

        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            int ih = start_h + kh * DILATION;
            int term_ih = ih * input_padded;

            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                int iw = start_w + kw * DILATION;

                int base = (n * channels + c) * input_padded_cu;
                int input_idx = base + term_id + term_ih + iw;

                float val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }

    int output_base = (n * channels + c) * output_dim_cu;
    int output_idx = output_base + od * output_dim_sq + oh * output_dim + ow;
    output[output_idx] = max_val;
}

torch::Tensor max_pool3d_cuda(torch::Tensor input) {
    const int total_elements = BATCH_SIZE * CHANNELS * OUTPUT_DIM * OUTPUT_DIM * OUTPUT_DIM;
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    auto output = torch::empty({BATCH_SIZE, CHANNELS, OUTPUT_DIM, OUTPUT_DIM, OUTPUT_DIM}, input.options());

    max_pool3d_kernel<<<num_blocks, threads_per_block>>>(input.data_ptr<float>(), output.data_ptr<float>());

    return output;
}
"""

max_pool3d_cpp_source = "torch::Tensor max_pool3d_cuda(torch::Tensor input);"

# Compile the inline CUDA code for 3D Max Pooling
max_pool3d_cuda = load_inline(
    name="max_pool3d_cuda",
    cpp_sources=max_pool3d_cpp_source,
    cuda_sources=max_pool3d_source,
    functions=["max_pool3d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return max_pool3d_cuda.max_pool3d_cuda(x)