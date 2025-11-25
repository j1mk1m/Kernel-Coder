import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the global parameters
batch_size = 16
channels = 32
depth = 128
height = 128
width = 256
kernel_size = 3
stride = 2
padding = 1

# CUDA kernel code
elementwise_avg_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

__global__ void avg_pool3d_kernel(
    const float* input,
    float* output,
    int kernel_size,
    int stride,
    int padding,
    int batch_size,
    int channels,
    int input_depth,
    int input_height,
    int input_width,
    int out_depth,
    int out_height,
    int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * out_depth * out_height * out_width) {
        return;
    }

    // Compute indices
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int od = (idx / (out_width * out_height)) % out_depth;
    int c = (idx / (out_depth * out_height * out_width)) % channels;
    int n = idx / (channels * out_depth * out_height * out_width);

    int start_d = od * stride;
    int start_h = oh * stride;
    int start_w = ow * stride;

    float sum = 0.0f;

    for (int kd = 0; kd < kernel_size; ++kd) {
        int d_padded = start_d + kd;
        int d = d_padded - padding;
        bool d_valid = (d >= 0) && (d < input_depth);
        for (int kh = 0; kh < kernel_size; ++kh) {
            int h_padded = start_h + kh;
            int h = h_padded - padding;
            bool h_valid = (h >= 0) && (h < input_height);
            for (int kw = 0; kw < kernel_size; ++kw) {
                int w_padded = start_w + kw;
                int w = w_padded - padding;
                bool w_valid = (w >= 0) && (w < input_width);

                if (d_valid && h_valid && w_valid) {
                    int input_offset = n * channels * input_depth * input_height * input_width +
                                      c * input_depth * input_height * input_width +
                                      d * input_height * input_width +
                                      h * input_width +
                                      w;
                    sum += input[input_offset];
                }
            }
        }
    }

    sum /= (kernel_size * kernel_size * kernel_size);

    // Compute output offset
    int output_offset = n * channels * out_depth * out_height * out_width +
                       c * out_depth * out_height * out_width +
                       od * out_height * out_width +
                       oh * out_width +
                       ow;

    output[output_offset] = sum;
}

torch::Tensor avg_pool3d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding
) {
    CHECK_INPUT(input);
    AT_ASSERTM(input.scalar_type() == at::ScalarType::Float, "Input must be float type");

    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);

    int out_depth = (input_depth + 2 * padding - kernel_size) / stride + 1;
    int out_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, channels, out_depth, out_height, out_width}, input.options());

    int total_elements = batch_size * channels * out_depth * out_height * out_width;

    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    avg_pool3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel_size,
        stride,
        padding,
        batch_size,
        channels,
        input_depth,
        input_height,
        input_width,
        out_depth,
        out_height,
        out_width
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}
"""

elementwise_avg_pool3d_cpp_source = (
    "torch::Tensor avg_pool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"
)

# Load the CUDA kernel
avg_pool3d = load_inline(
    name="avg_pool3d",
    cpp_sources=elementwise_avg_pool3d_cpp_source,
    cuda_sources=elementwise_avg_pool3d_source,
    functions=["avg_pool3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        return avg_pool3d(x, self.kernel_size, self.stride, self.padding)

def get_inputs():
    x = torch.rand(batch_size, channels, depth, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]