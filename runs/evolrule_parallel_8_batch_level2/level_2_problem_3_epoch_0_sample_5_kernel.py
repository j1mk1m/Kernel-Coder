import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

conv_transpose_sum_ln_fuse_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_3D_KERNEL_LOOP(i, n)                            \\
  for (int i = blockIdx.z * blockDim.z + threadIdx.z; i < n; \\
       i += blockDim.z * gridDim.z)

template <typename scalar_t>
__global__ void conv_transpose_3d_sum_ln_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ sum_weight,
    scalar_t* __restrict__ output,
    const int64_t input_depth,
    const int64_t input_height,
    const int64_t input_width,
    const int64_t in_channels,
    const int64_t out_channels,
    const int64_t kernel_d,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_d,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t padding_d,
    const int64_t padding_h,
    const int64_t padding_w,
    const int64_t output_padding_d,
    const int64_t output_padding_h,
    const int64_t output_padding_w,
    const int64_t output_depth,
    const int64_t output_height,
    const int64_t output_width,
    const float epsilon) {

    extern __shared__ scalar_t shared_storage[];
    scalar_t* shared = shared_storage;

    const int64_t batch_size = blockDim.x;
    const int64_t thread_idx = threadIdx.x;

    const int64_t d = blockIdx.x;
    const int64_t h = blockIdx.y;
    const int64_t w = blockIdx.z;

    // Compute output spatial coordinates
    int64_t out_d = d * stride_d - padding_d + output_padding_d;
    int64_t out_h = h * stride_h - padding_h + output_padding_h;
    int64_t out_w = w * stride_w - padding_w + output_padding_w;

    // Initialize output block
    for (int64_t b = thread_idx; b < batch_size; b += blockDim.x) {
        for (int64_t c = 0; c < out_channels; ++c) {
            shared[b * out_channels + c] = 0.0;
        }
    }
    __syncthreads();

    // Iterate over kernel
    for (int64_t kd = 0; kd < kernel_d; ++kd) {
        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                int64_t in_d = out_d - kd;
                int64_t in_h = out_h - kh;
                int64_t in_w = out_w - kw;

                if (in_d < 0 || in_d >= input_depth || 
                    in_h < 0 || in_h >= input_height || 
                    in_w < 0 || in_w >= input_width) {
                    continue;
                }

                // Load input and weight
                for (int64_t b = thread_idx; b < batch_size; b += blockDim.x) {
                    for (int64_t ic = 0; ic < in_channels; ++ic) {
                        scalar_t val = input[b * in_channels * input_depth * input_height * input_width +
                                            ic * input_depth * input_height * input_width +
                                            in_d * input_height * input_width +
                                            in_h * input_width +
                                            in_w];
                        for (int64_t oc = 0; oc < out_channels; ++oc) {
                            scalar_t weight_val = weight[oc * in_channels * kernel_d * kernel_h * kernel_w +
                                                        ic * kernel_d * kernel_h * kernel_w +
                                                        kd * kernel_h * kernel_w +
                                                        kh * kernel_w +
                                                        kw];
                            atomicAdd(&shared[b * out_channels + oc], val * weight_val);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    // Add bias and sum_weight
    for (int64_t b = thread_idx; b < batch_size; b += blockDim.x) {
        for (int64_t c = 0; c < out_channels; ++c) {
            shared[b * out_channels + c] += bias[c] + *sum_weight;
        }
    }
    __syncthreads();

    // Compute mean and variance for layer norm
    for (int64_t b = thread_idx; b < batch_size; b += blockDim.x) {
        float mean = 0.0;
        for (int64_t c = 0; c < out_channels; c++) {
            mean += shared[b * out_channels + c];
        }
        mean /= out_channels;

        float variance = 0.0;
        for (int64_t c = 0; c < out_channels; c++) {
            float dev = shared[b * out_channels + c] - mean;
            variance += dev * dev;
        }
        variance = variance / out_channels + epsilon;

        float inv_std = 1.0 / sqrt(variance);

        for (int64_t c = 0; c < out_channels; c++) {
            shared[b * out_channels + c] = (shared[b * out_channels + c] - mean) * inv_std;
        }
    }
    __syncthreads();

    // Write output to global memory
    for (int64_t b = thread_idx; b < batch_size; b += blockDim.x) {
        for (int64_t c = 0; c < out_channels; ++c) {
            output[b * out_channels * output_depth * output_height * output_width +
                   c * output_depth * output_height * output_width +
                   d * output_height * output_width +
                   h * output_width +
                   w] = shared[b * out_channels + c];
        }
    }
}

torch::Tensor conv_transpose_sum_ln_fused_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor sum_weight,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_d,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_d,
    int64_t padding_h,
    int64_t padding_w,
    int64_t output_padding_d,
    int64_t output_padding_h,
    int64_t output_padding_w,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width,
    float epsilon) {

    auto batch_size = input.size(0);
    auto output_size = {batch_size, out_channels, output_depth, output_height, output_width};
    auto output_tensor = torch::empty(output_size, input.options());

    const int threads_per_block = 1024;
    const dim3 blocks(output_depth, output_height, output_width);
    const dim3 threads(threads_per_block);

    size_t shared_mem_size = batch_size * out_channels * sizeof(float);

    conv_transpose_3d_sum_ln_kernel<float><<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        sum_weight.data_ptr<float>(),
        output_tensor.data_ptr<float>(),
        input_depth, input_height, input_width,
        in_channels, out_channels,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        output_depth, output_height, output_width,
        epsilon);

    return output_tensor;
}

"""

avg_pool_gelu_fuse_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void avg_pool_gelu_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t batch_size,
    const int64_t channels,
    const int64_t in_depth,
    const int64_t in_height,
    const int64_t in_width,
    const int64_t kernel_d,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_d,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_d,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t out_depth,
    const int64_t out_height,
    const int64_t out_width) {

    CUDA_3D_KERNEL_LOOP(index, batch_size * channels * out_depth * out_height * out_width) {
        int w = index % out_width;
        int h = (index / out_width) % out_height;
        int d = (index / (out_width * out_height)) % out_depth;
        int c = (index / (out_width * out_height * out_depth)) % channels;
        int n = index / (channels * out_depth * out_height * out_width);

        float avg_val = 0.0;
        int count = 0;
        for (int kd = -pad_d; kd < kernel_d - pad_d; ++kd) {
            for (int kh = -pad_h; kh < kernel_h - pad_h; ++kh) {
                for (int kw = -pad_w; kw < kernel_w - pad_w; ++kw) {
                    int id = d * stride_d + kd;
                    int ih = h * stride_h + kh;
                    int iw = w * stride_w + kw;

                    if (id >= 0 && id < in_depth &&
                        ih >= 0 && ih < in_height &&
                        iw >= 0 && iw < in_width) {
                        avg_val += input[n * channels * in_depth * in_height * in_width +
                                        c * in_depth * in_height * in_width +
                                        id * in_height * in_width +
                                        ih * in_width +
                                        iw];
                        count++;
                    }
                }
            }
        }
        avg_val /= count;

        // GELU approximation
        float x = avg_val;
        float gelu_val = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
        output[index] = gelu_val;
    }
}

torch::Tensor avg_pool_gelu_fused_cuda(
    torch::Tensor input,
    int64_t kernel_d,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_d,
    int64_t pad_h,
    int64_t pad_w,
    int64_t out_depth,
    int64_t out_height,
    int64_t out_width) {

    auto batch_size = input.size(0);
    auto channels = input.size(1);

    auto output_size = {batch_size, channels, out_depth, out_height, out_width};
    auto output = torch::empty(output_size, input.options());

    const int threads = 256;
    const dim3 blocks(
        (batch_size * channels * out_depth * out_height * out_width + threads - 1) / threads);

    avg_pool_gelu_kernel<float><<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, input.size(2), input.size(3), input.size(4),
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_depth, out_height, out_width);

    return output;
}
"""

# Compile the fused kernels
conv_transpose_sum_ln = load_inline(
    name="conv_transpose_sum_ln",
    cpp_sources="torch::Tensor conv_transpose_sum_ln_fused_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor sum_weight, int64_t input_depth, int64_t input_height, int64_t input_width, int64_t in_channels, int64_t out_channels, int64_t kernel_d, int64_t kernel_h, int64_t kernel_w, int64_t stride_d, int64_t stride_h, int64_t stride_w, int64_t padding_d, int64_t padding_h, int64_t padding_w, int64_t output_padding_d, int64_t output_padding_h, int64_t output_padding_w, int64_t output_depth, int64_t output_height, int64_t output_width, float epsilon);",
    cuda_sources=conv_transpose_sum_ln_fuse_source,
    functions=["conv_transpose_sum_ln_fused_cuda"],
    verbose=True,
)

avg_pool_gelu = load_inline(
    name="avg_pool_gelu",
    cpp_sources="torch::Tensor avg_pool_gelu_fused_cuda(torch::Tensor input, int64_t kernel_d, int64_t kernel_h, int64_t kernel_w, int64_t stride_d, int64_t stride_h, int64_t stride_w, int64_t pad_d, int64_t pad_h, int64_t pad_w, int64_t out_depth, int64_t out_height, int64_t out_width);",
    cuda_sources=avg_pool_gelu_fuse_source,
    functions=["avg_pool_gelu_fused_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        # ConvTranspose3d parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.epsilon = 1e-5  # For LayerNorm

        # AvgPool3d parameters
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_kernel_size  # Assuming stride same as kernel size
        self.pool_padding = 0  # Assuming no padding
        self.pool_output_depth = (norm_shape[0] + self.pool_kernel_size[0] - 1) // self.pool_kernel_size[0]
        self.pool_output_height = (norm_shape[1] + self.pool_kernel_size[1] - 1) // self.pool_kernel_size[1]
        self.pool_output_width = (norm_shape[2] + self.pool_kernel_size[2] - 1) // self.pool_kernel_size[2]

    def forward(self, x):
        # Compute necessary dimensions for fused kernels
        input_depth = x.size(2)
        input_height = x.size(3)
        input_width = x.size(4)
        output_depth = (input_depth - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0] + self.output_padding[0]
        output_height = (input_height - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1] + self.output_padding[1]
        output_width = (input_width - 1) * self.stride[2] - 2 * self.padding[2] + self.kernel_size[2] + self.output_padding[2]

        # Fused ConvTranspose, Sum, LayerNorm
        conv_transpose_out = conv_transpose_sum_ln.conv_transpose_sum_ln_fused_cuda(
            x,
            self.weight,
            self.bias,
            self.sum_weight,
            input_depth, input_height, input_width,
            self.in_channels, self.out_channels,
            *self.kernel_size,
            *self.stride,
            *self.padding,
            *self.output_padding,
            output_depth, output_height, output_width,
            self.epsilon
        )

        # Fused AvgPool and GELU
        pool_out_depth = (output_depth - self.pool_kernel_size[0]) // self.pool_stride[0] + 1
        pool_out_height = (output_height - self.pool_kernel_size[1]) // self.pool_stride[1] + 1
        pool_out_width = (output_width - self.pool_kernel_size[2]) // self.pool_stride[2] + 1
        return avg_pool_gelu.avg_pool_gelu_fused_cuda(
            conv_transpose_out,
            *self.pool_kernel_size,
            *self.pool_stride,
            self.pool_padding,
            self.pool_padding,
            self.pool_padding,
            pool_out_depth, pool_out_height, pool_out_width
        )

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]